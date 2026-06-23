// SPDX-FileCopyrightText: © 2025 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Runtime smart-search worker. Embeds a free-text query with the CLIP text tower
// and ranks the library's stored image embeddings by cosine similarity, then
// folds in any photos of named people whose name matches the query (combined
// search). The text encoder + tokenizer are loaded lazily on the first query and
// kept in memory so subsequent queries are fast.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::result::Result::Ok;

use relm4::Worker;
use relm4::prelude::*;
use tracing::{error, info};

use fotema_core::PersonId;
use fotema_core::PictureId;
use fotema_core::machine_learning::clip::{self, ClipTextEmbedder};
use fotema_core::people;
use fotema_core::search;

/// Maximum number of ranked results returned to the UI.
const MAX_RESULTS: usize = 500;

/// Score assigned to a photo matched via a named person. Kept at/above the CLIP
/// cosine range so an explicit name match always sorts above content matches.
const NAME_MATCH_SCORE: f32 = 1.0;

#[derive(Debug)]
pub enum ClipSearchTaskInput {
    /// Run a search for the given query text.
    Query(String),
}

#[derive(Debug)]
pub enum ClipSearchTaskOutput {
    /// Ranked results (picture id + score, highest first) for the given query.
    Results(String, Vec<(PictureId, f32)>),
}

pub struct ClipSearchTask {
    cache_dir: PathBuf,
    search_repo: search::Repository,
    people_repo: people::Repository,

    /// Lazily-loaded text encoder; kept warm between queries.
    text_embedder: Option<ClipTextEmbedder>,
}

impl ClipSearchTask {
    /// Load the text encoder if not already loaded.
    fn ensure_embedder(&mut self) -> anyhow::Result<&mut ClipTextEmbedder> {
        if self.text_embedder.is_none() {
            let (textual, tokenizer) = clip::ensure_text_model(&self.cache_dir)?;
            self.text_embedder = Some(ClipTextEmbedder::new(&textual, &tokenizer)?);
        }
        Ok(self.text_embedder.as_mut().unwrap())
    }

    /// Detect known person names appearing as substrings of the query (longest
    /// first, so "Anna Plenk" wins over "Anna"). Returns the matched person ids
    /// and the residual scene text (query with the matched names removed). All
    /// lowercased.
    fn extract_people(&self, query: &str) -> (Vec<PersonId>, String) {
        let mut people = self.people_repo.all_people().unwrap_or_default();
        people.retain(|p| !p.name.trim().is_empty());
        people.sort_by_key(|p| std::cmp::Reverse(p.name.len()));

        let mut residual = query.to_lowercase();
        let mut matched = Vec::new();
        for p in people {
            let name_lc = p.name.to_lowercase();
            if residual.contains(&name_lc) {
                matched.push(p.person_id);
                residual = residual.replacen(&name_lc, " ", 1);
            }
        }
        (matched, residual.trim().to_string())
    }

    /// Union of all photos of the given people (by inner i64 id).
    fn person_picture_set(&self, person_ids: &[PersonId]) -> HashSet<i64> {
        let mut set = HashSet::new();
        for id in person_ids {
            if let Ok(picture_ids) = self.people_repo.find_pictures_for_person(*id) {
                for picture_id in picture_ids {
                    set.insert(picture_id.id());
                }
            }
        }
        set
    }

    /// Rank stored image embeddings against a text query by cosine similarity.
    /// `restrict` optionally limits the candidate set (for the combined
    /// person+scene search).
    fn clip_rank(
        &mut self,
        text: &str,
        restrict: Option<&HashSet<i64>>,
    ) -> anyhow::Result<Vec<(PictureId, f32)>> {
        let query_vec = self.ensure_embedder()?.embedding(text)?;
        let embeddings = self.search_repo.all_clip_embeddings(clip::MODEL_NAME)?;
        let mut ranked: Vec<(PictureId, f32)> = embeddings
            .into_iter()
            .filter(|(pid, _)| restrict.is_none_or(|set| set.contains(&pid.id())))
            .map(|(pid, emb)| (pid, clip::cosine(&query_vec, &emb)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(MAX_RESULTS);
        Ok(ranked)
    }

    fn search(&mut self, query: &str) -> anyhow::Result<Vec<(PictureId, f32)>> {
        // Combined search: if the query contains a known person's full name, treat
        // the rest as a scene description and intersect that person's photos with
        // the CLIP results (e.g. "Anna Plenk am Strand" → Anna's beach photos).
        let (person_ids, scene) = self.extract_people(query);

        if !person_ids.is_empty() {
            let pics = self.person_picture_set(&person_ids);
            if scene.is_empty() {
                // Pure person search: all their photos.
                let mut ranked: Vec<(PictureId, f32)> = pics
                    .into_iter()
                    .map(|id| (PictureId::new(id), NAME_MATCH_SCORE))
                    .collect();
                ranked.truncate(MAX_RESULTS);
                return Ok(ranked);
            }
            // Person + scene: CLIP-rank the scene within that person's photos.
            return self.clip_rank(&scene, Some(&pics));
        }

        // No recognised full name. Fall back to: partial name matches (so typing
        // part of a name still works) unioned with whole-query CLIP content search.
        let mut scores: HashMap<i64, f32> = HashMap::new();
        let partial = self
            .people_repo
            .find_people_by_name_like(query)
            .unwrap_or_default();
        for person_id in partial {
            if let Ok(picture_ids) = self.people_repo.find_pictures_for_person(person_id) {
                for picture_id in picture_ids {
                    scores.insert(picture_id.id(), NAME_MATCH_SCORE);
                }
            }
        }

        let query_vec = self.ensure_embedder()?.embedding(query)?;
        let embeddings = self.search_repo.all_clip_embeddings(clip::MODEL_NAME)?;
        for (picture_id, embedding) in embeddings {
            let score = clip::cosine(&query_vec, &embedding);
            scores
                .entry(picture_id.id())
                .and_modify(|s| {
                    if score > *s {
                        *s = score;
                    }
                })
                .or_insert(score);
        }

        let mut ranked: Vec<(PictureId, f32)> = scores
            .into_iter()
            .map(|(id, score)| (PictureId::new(id), score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(MAX_RESULTS);
        Ok(ranked)
    }
}

impl Worker for ClipSearchTask {
    type Init = (PathBuf, search::Repository, people::Repository);
    type Input = ClipSearchTaskInput;
    type Output = ClipSearchTaskOutput;

    fn init((cache_dir, search_repo, people_repo): Self::Init, _sender: ComponentSender<Self>) -> Self {
        ClipSearchTask {
            cache_dir,
            search_repo,
            people_repo,
            text_embedder: None,
        }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        match msg {
            ClipSearchTaskInput::Query(query) => {
                let trimmed = query.trim();
                if trimmed.is_empty() {
                    let _ = sender.output(ClipSearchTaskOutput::Results(query, Vec::new()));
                    return;
                }
                info!("Smart search: {:?}", trimmed);
                match self.search(trimmed) {
                    Ok(results) => {
                        info!("Smart search returned {} results", results.len());
                        let _ = sender.output(ClipSearchTaskOutput::Results(query, results));
                    }
                    Err(e) => {
                        error!("Smart search failed: {:?}", e);
                        let _ = sender.output(ClipSearchTaskOutput::Results(query, Vec::new()));
                    }
                }
            }
        }
    }
}
