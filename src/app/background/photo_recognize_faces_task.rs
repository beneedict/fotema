// SPDX-FileCopyrightText: © 2024 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

use anyhow::*;
use rayon::prelude::*;
use relm4::Reducer;
use relm4::Worker;
use relm4::prelude::*;
use std::path::PathBuf;
use std::result::Result::Ok;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{error, info};

use fotema_core::machine_learning::face_recognizer::FaceEmbedder;
use fotema_core::people;
use fotema_core::photo;
use fotema_core::photo::detection_size::DetectionSize;
use fotema_core::photo::model::Orientation;
use fotema_core::thumbnailify::Thumbnailer;
use fotema_core::{FlatpakPathBuf, PictureId};

use crate::app::SettingsState;
use crate::app::components::progress_monitor::{ProgressMonitor, ProgressMonitorInput, TaskName};

/// Cosine similarity above which an unknown face is taken to be a named person.
/// ArcFace separates identities well; this leans towards precision.
const COSINE_THRESHOLD: f32 = 0.42;

#[derive(Debug)]
pub enum PhotoRecognizeFacesTaskInput {
    Start,

    /// Re-read XMP person tags for the whole library, then run recognition.
    RescanFaceTags,
}

#[derive(Debug)]
pub enum PhotoRecognizeFacesTaskOutput {
    // Face recognition has started.
    Started,

    // Face recognition has completed
    Completed,
}

#[derive(Clone)]
pub struct PhotoRecognizeFacesTask {
    // Stop flag
    stop: Arc<AtomicBool>,

    // Danger! Don't hold the repo mutex for too long as it blocks viewing images.
    repo: people::Repository,

    // Used to read picture paths for importing names from XMP metadata.
    photo_repo: photo::Repository,

    progress_monitor: Arc<Reducer<ProgressMonitor>>,

    settings_state: SettingsState,

    cache_dir: PathBuf,

    // Built once from cache_dir, so the thumbnail path is not recomputed per face.
    thumbnailer: Thumbnailer,
}

impl PhotoRecognizeFacesTask {
    fn recognize(&self, sender: ComponentSender<Self>) -> Result<()> {
        let start = std::time::Instant::now();

        // Import names already present in the photos' metadata before computing
        // embeddings, so imported faces drop out of the unnamed set. XMP face
        // tags are optional; when they are off, do not touch the sidecars.
        if self.settings_state.read().xmp_face_tags {
            match self.import_face_tags() {
                Ok(n) if n > 0 => info!("Imported {} face name(s) from photo metadata.", n),
                Ok(_) => {}
                Err(e) => error!("Face-tag import failed: {:?}", e),
            }
        } else {
            info!("XMP face tags are disabled; skipping name import");
        }

        // Named reference faces and the current embeddings in the DB. The
        // recognition pass compares these vectors; it does not run inference.
        let references = self.repo.find_recognition_references()?;

        // All faces (named or not) lacking a current ArcFace embedding. Covers
        // the one-time recompute after switching from SFace.
        let need_embedding = self.repo.find_faces_needing_embedding()?;

        info!(
            "Face recognition: {} faces need embeddings, {} reference faces",
            need_embedding.len(),
            references.len()
        );

        if need_embedding.is_empty() && references.is_empty() {
            let _ = sender.output(PhotoRecognizeFacesTaskOutput::Completed);
            return Ok(());
        }

        let _ = sender.output(PhotoRecognizeFacesTaskOutput::Started);

        // Cap concurrency: each worker thread loads its own ~166 MB ArcFace net,
        // so a bounded pool keeps memory (and GPU memory) in check.
        if !need_embedding.is_empty() {
            let (sface_path, arcface_path) = FaceEmbedder::ensure_models(&self.cache_dir)?;

            self.progress_monitor.emit(ProgressMonitorInput::Start(
                TaskName::RecognizeFaces,
                need_embedding.len(),
            ));

            let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
            pool.install(|| {
                need_embedding
                    .into_par_iter()
                    .take_any_while(|_| !self.stop.load(Ordering::Relaxed))
                    .for_each_init(
                        || FaceEmbedder::new(&sface_path, &arcface_path).ok(),
                        |embedder, face| {
                            if let Some(embedder) = embedder.as_mut() {
                                match embedder.embedding(&face) {
                                    Ok(embedding) => {
                                        if let Err(e) =
                                            self.repo.store_face_embedding(face.face_id, &embedding)
                                        {
                                            error!(
                                                "Failed storing embedding for face {}: {:?}",
                                                face.face_id, e
                                            );
                                        }
                                    }
                                    Err(e) => error!(
                                        "Failed computing embedding for face {}: {:?}",
                                        face.face_id, e
                                    ),
                                }
                            }
                            self.progress_monitor.emit(ProgressMonitorInput::Advance);
                        },
                    );
            });

            self.progress_monitor.emit(ProgressMonitorInput::Complete);
        }

        if !references.is_empty() {
            self.recognize_from_embeddings(&references)?;
        }

        info!(
            "Recognized/embedded faces in {} seconds.",
            start.elapsed().as_secs()
        );

        let _ = sender.output(PhotoRecognizeFacesTaskOutput::Completed);

        Ok(())
    }

    /// Unnamed faces grouped by their picture (id, path, orientation), for
    /// pictures whose XMP tags have not yet been matched.
    fn unnamed_faces_by_picture(
        &self,
    ) -> Result<
        Vec<(
            PictureId,
            FlatpakPathBuf,
            Option<Orientation>,
            Vec<fotema_core::people::model::DetectedFace>,
        )>,
    > {
        let mut groups: Vec<(
            PictureId,
            FlatpakPathBuf,
            Option<Orientation>,
            Vec<fotema_core::people::model::DetectedFace>,
        )> = Vec::new();
        for (picture_id, path, orientation, face) in
            self.photo_repo.find_unnamed_faces_with_pictures()?
        {
            match groups.last_mut() {
                Some(last) if last.0 == picture_id => last.3.push(face),
                _ => groups.push((picture_id, path, orientation, vec![face])),
            }
        }
        Ok(groups)
    }

    /// Pair named regions with detected faces and add the names as unconfirmed
    /// person assignments.
    fn assign_regions(
        faces: Vec<fotema_core::people::model::DetectedFace>,
        regions: Vec<photo::face_tags::FaceTag>,
        size_of: &dyn Fn(&fotema_core::people::model::DetectedFace) -> Option<(f32, f32)>,
        people_repo: &mut people::Repository,
        negatives: &std::collections::HashMap<i64, std::collections::HashSet<i64>>,
    ) -> usize {
        if regions.is_empty() || faces.is_empty() {
            return 0;
        }

        let pairs = Self::pair_regions_with_faces(faces, regions, size_of);
        let mut named = 0usize;
        for (region, face) in pairs.iter() {
            let name = region.name.trim();
            if name.is_empty() {
                continue;
            }
            let result = match people_repo.find_person_id_by_name(name) {
                Ok(Some(person_id)) => {
                    // Respect a prior "not this person" correction.
                    if negatives
                        .get(&face.face_id.id())
                        .is_some_and(|rejected| rejected.contains(&person_id.id()))
                    {
                        continue;
                    }
                    people_repo.mark_as_person_unconfirmed(face.face_id, person_id)
                }
                Ok(None) => people_repo.add_person_unconfirmed(face.face_id, name),
                Err(e) => {
                    error!("Looking up person '{}' failed: {:?}", name, e);
                    continue;
                }
            };
            match result {
                Ok(()) => named += 1,
                Err(e) => error!(
                    "Importing face {} as '{}' failed: {:?}",
                    face.face_id, name, e
                ),
            }
        }
        named
    }

    /// Pair each named region with one detected face, preferring a region/face
    /// overlap when the sidecar contains full areas. Files with only a
    /// horizontal centre use the careful left-to-right rule.
    fn pair_regions_with_faces(
        mut faces: Vec<fotema_core::people::model::DetectedFace>,
        mut regions: Vec<photo::face_tags::FaceTag>,
        size_of: &dyn Fn(&fotema_core::people::model::DetectedFace) -> Option<(f32, f32)>,
    ) -> Vec<(
        photo::face_tags::FaceTag,
        fotema_core::people::model::DetectedFace,
    )> {
        const MIN_OVERLAP: f32 = 0.3;

        let with_area = regions.iter().all(|r| r.area.is_some());
        let face_areas: Vec<Option<photo::face_tags::TagArea>> =
            faces.iter().map(|f| Self::face_area(f, size_of)).collect();

        if with_area && face_areas.iter().any(Option::is_some) {
            let mut pairs = Vec::new();
            let mut used = vec![false; faces.len()];

            for region in &regions {
                let Some(area) = region.area else { continue };
                let mut best: Option<(usize, f32)> = None;
                for (i, face_area) in face_areas.iter().enumerate() {
                    if used[i] {
                        continue;
                    }
                    let Some(face_area) = face_area else {
                        continue;
                    };
                    let overlap = area.iou(*face_area);
                    if overlap >= MIN_OVERLAP
                        && best.is_none_or(|(_, best_overlap)| overlap > best_overlap)
                    {
                        best = Some((i, overlap));
                    }
                }
                if let Some((i, _)) = best {
                    used[i] = true;
                    pairs.push((region.clone(), faces[i].clone()));
                }
            }
            return pairs;
        }

        if regions.len() != faces.len() {
            return Vec::new();
        }

        let by_x = |a: f32, b: f32| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal);
        regions.sort_by(|a, b| by_x(a.center_x, b.center_x));
        faces.sort_by(|a, b| {
            by_x(
                a.bounds.x + a.bounds.width / 2.0,
                b.bounds.x + b.bounds.width / 2.0,
            )
        });
        regions.into_iter().zip(faces).collect()
    }

    fn face_area(
        face: &fotema_core::people::model::DetectedFace,
        size_of: &dyn Fn(&fotema_core::people::model::DetectedFace) -> Option<(f32, f32)>,
    ) -> Option<photo::face_tags::TagArea> {
        let (width, height) = size_of(face)?;
        photo::face_tags::TagArea::from_pixel_bounds(
            face.bounds.x,
            face.bounds.y,
            face.bounds.width,
            face.bounds.height,
            width,
            height,
        )
    }

    fn detection_size_for(
        &self,
        path: &FlatpakPathBuf,
        orientation: Option<Orientation>,
    ) -> impl Fn(&fotema_core::people::model::DetectedFace) -> Option<(f32, f32)> + use<> {
        let size = DetectionSize::new(&self.thumbnailer, path, orientation);
        move |face: &fotema_core::people::model::DetectedFace| size.size_of(face.is_source_original)
    }

    /// Read XMP names and assign them to the matching unnamed faces. Runs at
    /// most once per picture unless the user triggers a re-scan.
    fn import_face_tags(&self) -> Result<usize> {
        let groups = self.unnamed_faces_by_picture()?;
        if groups.is_empty() {
            return Ok(0);
        }

        let mut people_repo = self.repo.clone();
        let negatives = self.repo.find_negative_associations().unwrap_or_default();
        let mut processed: Vec<PictureId> = Vec::with_capacity(groups.len());
        let mut imported = 0usize;

        for (picture_id, path, orientation, faces) in groups {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            processed.push(picture_id);
            let regions = photo::face_tags::read_face_tags(&path.sandbox_path);
            let size_of = self.detection_size_for(&path, orientation);
            imported += Self::assign_regions(faces, regions, &size_of, &mut people_repo, &negatives);
        }

        if !processed.is_empty() {
            if let Err(e) = self.photo_repo.clone().mark_face_tags_imported(&processed) {
                error!("Failed marking pictures as tag-imported: {:?}", e);
            }
        }
        Ok(imported)
    }

    fn rescan_face_tags(&self) -> Result<usize> {
        if !self.settings_state.read().xmp_face_tags {
            return Ok(0);
        }
        let _ = self.photo_repo.clone().reset_face_tags_imported();
        self.import_face_tags()
    }

    /// Match unnamed faces to named people using only the embeddings already
    /// stored in the database. Pure cosine comparison: no model load, image
    /// read or inference per face.
    fn recognize_from_embeddings(
        &self,
        references: &[people::RecognitionReference],
    ) -> Result<usize> {
        let min_recognized_at = references
            .iter()
            .map(|reference| reference.recognized_at)
            .min()
            .expect("references is not empty");

        let unnamed = self
            .repo
            .find_unnamed_with_embeddings()?
            .into_iter()
            .filter(|face| face.detected_at > min_recognized_at)
            .collect::<Vec<_>>();

        // "This face is NOT person X" corrections, so a rejected pair is never
        // re-assigned by a later recognition pass.
        let negatives = self.repo.find_negative_associations().unwrap_or_default();

        let assignments: Vec<(fotema_core::FaceId, fotema_core::PersonId)> = unnamed
            .par_iter()
            .take_any_while(|_| !self.stop.load(Ordering::Relaxed))
            .filter_map(|face| {
                let rejected = negatives.get(&face.face_id.id());
                let mut best: Option<(fotema_core::PersonId, f32)> = None;
                for reference in references {
                    // Only match against people known before this face was detected.
                    if reference.recognized_at > face.detected_at {
                        continue;
                    }
                    if rejected.is_some_and(|set| set.contains(&reference.person_id.id())) {
                        continue;
                    }
                    let score = people::Repository::cosine_normalized(
                        &face.embedding,
                        &reference.embedding,
                    );
                    if score >= COSINE_THRESHOLD
                        && best.is_none_or(|(_, best_score)| score > best_score)
                    {
                        best = Some((reference.person_id, score));
                    }
                }
                best.map(|(person_id, score)| {
                    info!(
                        "Face {} looks like person {} (score {:.3})",
                        face.face_id, person_id, score
                    );
                    (face.face_id, person_id)
                })
            })
            .collect();

        let count = assignments.len();
        let mut repo = self.repo.clone();
        for (face_id, person_id) in assignments {
            if let Err(e) = repo.mark_as_person_unconfirmed(face_id, person_id) {
                error!("Failed marking face {} as person: {:?}", face_id, e);
            }
        }

        let mut person_ids: Vec<i64> = references
            .iter()
            .map(|reference| reference.person_id.id())
            .collect();
        person_ids.sort_unstable();
        person_ids.dedup();
        for id in person_ids {
            if let Err(e) = repo.mark_face_recognition_complete(fotema_core::PersonId::new(id)) {
                error!(
                    "Failed marking recognition complete for person {}: {:?}",
                    id, e
                );
            }
        }

        Ok(count)
    }
}

impl Worker for PhotoRecognizeFacesTask {
    type Init = (
        Arc<AtomicBool>,
        PathBuf,
        people::Repository,
        photo::Repository,
        Thumbnailer,
        SettingsState,
        Arc<Reducer<ProgressMonitor>>,
    );
    type Input = PhotoRecognizeFacesTaskInput;
    type Output = PhotoRecognizeFacesTaskOutput;

    fn init(
        (stop, cache_dir, repo, photo_repo, thumbnailer, settings_state, progress_monitor): Self::Init,
        _sender: ComponentSender<Self>,
    ) -> Self {
        PhotoRecognizeFacesTask {
            stop,
            cache_dir,
            repo,
            photo_repo,
            thumbnailer,
            settings_state,
            progress_monitor,
        }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        match msg {
            PhotoRecognizeFacesTaskInput::Start => {
                info!("Recognizing photo faces...");
                let this = self.clone();

                // Avoid runtime panic from calling block_on
                rayon::spawn(move || {
                    if let Err(e) = this.recognize(sender.clone()) {
                        error!("Failed to recognize photo faces: {}", e);
                        let _ = sender.output(PhotoRecognizeFacesTaskOutput::Completed);
                    }
                });
            }
            PhotoRecognizeFacesTaskInput::RescanFaceTags => {
                info!("Rescanning XMP face tags...");
                let this = self.clone();
                rayon::spawn(move || {
                    if let Err(e) = this.rescan_face_tags() {
                        error!("Failed to rescan XMP face tags: {}", e);
                    }
                    if let Err(e) = this.recognize(sender.clone()) {
                        error!("Failed to recognize photo faces: {}", e);
                        let _ = sender.output(PhotoRecognizeFacesTaskOutput::Completed);
                    }
                });
            }
        };
    }
}
