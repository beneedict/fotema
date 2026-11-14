// SPDX-FileCopyrightText: © 2024 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

use crate::photo::model::PictureId;

use crate::machine_learning::face_extractor;
use crate::path_encoding;
use crate::people::FaceId;
use crate::people::FaceToMigrate;
use crate::people::MigratedFace;
use crate::people::PersonId;
use crate::people::model;
use crate::people::model::PersonForRecognition;
use crate::people::model::Rect;

use anyhow::*;
use chrono::{DateTime, Utc};
use rusqlite;
use rusqlite::Row;
use rusqlite::params;
use std::path::{Path, PathBuf};
use std::result::Result::Ok;
use std::sync::{Arc, Mutex};
use tracing::warn;

/// A confirmed, named reference face together with its stored ArcFace embedding
/// (already L2-normalised) and its person's last-recognition timestamp. The
/// lightweight recognition pass compares unnamed faces against these stored
/// embeddings instead of recomputing them with the model.
#[derive(Debug, Clone)]
pub struct RecognitionReference {
    pub person_id: PersonId,
    pub recognized_at: DateTime<Utc>,
    pub embedding: Vec<f32>,
}

/// An unnamed, non-ignored face with its stored (L2-normalised) ArcFace
/// embedding and detection time.
#[derive(Debug, Clone)]
pub struct UnnamedEmbedding {
    pub face_id: FaceId,
    pub detected_at: DateTime<Utc>,
    pub embedding: Vec<f32>,
}

/// Minimum detected-face size (shorter side, in source-image pixels) for a face
/// to appear in the "unknown people" grid and be considered for recognition.
/// Faces smaller than this are too low-resolution to identify reliably and only
/// add noise (weak embeddings, spurious clusters). Tunable; raise it to be
/// stricter. Existing faces are only filtered from queries, never deleted, so
/// lowering it brings them back.
pub const MIN_FACE_PX: f64 = 24.0;

/// The statement that clears the sidecar marker of the picture that holds a
/// face. The next run of the export task then writes the sidecar of that
/// picture again.
const CLEAR_EXPORT_BY_FACE: &str = "UPDATE pictures SET face_tags_exported = 0
     WHERE picture_id IN (SELECT picture_id FROM pictures_faces WHERE face_id = ?1)";

/// The same, for every picture that holds a face of a person.
const CLEAR_EXPORT_BY_PERSON: &str = "UPDATE pictures SET face_tags_exported = 0
     WHERE picture_id IN (SELECT picture_id FROM pictures_faces WHERE person_id = ?1)";

/// Repository of people data.
/// Repository is backed by a Sqlite database.
#[derive(Debug, Clone)]
pub struct Repository {
    /// Cache direcctory
    cache_dir_base_path: PathBuf,

    /// Data directory
    data_dir_base_path: PathBuf,

    /// Connection to backing Sqlite database.
    con: Arc<Mutex<rusqlite::Connection>>,
}

impl Repository {
    /// Builds a Repository and creates operational tables.
    pub fn open(
        cache_dir_base_path: &Path,
        data_dir_base_path: &Path,
        con: Arc<Mutex<rusqlite::Connection>>,
    ) -> Result<Repository> {
        let cache_dir_base_path = PathBuf::from(cache_dir_base_path);
        let data_dir_base_path = PathBuf::from(data_dir_base_path);

        let repo = Repository {
            cache_dir_base_path,
            data_dir_base_path,
            con,
        };

        Ok(repo)
    }

    /// Deletes faces for a picture so a picture can be re-scanned and new faces.
    /// We must delete before re-scanning a picture for faces to avoid a unique constraint
    /// violation on the bounds_path.
    pub fn delete_faces(&self, picture_id: PictureId) -> Result<()> {
        let con = self.con.lock().unwrap();

        // The faces of this picture are detected again, so the sidecar is out of date.
        con.execute(
            "UPDATE pictures SET face_tags_exported = 0 WHERE picture_id = ?1",
            [picture_id.id()],
        )?;

        let mut stmt = con.prepare(
            "DELETE FROM pictures_faces
            WHERE pictures_faces.picture_id = ?1",
        )?;

        stmt.execute([picture_id.id()])?;

        Ok(())
    }

    /// Finds faces and people for the thumbnail bar.
    /// Faces are ordered from left to right, top to bottom.
    pub fn find_faces(
        &self,
        picture_id: &PictureId,
    ) -> Result<Vec<(model::Face, Option<model::Person>)>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT
                faces.face_id AS face_id,
                faces.thumbnail_path AS face_thumbnail_path,
                people.person_id AS person_id,
                people.name AS person_name,
                person_face.thumbnail_path AS person_thumbnail_path
            FROM pictures_faces AS faces
            LEFT OUTER JOIN people USING (person_id)
            LEFT OUTER JOIN pictures_faces AS person_face
                ON (person_face.person_id = faces.person_id AND person_face.is_thumbnail = TRUE)
            WHERE faces.picture_id = ?1 AND faces.is_ignored = FALSE
            ORDER BY faces.nose_x ASC, faces.nose_y ASC",
        )?;

        let result = stmt
            .query_map([picture_id.id()], |row| self.to_face_and_person(row))?
            .flatten()
            .collect();

        Ok(result)
    }

    /// All detected faces not yet assigned to a person (and not ignored), across
    /// the whole library, grouped by similarity (via stored SFace embeddings)
    /// and ordered so the most frequently occurring unknown person comes first.
    /// Faces without an embedding yet are appended at the end. Powers the
    /// "unknown people" overview.
    pub fn find_unnamed_faces(&self) -> Result<Vec<model::Face>> {
        let rows: Vec<(model::Face, Option<Vec<f32>>)> = {
            let con = self.con.lock().unwrap();
            let mut stmt = con.prepare(
                "SELECT
                    faces.face_id AS face_id,
                    faces.thumbnail_path AS face_thumbnail_path,
                    faces.embedding AS embedding
                FROM pictures_faces AS faces
                WHERE faces.person_id IS NULL AND faces.is_ignored = FALSE
                    AND faces.bounds_width >= ?1
                    AND faces.bounds_height >= ?2
                ORDER BY faces.face_id",
            )?;

            let data_dir = self.data_dir_base_path.clone();
            stmt.query_map([MIN_FACE_PX, MIN_FACE_PX], move |row| {
                let face_id = row.get("face_id").map(FaceId::new)?;
                let thumbnail_path = row
                    .get("face_thumbnail_path")
                    .map(|p: String| data_dir.join(p))?;
                let bytes: Option<Vec<u8>> = row.get("embedding")?;
                let embedding = bytes.map(|b| {
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect::<Vec<f32>>()
                });
                Ok((
                    model::Face {
                        face_id,
                        thumbnail_path,
                    },
                    embedding,
                ))
            })?
            .flatten()
            .collect()
        };

        Ok(Self::cluster_and_order(rows))
    }

    /// Faces the user has hidden via "ignore face" (still unnamed). Powers the
    /// "show ignored faces" view, from which they can be restored. Newest first;
    /// no clustering (these are typically non-faces or strangers).
    pub fn find_ignored_faces(&self) -> Result<Vec<model::Face>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT
                faces.face_id AS face_id,
                faces.thumbnail_path AS face_thumbnail_path
            FROM pictures_faces AS faces
            WHERE faces.person_id IS NULL AND faces.is_ignored = TRUE
            ORDER BY faces.face_id DESC",
        )?;

        let data_dir = self.data_dir_base_path.clone();
        let result: Vec<model::Face> = stmt
            .query_map([], move |row| {
                let face_id = row.get("face_id").map(FaceId::new)?;
                let thumbnail_path = row
                    .get("face_thumbnail_path")
                    .map(|p: String| data_dir.join(p))?;
                Ok(model::Face {
                    face_id,
                    thumbnail_path,
                })
            })?
            .flatten()
            .collect();

        Ok(result)
    }

    /// Greedy clustering of faces by normalised-embedding L2 distance, returning
    /// faces ordered by descending cluster size (most-seen unknown person first),
    /// with faces that have no embedding appended at the end.
    fn cluster_and_order(faces: Vec<(model::Face, Option<Vec<f32>>)>) -> Vec<model::Face> {
        use crate::machine_learning::face_recognizer::EMBEDDING_DIM;
        // L2 distance of L2-normalised ArcFace features below which two faces are
        // considered the same person (≈ cosine 0.40). Tunable.
        const THRESHOLD: f32 = 1.10;

        let mut with_embedding: Vec<(model::Face, Vec<f32>)> = Vec::new();
        let mut without: Vec<model::Face> = Vec::new();
        for (face, embedding) in faces {
            match embedding {
                // Only the current model's dimension; stale embeddings (e.g.
                // old SFace) are treated as missing until recomputed.
                Some(e) if e.len() == EMBEDDING_DIM => {
                    let norm = e.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let e = if norm > 0.0 {
                        e.iter().map(|x| x / norm).collect()
                    } else {
                        e
                    };
                    with_embedding.push((face, e));
                }
                _ => without.push(face),
            }
        }

        let mut clusters: Vec<Vec<(model::Face, Vec<f32>)>> = Vec::new();
        for (face, e) in with_embedding {
            let mut found = None;
            for (i, cluster) in clusters.iter().enumerate() {
                let rep = &cluster[0].1;
                let dist = e
                    .iter()
                    .zip(rep)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                if dist < THRESHOLD {
                    found = Some(i);
                    break;
                }
            }
            match found {
                Some(i) => clusters[i].push((face, e)),
                None => clusters.push(vec![(face, e)]),
            }
        }

        clusters.sort_by(|a, b| b.len().cmp(&a.len()));

        let mut result: Vec<model::Face> = clusters
            .into_iter()
            .flat_map(|c| c.into_iter().map(|(f, _)| f))
            .collect();
        result.extend(without);
        result
    }

    /// Store an SFace embedding (little-endian f32 vector) for a face.
    pub fn store_face_embedding(&self, face_id: FaceId, embedding: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let con = self.con.lock().unwrap();
        con.execute(
            "UPDATE pictures_faces SET embedding = ?1 WHERE face_id = ?2",
            rusqlite::params![bytes, face_id.id()],
        )?;
        Ok(())
    }

    /// Face ids that already have an embedding computed.
    pub fn faces_with_embedding(&self) -> Result<std::collections::HashSet<i64>> {
        let con = self.con.lock().unwrap();
        // Only count embeddings of the current model's dimension (512 floats =
        // 2048 bytes). Older 128-d SFace embeddings are ignored so they get
        // recomputed with the new ArcFace model.
        let mut stmt =
            con.prepare("SELECT face_id FROM pictures_faces WHERE length(embedding) = 2048")?;
        let set = stmt
            .query_map([], |row| row.get::<_, i64>("face_id"))?
            .flatten()
            .collect();
        Ok(set)
    }

    pub fn ignore_unknown_faces(&mut self, picture_id: PictureId) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        {
            let mut stmt = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    is_ignored = TRUE
                WHERE picture_id = ?1 AND person_id IS NULL",
            )?;
            stmt.execute(params![picture_id.id(),])?;
        }

        tx.commit()?;
        Ok(())
    }

    pub fn restore_ignored_faces(&mut self, picture_id: PictureId) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        {
            let mut stmt = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    is_ignored = FALSE
                WHERE picture_id = ?1",
            )?;
            stmt.execute(params![picture_id.id(),])?;
        }

        tx.commit()?;
        Ok(())
    }

    pub fn get_person(&self, person_id: PersonId) -> Result<Option<model::Person>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT
                p.person_id AS person_id,
                p.name AS person_name,
                f.thumbnail_path AS person_thumbnail_path
            FROM people AS p
            LEFT OUTER JOIN pictures_faces AS f
                ON (f.person_id = p.person_id AND f.is_thumbnail = TRUE)
            WHERE person_id = ?1",
        )?;

        let result: Option<model::Person> = stmt
            .query_map([person_id.id()], |row| self.to_person(row))?
            .flatten()
            .nth(0);

        Ok(result)
    }

    pub fn delete_person(&mut self, person_id: PersonId) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        // The person is deleted, so every picture that held a face of them has
        // a sidecar that is out of date.
        tx.execute(CLEAR_EXPORT_BY_PERSON, params![person_id.id()])?;

        {
            // Detach every face so it returns to the "unknown people" overview
            // instead of dangling on a person that no longer exists.
            let mut stmt = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    person_id = NULL,
                    is_confirmed = FALSE,
                    is_thumbnail = FALSE
                WHERE person_id = ?1",
            )?;
            stmt.execute(params![person_id.id(),])?;

            let mut stmt = tx.prepare_cached("DELETE FROM people WHERE person_id = ?1")?;
            stmt.execute(params![person_id.id(),])?;
        }

        tx.commit()?;
        Ok(())
    }

    pub fn rename_person(&mut self, person_id: PersonId, name: &str) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        // Every picture of this person now holds the wrong name in its sidecar.
        tx.execute(CLEAR_EXPORT_BY_PERSON, params![person_id.id()])?;

        {
            let mut stmt = tx.prepare_cached(
                "UPDATE people
                SET
                    name = ?2
                WHERE person_id = ?1",
            )?;
            stmt.execute(params![person_id.id(), name,])?;
        }

        tx.commit()?;
        Ok(())
    }

    pub fn all_people(&self) -> Result<Vec<model::Person>> {
        self.people_by_ignored(false)
    }

    /// People the user has hidden (for the "show ignored people" view, from which
    /// they can be restored). Their data and face assignments are untouched.
    pub fn all_ignored_people(&self) -> Result<Vec<model::Person>> {
        self.people_by_ignored(true)
    }

    /// Shared people query, filtered by whether the person is ignored.
    fn people_by_ignored(&self, is_ignored: bool) -> Result<Vec<model::Person>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT
                p.person_id AS person_id,
                p.name AS person_name,
                p.is_ignored AS is_ignored,
                f.thumbnail_path AS person_thumbnail_path
            FROM people AS p
            LEFT OUTER JOIN pictures_faces AS f
                ON (f.person_id = p.person_id AND f.is_thumbnail = TRUE)
            WHERE p.is_ignored = ?1
            ORDER BY name ASC",
        )?;

        let result: Vec<model::Person> = stmt
            .query_map([is_ignored], |row| self.to_person(row))?
            .flatten()
            .collect();

        Ok(result)
    }

    /// Hide or restore a whole person. Reversible and data-preserving: their face
    /// assignments stay; only their visibility (overview + auto-recognition seed)
    /// changes.
    pub fn set_person_ignored(&mut self, person_id: PersonId, ignored: bool) -> Result<()> {
        let con = self.con.lock().unwrap();

        // An ignored person must not appear in a sidecar, so write them again.
        con.execute(CLEAR_EXPORT_BY_PERSON, params![person_id.id()])?;

        con.execute(
            "UPDATE people SET is_ignored = ?2 WHERE person_id = ?1",
            params![person_id.id(), ignored],
        )?;
        Ok(())
    }

    /// Find an existing person by exact name (case-insensitive). Used when
    /// importing names from photo metadata so duplicates aren't created.
    pub fn find_person_id_by_name(&self, name: &str) -> Result<Option<PersonId>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare_cached(
            "SELECT person_id FROM people WHERE name = ?1 COLLATE NOCASE LIMIT 1",
        )?;
        let mut rows = stmt.query_map(params![name], |row| row.get::<_, i64>(0))?;
        let id = rows.next().transpose()?;
        Ok(id.map(PersonId::new))
    }

    /// Find people whose name contains `query` (case-insensitive substring). Used
    /// by smart search to fold a person's photos into the results when the query
    /// looks like a name. Empty/whitespace queries match nothing.
    pub fn find_people_by_name_like(&self, query: &str) -> Result<Vec<PersonId>> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }
        // Escape LIKE wildcards in the user's text so they are matched literally.
        let escaped = query
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let pattern = format!("%{escaped}%");
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare_cached(
            "SELECT person_id FROM people
             WHERE name LIKE ?1 ESCAPE '\\' COLLATE NOCASE",
        )?;
        let ids = stmt
            .query_map(params![pattern], |row| row.get::<_, i64>(0))?
            .flatten()
            .map(PersonId::new)
            .collect();
        Ok(ids)
    }

    /// Order known people by how similar their faces' SFace embeddings are to
    /// the given face's embedding (closest first), so the most likely match is
    /// suggested at the top of the naming list. People with no comparable
    /// embedding keep their alphabetical position at the end. Falls back to
    /// plain alphabetical order when the face itself has no embedding.
    pub fn people_by_similarity(&self, face_id: FaceId) -> Result<Vec<model::Person>> {
        use crate::machine_learning::face_recognizer::EMBEDDING_DIM;
        let people = self.all_people()?; // alphabetical
        if people.is_empty() {
            return Ok(people);
        }

        // Embedding of the face we want suggestions for.
        let face_embedding: Option<Vec<f32>> = {
            let con = self.con.lock().unwrap();
            let mut stmt =
                con.prepare("SELECT embedding FROM pictures_faces WHERE face_id = ?1")?;
            let mut rows = stmt.query_map(params![face_id.id()], |row| {
                row.get::<_, Option<Vec<u8>>>(0)
            })?;
            rows.next()
                .transpose()?
                .flatten()
                .map(|b| Self::bytes_to_normalized(&b))
        };
        let Some(face_embedding) = face_embedding else {
            return Ok(people);
        };
        // Stale-dimension embedding (e.g. old SFace): can't compare meaningfully.
        if face_embedding.len() != EMBEDDING_DIM {
            return Ok(people);
        }

        // Minimum distance from this face to each named person's faces.
        let mut best: std::collections::HashMap<i64, f32> = std::collections::HashMap::new();
        {
            let con = self.con.lock().unwrap();
            let mut stmt = con.prepare(
                "SELECT person_id, embedding FROM pictures_faces
                 WHERE person_id IS NOT NULL AND embedding IS NOT NULL",
            )?;
            let rows = stmt.query_map([], |row| {
                let pid: i64 = row.get("person_id")?;
                let bytes: Vec<u8> = row.get("embedding")?;
                Ok((pid, bytes))
            })?;
            for (pid, bytes) in rows.flatten() {
                let other = Self::bytes_to_normalized(&bytes);
                if other.len() != EMBEDDING_DIM {
                    continue;
                }
                let dist = Self::l2(&face_embedding, &other);
                best.entry(pid)
                    .and_modify(|m| {
                        if dist < *m {
                            *m = dist;
                        }
                    })
                    .or_insert(dist);
            }
        }

        let mut with_dist: Vec<(model::Person, f32)> = Vec::new();
        let mut without: Vec<model::Person> = Vec::new();
        for p in people {
            match best.get(&p.person_id.id()) {
                Some(d) => with_dist.push((p, *d)),
                None => without.push(p),
            }
        }
        with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut result: Vec<model::Person> = with_dist.into_iter().map(|(p, _)| p).collect();
        result.extend(without);
        Ok(result)
    }

    /// Confirmed, named faces that carry a current (512-dim ArcFace) embedding,
    /// each paired with its person's last-recognition time. These are the
    /// references the lightweight recognition pass matches unnamed faces
    /// against — using the embeddings already in the DB, so no model load, image
    /// read or inference is needed. A person can contribute several reference
    /// faces (better recall); the match takes the closest.
    pub fn find_recognition_references(&self) -> Result<Vec<RecognitionReference>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT faces.person_id   AS person_id,
                    people.recognized_at AS recognized_at,
                    faces.embedding   AS embedding
             FROM pictures_faces AS faces
             INNER JOIN people USING (person_id)
             WHERE faces.is_confirmed = TRUE
               AND people.is_ignored = FALSE
               AND length(faces.embedding) = 2048",
        )?;
        let rows = stmt.query_map([], |row| {
            let person_id = row.get("person_id").map(PersonId::new)?;
            let recognized_at: DateTime<Utc> = row.get("recognized_at")?;
            let bytes: Vec<u8> = row.get("embedding")?;
            Ok((person_id, recognized_at, bytes))
        })?;

        let mut out = Vec::new();
        for (person_id, recognized_at, bytes) in rows.flatten() {
            out.push(RecognitionReference {
                person_id,
                recognized_at,
                embedding: Self::bytes_to_normalized(&bytes),
            });
        }
        Ok(out)
    }

    /// Unnamed, non-ignored faces that carry a current (512-dim ArcFace)
    /// embedding, with their detection time. The recognition candidates.
    pub fn find_unnamed_with_embeddings(&self) -> Result<Vec<UnnamedEmbedding>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT face_id, detected_at, embedding
             FROM pictures_faces
             WHERE person_id IS NULL
               AND is_ignored = FALSE
               AND bounds_width >= ?1
               AND bounds_height >= ?2
               AND length(embedding) = 2048",
        )?;
        let rows = stmt.query_map([MIN_FACE_PX, MIN_FACE_PX], |row| {
            let face_id = row.get("face_id").map(FaceId::new)?;
            let detected_at: DateTime<Utc> = row.get("detected_at")?;
            let bytes: Vec<u8> = row.get("embedding")?;
            Ok((face_id, detected_at, bytes))
        })?;

        let mut out = Vec::new();
        for (face_id, detected_at, bytes) in rows.flatten() {
            out.push(UnnamedEmbedding {
                face_id,
                detected_at,
                embedding: Self::bytes_to_normalized(&bytes),
            });
        }
        Ok(out)
    }

    /// Cosine similarity of two already-L2-normalised embeddings (a plain dot
    /// product). Higher = more similar.
    pub fn cosine_normalized(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Decode a little-endian f32 embedding BLOB and L2-normalise it.
    fn bytes_to_normalized(bytes: &[u8]) -> Vec<f32> {
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v
        }
    }

    /// Euclidean (L2) distance between two equal-length vectors.
    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// All known people that must have a face recognition performed.
    /// Select the best face for recognition, where "best" is the face with
    /// the highest confidence for a face that the user has confirmed is a particular person.
    pub fn find_people_for_recognition(&self) -> Result<Vec<model::PersonForRecognition>> {
        let con = self.con.lock().unwrap();

        // NOTE: this is non-standard SQL that might not work in DBs that aren't SQLite.
        let mut stmt = con.prepare(
            "SELECT
                person_id,
                recognized_at,

                face_id,
                detected_at,

                is_source_original,

                bounds_path,
                thumbnail_path,

                bounds_x,
                bounds_y,
                bounds_width,
                bounds_height,

                right_eye_x,
                right_eye_y,

                left_eye_x,
                left_eye_y,

                nose_x,
                nose_y,

                right_mouth_corner_x,
                right_mouth_corner_y,

                left_mouth_corner_x,
                left_mouth_corner_y,

                max(confidence) AS confidence
            FROM  pictures_faces AS faces
            INNER JOIN people USING (person_id)
            WHERE faces.is_confirmed = TRUE
            GROUP BY faces.person_id",
        )?;

        let result: Vec<model::PersonForRecognition> = stmt
            .query_map([], |row| self.to_person_for_recognition(row))?
            .flatten()
            .collect();

        Ok(result)
    }

    /// Find new faces as candidates for face recognition for a given person.
    /// Only returns faces that haven't been recognized before for the person.
    pub fn find_unknown_faces(&self) -> Result<Vec<model::DetectedFace>> {
        let con = self.con.lock().unwrap();

        // NOTE: this is non-standard SQL that might not work in DBs that aren't SQLite.
        let mut stmt = con.prepare(
            "SELECT
                face_id,
                detected_at,

                is_source_original,

                bounds_path,
                thumbnail_path,

                bounds_x,
                bounds_y,
                bounds_width,
                bounds_height,

                right_eye_x,
                right_eye_y,

                left_eye_x,
                left_eye_y,

                nose_x,
                nose_y,

                right_mouth_corner_x,
                right_mouth_corner_y,

                left_mouth_corner_x,
                left_mouth_corner_y,

                confidence
            FROM  pictures_faces AS faces
            WHERE faces.person_id IS NULL
            AND faces.is_ignored = FALSE",
        )?;

        let result: Vec<model::DetectedFace> = stmt
            .query_map([], |row| self.to_detected_face(row))?
            .flatten()
            .collect();

        Ok(result)
    }

    /// All non-ignored faces (named or not) that lack a current-dimension
    /// embedding, so embeddings are (re)computed for everything — needed after
    /// switching recognition models. The 2048-byte test = 512 floats (ArcFace).
    pub fn find_faces_needing_embedding(&self) -> Result<Vec<model::DetectedFace>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT
                face_id,
                detected_at,
                is_source_original,
                bounds_path,
                thumbnail_path,
                bounds_x,
                bounds_y,
                bounds_width,
                bounds_height,
                right_eye_x,
                right_eye_y,
                left_eye_x,
                left_eye_y,
                nose_x,
                nose_y,
                right_mouth_corner_x,
                right_mouth_corner_y,
                left_mouth_corner_x,
                left_mouth_corner_y,
                confidence
            FROM pictures_faces AS faces
            WHERE faces.is_ignored = FALSE
            AND (faces.embedding IS NULL OR length(faces.embedding) != 2048)",
        )?;

        let result: Vec<model::DetectedFace> = stmt
            .query_map([], |row| self.to_detected_face(row))?
            .flatten()
            .collect();

        Ok(result)
    }

    /// Finds all pictures that feature a known person.
    pub fn find_pictures_for_person(&self, person_id: PersonId) -> Result<Vec<PictureId>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT DISTINCT
                picture_id
            FROM  pictures_faces
            WHERE person_id == ?1",
        )?;

        let result: Vec<PictureId> = stmt
            .query_map([person_id.id()], |row| {
                row.get("picture_id").map(PictureId::new)
            })?
            .flatten()
            .collect();

        Ok(result)
    }

    /// Pictures where this person appears via an UNCONFIRMED (auto-suggested)
    /// face — the review set for "show only suggestions".
    pub fn find_unconfirmed_pictures_for_person(
        &self,
        person_id: PersonId,
    ) -> Result<Vec<PictureId>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT DISTINCT
                picture_id
            FROM  pictures_faces
            WHERE person_id == ?1 AND is_confirmed = FALSE",
        )?;

        let result: Vec<PictureId> = stmt
            .query_map([person_id.id()], |row| {
                row.get("picture_id").map(PictureId::new)
            })?
            .flatten()
            .collect();

        Ok(result)
    }

    /// Like [`Self::find_unconfirmed_pictures_for_person`], but also returns the
    /// recognition score per picture so callers can bucket suggestions into
    /// confidence tiers. A picture's score is the strongest unconfirmed face of
    /// this person in it. Faces assigned before scores were stored (NULL) fall
    /// back to 0.42 (the old precision-leaning threshold → "Possible" tier).
    pub fn find_unconfirmed_scored_for_person(
        &self,
        person_id: PersonId,
    ) -> Result<Vec<(PictureId, f32)>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT
                picture_id,
                MAX(COALESCE(recognition_score, 0.42)) AS score
            FROM  pictures_faces
            WHERE person_id == ?1 AND is_confirmed = FALSE
            GROUP BY picture_id",
        )?;

        let result: Vec<(PictureId, f32)> = stmt
            .query_map([person_id.id()], |row| {
                Ok((
                    PictureId::new(row.get("picture_id")?),
                    row.get::<_, f64>("score")? as f32,
                ))
            })?
            .flatten()
            .collect();

        Ok(result)
    }

    // FIXME probably need a mechanism to undo this in the likely event of user error.
    pub fn mark_ignore(&mut self, face_id: FaceId) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        // The confirmed person link of this face is removed, so the sidecar
        // is out of date.
        tx.execute(CLEAR_EXPORT_BY_FACE, params![face_id.id()])?;

        {
            let mut stmt = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    is_ignored = TRUE,
                    is_confirmed = FALSE,
                    is_thumbnail = FALSE,
                    person_id = NULL
                WHERE face_id = ?1",
            )?;

            stmt.execute(params![face_id.id(),])?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Restore a previously ignored face: clear `is_ignored` so it returns to the
    /// unknown-faces pool. Reverses `mark_ignore` (the prior person link, if any,
    /// is not restored — the face can simply be named again).
    pub fn restore_face(&mut self, face_id: FaceId) -> Result<()> {
        let con = self.con.lock().unwrap();
        con.execute(
            "UPDATE pictures_faces SET is_ignored = FALSE WHERE face_id = ?1",
            params![face_id.id()],
        )?;
        Ok(())
    }

    pub fn mark_face_scan_broken(&mut self, picture_id: &PictureId) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO pictures_face_scans (
                    picture_id,
                    is_broken,
                    face_count,
                    scan_ts
                ) VALUES (
                    ?1, TRUE, 0, CURRENT_TIMESTAMP
                ) ON CONFLICT (picture_id) DO UPDATE SET
                    is_broken = true,
                    face_count = 0,
                    scan_ts = CURRENT_TIMESTAMP
                ",
            )?;

            stmt.execute(params![picture_id.id(),])?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Faces of a picture with the fields that must survive a detector change.
    pub fn find_faces_for_picture(
        &self,
        picture_id: &PictureId,
    ) -> Result<Vec<model::ExistingFace>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT
                face_id,
                bounds_x,
                bounds_y,
                bounds_width,
                bounds_height,
                is_source_original,
                person_id,
                is_confirmed,
                is_thumbnail,
                is_ignored
             FROM pictures_faces
             WHERE picture_id = ?1",
        )?;

        let rows = stmt.query_map([picture_id.id()], |row| {
            Ok(model::ExistingFace {
                face_id: FaceId::new(row.get("face_id")?),
                bounds: Rect {
                    x: row.get("bounds_x")?,
                    y: row.get("bounds_y")?,
                    width: row.get("bounds_width")?,
                    height: row.get("bounds_height")?,
                },
                is_source_original: row.get("is_source_original")?,
                person_id: row.get::<_, Option<i64>>("person_id")?.map(PersonId::new),
                is_confirmed: row.get("is_confirmed")?,
                is_thumbnail: row.get("is_thumbnail")?,
                is_ignored: row.get("is_ignored")?,
            })
        })?;

        Ok(rows.flatten().collect())
    }

    /// Insert a picture's face scan and its faces.
    ///
    /// Existing faces are matched to the new detections. A matched old face
    /// transfers its person assignment, confirmation, thumbnail flag, and
    /// ignored flag to the new face, then the old row is removed. Unmatched
    /// old faces are kept when they carry a person assignment, a person
    /// thumbnail, or an ignore decision. This is what keeps names intact when
    /// the user switches face detectors.
    pub fn replace_face_scans(
        &mut self,
        picture_id: &PictureId,
        faces: &[face_extractor::Face],
        detector: &str,
        old_faces: &[model::ExistingFace],
    ) -> Result<()> {
        let matches = Self::match_existing_faces(faces, old_faces);

        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        let mut scan_insert_stmt = tx.prepare_cached(
            "INSERT INTO pictures_face_scans (
                picture_id,
                is_broken,
                face_count,
                scan_ts,
                detector
            ) VALUES (
                ?1, ?2, ?3, CURRENT_TIMESTAMP, ?4
            ) ON CONFLICT (picture_id) DO UPDATE SET
                is_broken = ?2,
                face_count = ?3,
                scan_ts = CURRENT_TIMESTAMP,
                detector = ?4
            ",
        )?;

        let mut face_insert_stmt = tx.prepare_cached(
            "INSERT INTO pictures_faces (
                picture_id,
                thumbnail_path,
                bounds_path,
                model_name,

                person_id,
                is_thumbnail,
                is_confirmed,
                is_source_original,

                bounds_x,
                bounds_y,
                bounds_width,
                bounds_height,

                right_eye_x,
                right_eye_y,

                left_eye_x,
                left_eye_y,

                nose_x,
                nose_y,

                right_mouth_corner_x,
                right_mouth_corner_y,

                left_mouth_corner_x,
                left_mouth_corner_y,

                confidence,
                is_ignored
            ) VALUES (
                ?1, ?2, ?3, ?4,
                ?5, ?6, ?7, false,
                ?8, ?9, ?10, ?11,
                ?12, ?13,
                ?14, ?15,
                ?16, ?17,
                ?18, ?19,
                ?20, ?21,
                ?22, ?23
            )
            ",
        )?;

        for (index, face) in faces.iter().enumerate() {
            let old = matches[index].map(|old_index| &old_faces[old_index]);
            let thumbnail_path = face.thumbnail_path.strip_prefix(&self.data_dir_base_path)?;
            let bounds_path = face.bounds_path.strip_prefix(&self.data_dir_base_path)?;

            let right_eye = face.right_eye();
            let left_eye = face.left_eye();
            let nose = face.nose();
            let right_mouth_corner = face.right_mouth_corner();
            let left_mouth_corner = face.left_mouth_corner();

            face_insert_stmt.execute(params![
                picture_id.id(),
                thumbnail_path.to_string_lossy(),
                bounds_path.to_string_lossy(),
                face.model_name,
                old.and_then(|o| o.person_id).map(|p| p.id()),
                old.is_some_and(|o| o.is_thumbnail),
                old.is_some_and(|o| o.is_confirmed),
                face.bounds.x,
                face.bounds.y,
                face.bounds.width,
                face.bounds.height,
                right_eye.map(|x| x.0),
                right_eye.map(|x| x.1),
                left_eye.map(|x| x.0),
                left_eye.map(|x| x.1),
                nose.map(|x| x.0),
                nose.map(|x| x.1),
                right_mouth_corner.map(|x| x.0),
                right_mouth_corner.map(|x| x.1),
                left_mouth_corner.map(|x| x.0),
                left_mouth_corner.map(|x| x.1),
                face.confidence,
                old.is_some_and(|o| o.is_ignored),
            ])?;
        }

        // Delete matched old faces: their new counterparts now carry the
        // assignment. Unmatched old faces are only deleted when they contain
        // no user decision; named, ignored, and person-thumbnail faces stay so
        // the new detector cannot lose them.
        let matched_old: std::collections::HashSet<usize> =
            matches.iter().flatten().copied().collect();
        let mut kept_old = 0usize;
        {
            let mut delete_stmt =
                tx.prepare_cached("DELETE FROM pictures_faces WHERE face_id = ?1")?;
            for (index, old) in old_faces.iter().enumerate() {
                let keep = !matched_old.contains(&index)
                    && (old.person_id.is_some() || old.is_thumbnail || old.is_ignored);
                if keep {
                    kept_old += 1;
                } else {
                    delete_stmt.execute([old.face_id.id()])?;
                }
            }
        }

        let face_count = faces.len() + kept_old;
        scan_insert_stmt.execute(params![picture_id.id(), false, face_count as u32, detector])?;

        drop(face_insert_stmt);
        drop(scan_insert_stmt);
        tx.commit()?;
        Ok(())
    }

    /// Insert a face scan for a picture that has no previous faces.
    pub fn add_face_scans(
        &mut self,
        picture_id: &PictureId,
        faces: &[face_extractor::Face],
        detector: &str,
    ) -> Result<()> {
        self.replace_face_scans(picture_id, faces, detector, &[])
    }

    /// Greedy one-to-one matching from new detections to old faces. First by
    /// bounding-box overlap, then by centre containment for boxes that moved.
    fn match_existing_faces(
        faces: &[face_extractor::Face],
        old_faces: &[model::ExistingFace],
    ) -> Vec<Option<usize>> {
        let mut used = vec![false; old_faces.len()];
        let mut matches = vec![None; faces.len()];

        for (new_index, new_face) in faces.iter().enumerate() {
            let mut best_iou = 0.3f32;
            let mut best_old = None;
            for (old_index, old) in old_faces.iter().enumerate() {
                if used[old_index] || old.is_source_original {
                    continue;
                }
                let iou = Self::rect_iou(&old.bounds, &new_face.bounds);
                if iou > best_iou {
                    best_iou = iou;
                    best_old = Some(old_index);
                }
            }
            if let Some(old_index) = best_old {
                matches[new_index] = Some(old_index);
                used[old_index] = true;
            }
        }

        for (new_index, new_face) in faces.iter().enumerate() {
            if matches[new_index].is_some() {
                continue;
            }
            let new_centre_x = new_face.bounds.x + new_face.bounds.width / 2.0;
            let new_centre_y = new_face.bounds.y + new_face.bounds.height / 2.0;
            let mut best_distance = f32::MAX;
            let mut best_old = None;
            for (old_index, old) in old_faces.iter().enumerate() {
                if used[old_index] || old.is_source_original {
                    continue;
                }
                let old_centre_x = old.bounds.x + old.bounds.width / 2.0;
                let old_centre_y = old.bounds.y + old.bounds.height / 2.0;
                let inside = old_centre_x >= new_face.bounds.x
                    && old_centre_x <= new_face.bounds.x + new_face.bounds.width
                    && old_centre_y >= new_face.bounds.y
                    && old_centre_y <= new_face.bounds.y + new_face.bounds.height;
                let old_size = old.bounds.width.max(old.bounds.height);
                let new_size = new_face.bounds.width.max(new_face.bounds.height).max(1.0);
                let ratio = old_size / new_size;
                if inside && (0.5..=2.0).contains(&ratio) {
                    let distance = (old_centre_x - new_centre_x).powi(2)
                        + (old_centre_y - new_centre_y).powi(2);
                    if distance < best_distance {
                        best_distance = distance;
                        best_old = Some(old_index);
                    }
                }
            }
            if let Some(old_index) = best_old {
                matches[new_index] = Some(old_index);
                used[old_index] = true;
            }
        }

        matches
    }

    fn rect_iou(a: &Rect, b: &face_extractor::Rect) -> f32 {
        let x1 = a.x.max(b.x);
        let y1 = a.y.max(b.y);
        let x2 = (a.x + a.width).min(b.x + b.width);
        let y2 = (a.y + a.height).min(b.y + b.height);
        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let union = a.width * a.height + b.width * b.height - intersection;
        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    /// Add a new named person derived from a face.
    pub fn add_person(&mut self, face_id: FaceId, name: &str) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        // The names of this picture changed, so the sidecar is out of date.
        tx.execute(CLEAR_EXPORT_BY_FACE, params![face_id.id()])?;

        {
            // GTK allows the text gtk::Entry input box to be activated multiple times
            // which results in duplicate people being created :-(
            // FIXME can we configure gtk::Entry to only be activatible once?
            let mut insert_person = tx.prepare_cached(
                "
                INSERT INTO people (name)
                SELECT ?1 AS name
                FROM pictures_faces
                WHERE face_id = ?2 AND person_id IS NULL
                ",
            )?;

            insert_person.execute(params![name, face_id.id(),])?;

            // Zero if no rows inserted.
            // See https://www.sqlite.org/c3ref/last_insert_rowid.html
            let person_id = tx.last_insert_rowid();
            if person_id == 0 {
                warn!("Detected double insert of person. Skipping.");
                return Ok(());
            }

            let mut update_face = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    person_id = ?2,
                    is_confirmed = TRUE,
                    is_thumbnail = TRUE
                WHERE face_id = ?1",
            )?;

            update_face.execute(params![face_id.id(), person_id,])?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Like [`add_person`], but the assignment is left unconfirmed: the name is
    /// a recommendation (e.g. imported from photo XMP) that the user can
    /// override. The first face still becomes the person's thumbnail so the
    /// person has an avatar.
    pub fn add_person_unconfirmed(&mut self, face_id: FaceId, name: &str) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        {
            let mut insert_person = tx.prepare_cached(
                "
                INSERT INTO people (name)
                SELECT ?1 AS name
                FROM pictures_faces
                WHERE face_id = ?2 AND person_id IS NULL
                ",
            )?;

            insert_person.execute(params![name, face_id.id(),])?;

            let person_id = tx.last_insert_rowid();
            if person_id == 0 {
                warn!("Detected double insert of person. Skipping.");
                return Ok(());
            }

            let mut update_face = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    person_id = ?2,
                    is_confirmed = FALSE,
                    is_thumbnail = TRUE
                WHERE face_id = ?1",
            )?;

            update_face.execute(params![face_id.id(), person_id,])?;
        }

        tx.commit()?;
        Ok(())
    }

    /// User is manually marking a face as a person
    pub fn mark_as_person(&mut self, face_id: FaceId, person_id: PersonId) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        // The names of this picture changed, so the sidecar is out of date.
        tx.execute(CLEAR_EXPORT_BY_FACE, params![face_id.id()])?;

        {
            let mut stmt = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    person_id = ?2,
                    is_confirmed = TRUE
                WHERE face_id = ?1",
            )?;

            stmt.execute(params![face_id.id(), person_id.id(),])?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Face recognition is automatically marking a face as a person. `score` is
    /// the cosine similarity to the best matching reference (None for sources
    /// without one, e.g. imported XMP names) and feeds the confidence tiers.
    pub fn mark_as_person_unconfirmed(
        &mut self,
        face_id: FaceId,
        person_id: PersonId,
        score: Option<f32>,
    ) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        {
            let mut stmt = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    person_id = ?2,
                    is_confirmed = FALSE,
                    is_thumbnail = FALSE,
                    recognition_score = ?3
                WHERE face_id = ?1",
            )?;

            stmt.execute(params![face_id.id(), person_id.id(), score])?;
        }

        tx.commit()?;
        Ok(())
    }

    pub fn mark_face_recognition_complete(&mut self, person_id: PersonId) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        {
            let mut stmt = tx.prepare_cached(
                "UPDATE people
                SET
                    recognized_at = CURRENT_TIMESTAMP
                WHERE person_id = ?1",
            )?;

            stmt.execute(params![person_id.id(),])?;
        }

        tx.commit()?;
        Ok(())
    }

    /// All "this face is not this person" rejections as `face_id -> {person_id}`.
    /// Recognition uses it to never re-assign a face to a rejected person.
    pub fn find_negative_associations(
        &self,
    ) -> Result<std::collections::HashMap<i64, std::collections::HashSet<i64>>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare("SELECT face_id, person_id FROM face_not_person")?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;
        let mut map: std::collections::HashMap<i64, std::collections::HashSet<i64>> =
            std::collections::HashMap::new();
        for (face_id, person_id) in rows.flatten() {
            map.entry(face_id).or_default().insert(person_id);
        }
        Ok(map)
    }

    pub fn mark_not_person(&mut self, face_id: FaceId) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        // The names of this picture changed, so the sidecar is out of date.
        tx.execute(CLEAR_EXPORT_BY_FACE, params![face_id.id()])?;

        {
            // Negative learning: remember that this face is NOT its current
            // person, so recognition never re-assigns it to them.
            let mut remember = tx.prepare_cached(
                "INSERT OR IGNORE INTO face_not_person (face_id, person_id)
                 SELECT face_id, person_id FROM pictures_faces
                 WHERE face_id = ?1 AND person_id IS NOT NULL",
            )?;
            remember.execute(params![face_id.id(),])?;

            let mut stmt = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    person_id = NULL,
                    is_confirmed = FALSE,
                    is_thumbnail = FALSE
                WHERE face_id = ?1",
            )?;

            stmt.execute(params![face_id.id(),])?;
        }

        tx.commit()?;
        Ok(())
    }

    pub fn set_person_thumbnail(&mut self, person_id: PersonId, face_id: FaceId) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        // This face becomes confirmed, so the sidecar is out of date.
        tx.execute(CLEAR_EXPORT_BY_FACE, params![face_id.id()])?;

        {
            let mut stmt = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    is_thumbnail = FALSE
                WHERE
                    person_id = ?1
                    AND face_id != ?2",
            )?;

            stmt.execute(params![person_id.id(), face_id.id(),])?;

            let mut stmt = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    is_confirmed = TRUE,
                    is_thumbnail = TRUE
                WHERE face_id = ?1",
            )?;

            stmt.execute(params![face_id.id(),])?;
        }

        tx.commit()?;
        Ok(())
    }

    fn to_face_and_person(
        &self,
        row: &Row<'_>,
    ) -> rusqlite::Result<(model::Face, Option<model::Person>)> {
        let face_id = row.get("face_id").map(FaceId::new)?;

        let face_thumbnail_path = row
            .get("face_thumbnail_path")
            .map(|p: String| self.data_dir_base_path.join(p))?;

        let face = model::Face {
            face_id,
            thumbnail_path: face_thumbnail_path,
        };

        let person_id = row.get("person_id").map(PersonId::new).ok();

        let person_name = row.get("person_name").ok();

        let person = if let (Some(person_id), Some(name)) = (person_id, person_name) {
            let person_thumbnail_path = row
                .get("person_thumbnail_path")
                .map(|p: String| self.data_dir_base_path.join(p))
                .ok();

            let large_thumbnail_path = if let Some(ref small_thumbnail_path) = person_thumbnail_path
            {
                Some(
                    self.cache_dir_base_path
                        .join("face_thumbnails")
                        .join("large")
                        .join(
                            small_thumbnail_path
                                .file_name()
                                .expect("Must have file name"),
                        ),
                )
            } else {
                None
            };

            Some(model::Person {
                person_id,
                name,
                small_thumbnail_path: person_thumbnail_path,
                large_thumbnail_path: large_thumbnail_path,
                // Not selected on this path; not used for ignore decisions here.
                is_ignored: false,
            })
        } else {
            None
        };

        std::result::Result::Ok((face, person))
    }

    fn to_person(&self, row: &Row<'_>) -> rusqlite::Result<model::Person> {
        let person_id = row.get("person_id").map(PersonId::new)?;

        let name = row.get("person_name")?;

        let small_thumbnail_path = row
            .get("person_thumbnail_path")
            .map(|p: String| self.data_dir_base_path.join(p))
            .ok();

        // FIXME should this path be in database?
        let large_thumbnail_path = if let Some(ref small_thumbnail_path) = small_thumbnail_path {
            Some(
                self.cache_dir_base_path
                    .join("face_thumbnails")
                    .join("large")
                    .join(
                        small_thumbnail_path
                            .file_name()
                            .expect("Must have file name"),
                    ),
            )
        } else {
            None
        };

        let is_ignored = row.get("is_ignored").unwrap_or(false);

        std::result::Result::Ok(model::Person {
            person_id,
            name,
            small_thumbnail_path,
            large_thumbnail_path,
            is_ignored,
        })
    }

    fn to_detected_face(&self, row: &Row<'_>) -> rusqlite::Result<model::DetectedFace> {
        let face_id = row.get("face_id").map(FaceId::new)?;

        let face_path = row
            .get("bounds_path")
            .map(|p: String| self.data_dir_base_path.join(p))?;

        let thumbnail_path = row
            .get("thumbnail_path")
            .map(|p: String| self.data_dir_base_path.join(p))?;

        let bounds = Rect {
            x: row.get("bounds_x")?,
            y: row.get("bounds_y")?,
            width: row.get("bounds_width")?,
            height: row.get("bounds_height")?,
        };

        let right_eye_x = row.get("right_eye_x")?;
        let right_eye_y = row.get("right_eye_y")?;

        let left_eye_x = row.get("left_eye_x")?;
        let left_eye_y = row.get("left_eye_y")?;

        let nose_x = row.get("nose_x")?;
        let nose_y = row.get("nose_y")?;

        let right_mouth_corner_x = row.get("right_mouth_corner_x")?;
        let right_mouth_corner_y = row.get("right_mouth_corner_y")?;

        let left_mouth_corner_x = row.get("left_mouth_corner_x")?;
        let left_mouth_corner_y = row.get("left_mouth_corner_y")?;

        let confidence = row.get("confidence")?;

        let detected_at = row.get("detected_at")?;

        let is_source_original: bool = row.get("is_source_original")?;

        let face = model::DetectedFace {
            face_id,
            face_path,
            small_thumbnail_path: thumbnail_path,
            is_source_original,
            bounds,
            right_eye: (right_eye_x, right_eye_y),
            left_eye: (left_eye_x, left_eye_y),
            nose: (nose_x, nose_y),
            right_mouth_corner: (right_mouth_corner_x, right_mouth_corner_y),
            left_mouth_corner: (left_mouth_corner_x, left_mouth_corner_y),
            confidence,
            detected_at,
        };

        std::result::Result::Ok(face)
    }

    fn to_person_for_recognition(
        &self,
        row: &Row<'_>,
    ) -> rusqlite::Result<model::PersonForRecognition> {
        let person_id = row.get("person_id").map(PersonId::new)?;
        let recognized_at = row.get("recognized_at")?;
        let face = self.to_detected_face(row)?;

        let person = PersonForRecognition {
            person_id,
            recognized_at,
            face,
        };

        std::result::Result::Ok(person)
    }

    pub fn migrate_get_all(&self) -> Result<Vec<FaceToMigrate>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT
                migrate_faces.face_id AS face_id,
                migrate_faces.face_index AS face_index,
                pictures.picture_path_b64 AS picture_path_b64,
                pictures_faces.bounds_path AS bounds_path,
                pictures_faces.thumbnail_path AS thumbnail_path
            FROM migrate_faces
            INNER JOIN pictures_faces USING (face_id)
            INNER JOIN pictures USING (picture_id)",
        )?;

        let result: Vec<model::FaceToMigrate> = stmt
            .query_map([], |row| self.to_face_to_migrate(row))?
            .flatten()
            .collect();

        Ok(result)
    }

    pub fn migrate_update_face_paths(&mut self, mf: MigratedFace) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;

        {
            let mut stmt = tx.prepare_cached(
                "UPDATE pictures_faces
                SET
                    bounds_path = ?1,
                    thumbnail_path = ?2
                WHERE face_id = ?3",
            )?;

            stmt.execute(params![
                mf.bounds_path.to_string_lossy(),
                mf.thumbnail_path.to_string_lossy(),
                mf.face_id.id(),
            ])?;
        }

        tx.commit()?;
        Ok(())
    }

    pub fn migrate_truncate(&mut self) -> Result<()> {
        let mut con = self.con.lock().unwrap();
        let tx = con.transaction()?;
        {
            tx.execute("DELETE FROM migrate_faces", [])?;
        }
        tx.commit()?;
        Ok(())
    }

    fn to_face_to_migrate(&self, row: &Row<'_>) -> rusqlite::Result<FaceToMigrate> {
        let face_id = row.get("face_id").map(FaceId::new)?;
        let face_index: u32 = row.get("face_index")?;

        let picture_relative_path = row
            .get("picture_path_b64")
            .ok()
            .and_then(|x: String| path_encoding::from_base64(&x).ok())
            .expect("Must have picture path");

        let bounds_path = row
            .get("bounds_path")
            .map(|p: String| self.data_dir_base_path.join(p))?;

        let thumbnail_path = row
            .get("thumbnail_path")
            .map(|p: String| self.data_dir_base_path.join(p))?;

        let face = model::FaceToMigrate {
            face_id,
            face_index,
            picture_relative_path,
            bounds_path,
            thumbnail_path,
        };

        std::result::Result::Ok(face)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database;
    use crate::machine_learning::face_extractor::{Face, Rect as FaceRect};
    use crate::photo::model::PictureId;
    use std::sync::{Arc, Mutex};

    fn test_repo() -> Repository {
        let con = database::setup_in_memory().unwrap();
        let con = Arc::new(Mutex::new(con));
        {
            let con = con.lock().unwrap();
            con.execute(
                "INSERT INTO pictures (
                    picture_id, picture_path_b64, picture_path_lossy,
                    fs_created_ts, link_path_b64, link_path_lossy
                ) VALUES (1, 'cGljLmpwZw==', 'pic.jpg', CURRENT_TIMESTAMP, 'bGluaw==', 'link')",
                [],
            )
            .unwrap();
            con.execute(
                "INSERT INTO people (person_id, name) VALUES (10, 'Alice')",
                [],
            )
            .unwrap();
            con.execute(
                "INSERT INTO pictures_faces (
                    face_id, picture_id, model_name, person_id,
                    is_thumbnail, is_confirmed, is_source_original,
                    thumbnail_path, bounds_path,
                    bounds_x, bounds_y, bounds_width, bounds_height,
                    right_eye_x, right_eye_y, left_eye_x, left_eye_y,
                    nose_x, nose_y, right_mouth_corner_x, right_mouth_corner_y,
                    left_mouth_corner_x, left_mouth_corner_y,
                    confidence, is_ignored
                ) VALUES (
                    100, 1, 'blaze_face_640_default', 10,
                    TRUE, TRUE, FALSE,
                    'old_thumb.jpg', 'old_bounds.jpg',
                    10, 10, 20, 20,
                    14, 14, 16, 14, 15, 17, 16, 20, 14, 20,
                    0.9, FALSE
                )",
                [],
            )
            .unwrap();
        }
        Repository::open(Path::new("/tmp"), Path::new("/tmp"), con).unwrap()
    }

    fn face(bounds: FaceRect) -> Face {
        Face::new(
            PathBuf::from("/tmp/new_thumb.jpg"),
            PathBuf::from("/tmp/new_bounds.jpg"),
            bounds,
            0.95,
            Some(vec![
                (14.0, 14.0),
                (16.0, 14.0),
                (15.0, 17.0),
                (16.0, 20.0),
                (14.0, 20.0),
            ]),
            "yunet".into(),
        )
    }

    #[test]
    fn replace_face_scans_preserves_person_assignment() {
        let mut repo = test_repo();
        let old_faces = repo.find_faces_for_picture(&PictureId::new(1)).unwrap();
        assert_eq!(old_faces.len(), 1);
        assert_eq!(old_faces[0].person_id, Some(PersonId::new(10)));

        repo.replace_face_scans(
            &PictureId::new(1),
            &[face(FaceRect {
                x: 10.0,
                y: 10.0,
                width: 20.0,
                height: 20.0,
            })],
            "yunet",
            &old_faces,
        )
        .unwrap();

        let con = repo.con.lock().unwrap();
        let mut stmt = con
            .prepare(
                "SELECT person_id, is_confirmed, is_thumbnail, is_ignored, model_name
                 FROM pictures_faces WHERE picture_id = 1",
            )
            .unwrap();
        let rows: Vec<(Option<i64>, bool, bool, bool, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })
            .unwrap()
            .flatten()
            .collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, Some(10));
        assert!(rows[0].1);
        assert!(rows[0].2);
        assert!(!rows[0].3);
        assert_eq!(rows[0].4, "yunet");
    }

    #[test]
    fn replace_face_scans_keeps_unmatched_named_face() {
        let mut repo = test_repo();
        let old_faces = repo.find_faces_for_picture(&PictureId::new(1)).unwrap();

        repo.replace_face_scans(
            &PictureId::new(1),
            &[face(FaceRect {
                x: 200.0,
                y: 200.0,
                width: 40.0,
                height: 40.0,
            })],
            "yunet",
            &old_faces,
        )
        .unwrap();

        let con = repo.con.lock().unwrap();
        let count: i64 = con
            .query_row(
                "SELECT COUNT(*) FROM pictures_faces WHERE picture_id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2);

        let named: i64 = con
            .query_row(
                "SELECT COUNT(*) FROM pictures_faces
                 WHERE picture_id = 1 AND person_id = 10",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(named, 1);
    }
}
