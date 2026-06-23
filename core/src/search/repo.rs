// SPDX-FileCopyrightText: © 2025 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

//! Repository for CLIP image embeddings, backed by the shared SQLite connection.
//! Mirrors the embedding storage in [`crate::people::repo`]: vectors are stored as
//! little-endian f32 BLOBs and L2-normalised, so cosine similarity is a plain dot
//! product at search time.

use std::result::Result::Ok;
use std::sync::{Arc, Mutex};

use anyhow::*;
use chrono::Utc;
use rusqlite::params;

use crate::photo::model::PictureId;

#[derive(Debug, Clone)]
pub struct Repository {
    con: Arc<Mutex<rusqlite::Connection>>,
}

impl Repository {
    pub fn open(con: Arc<Mutex<rusqlite::Connection>>) -> Result<Repository> {
        Ok(Repository { con })
    }

    /// Store (or replace) the CLIP embedding for a picture. The vector should
    /// already be L2-normalised by the embedder.
    pub fn store_clip_embedding(
        &self,
        picture_id: PictureId,
        model_name: &str,
        embedding: &[f32],
    ) -> Result<()> {
        let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let now = Utc::now();
        let con = self.con.lock().unwrap();
        con.execute(
            "INSERT INTO pictures_clip_embeddings (picture_id, model_name, embedding, embedded_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT (picture_id) DO UPDATE SET
                 model_name = excluded.model_name,
                 embedding = excluded.embedding,
                 embedded_at = excluded.embedded_at",
            params![picture_id.id(), model_name, bytes, now],
        )?;
        Ok(())
    }

    /// All stored embeddings for the given model, as (picture_id, vector). Used to
    /// rank a query against the whole library. Embeddings stored under a different
    /// model_name are ignored (they get recomputed by the background task).
    pub fn all_clip_embeddings(&self, model_name: &str) -> Result<Vec<(PictureId, Vec<f32>)>> {
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(
            "SELECT picture_id, embedding
             FROM pictures_clip_embeddings
             WHERE model_name = ?1",
        )?;
        let rows = stmt.query_map(params![model_name], |row| {
            let picture_id = row.get("picture_id").map(PictureId::new)?;
            let bytes: Vec<u8> = row.get("embedding")?;
            Ok((picture_id, bytes))
        })?;

        let mut out = Vec::new();
        for (picture_id, bytes) in rows.flatten() {
            let v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            out.push((picture_id, v));
        }
        Ok(out)
    }

    /// Number of pictures already embedded for the given model.
    pub fn count_clip_embeddings(&self, model_name: &str) -> Result<usize> {
        let con = self.con.lock().unwrap();
        let n: i64 = con.query_row(
            "SELECT COUNT(*) FROM pictures_clip_embeddings WHERE model_name = ?1",
            params![model_name],
            |row| row.get(0),
        )?;
        Ok(n as usize)
    }
}
