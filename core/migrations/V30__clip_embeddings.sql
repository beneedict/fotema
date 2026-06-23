-- SPDX-FileCopyrightText: © 2025 David Bliss
--
-- SPDX-License-Identifier: GPL-3.0-or-later

-- CLIP "smart search" embeddings: one vector per picture (f32, little-endian,
-- L2-normalised) produced by a multilingual CLIP image encoder. A free-text query
-- is embedded into the same space and ranked by cosine similarity. Additive only:
-- existing databases keep working and embeddings are computed lazily in the
-- background for pictures that already have thumbnails.
CREATE TABLE pictures_clip_embeddings (
    picture_id   INTEGER PRIMARY KEY,
    model_name   TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    embedded_at  DATETIME NOT NULL,
    FOREIGN KEY (picture_id) REFERENCES pictures (picture_id) ON DELETE CASCADE
);
