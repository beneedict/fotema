// SPDX-FileCopyrightText: © 2025 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

//! CLIP "smart search": storage and retrieval of per-picture image embeddings.
//! The embeddings themselves are produced by
//! [`crate::machine_learning::clip`]; this module only persists them and serves
//! them for ranking at search time.

pub mod repo;

pub use repo::Repository;
