// SPDX-FileCopyrightText: © 2025 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Computes a CLIP image embedding for every photo so the smart-search tab can
// rank the whole library against a free-text query. Mirrors the structure of
// `photo_recognize_faces_task.rs`: a bounded rayon pool, one model session per
// worker thread, progress reported via ProgressMonitor, and a stop flag.

use anyhow::*;
use relm4::Reducer;
use relm4::Worker;
use relm4::prelude::*;

use std::path::PathBuf;
use std::result::Result::Ok;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::{error, info};

use fotema_core::machine_learning::clip::{self, ClipImageEmbedder};
use fotema_core::photo;
use fotema_core::search;
use fotema_core::thumbnailify::{ThumbnailSize, Thumbnailer};

use crate::app::components::progress_monitor::{ProgressMonitor, ProgressMonitorInput, TaskName};

#[derive(Debug)]
pub enum PhotoClipEmbedTaskInput {
    Start,
}

#[derive(Debug)]
pub enum PhotoClipEmbedTaskOutput {
    Started,
    Completed,
}

#[derive(Clone)]
pub struct PhotoClipEmbedTask {
    stop: Arc<AtomicBool>,
    cache_dir: PathBuf,
    thumbnailer: Thumbnailer,
    photo_repo: photo::Repository,
    search_repo: search::Repository,
    progress_monitor: Arc<Reducer<ProgressMonitor>>,
}

impl PhotoClipEmbedTask {
    fn embed(&self, sender: ComponentSender<Self>) -> Result<()> {
        let start = std::time::Instant::now();

        // Pictures without an embedding for the current model. Only those with an
        // existing thumbnail can be embedded (the image encoder reads it).
        let candidates = self
            .photo_repo
            .find_clip_embedding_candidates(clip::MODEL_NAME)?;

        // Resolve each candidate to its cached thumbnail up front; skip any that
        // haven't been thumbnailed yet (they'll be picked up on a later pass).
        let work: Vec<(fotema_core::PictureId, PathBuf)> = candidates
            .into_iter()
            .filter_map(|c| {
                let hash = c.thumbnail_hash();
                self.thumbnailer
                    .nearest_thumbnail(&hash, ThumbnailSize::XLarge)
                    .map(|path| (c.picture_id, path))
            })
            .collect();

        let count = work.len();
        info!("Found {} photos needing a CLIP embedding", count);

        if count == 0 {
            let _ = sender.output(PhotoClipEmbedTaskOutput::Completed);
            return Ok(());
        }

        // Download the image encoder before spawning workers so they don't race to
        // fetch the same file. Done off the main thread (this runs in rayon::spawn).
        let visual_path = clip::ensure_visual_model(&self.cache_dir)?;

        let _ = sender.output(PhotoClipEmbedTaskOutput::Started);
        self.progress_monitor
            .emit(ProgressMonitorInput::Start(TaskName::ClipEmbed, count));

        // Single session, processed sequentially. A GPU execution provider
        // (WebGPU/Dawn→Vulkan) parallelises internally, and creating multiple
        // concurrent GPU sessions corrupts the heap (Dawn is not safe to spin up
        // per-thread). On CPU a single session already uses all cores via intra-op
        // threading, so throughput is essentially the same as a pool of sessions.
        let mut embedder = match ClipImageEmbedder::new(&visual_path) {
            Ok(e) => e,
            Err(e) => {
                error!("Failed to create CLIP image embedder: {:?}", e);
                self.progress_monitor.emit(ProgressMonitorInput::Complete);
                let _ = sender.output(PhotoClipEmbedTaskOutput::Completed);
                return Ok(());
            }
        };

        for (picture_id, thumbnail_path) in work {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            match embedder.embedding(&thumbnail_path) {
                Ok(embedding) => {
                    if let Err(e) =
                        self.search_repo
                            .store_clip_embedding(picture_id, clip::MODEL_NAME, &embedding)
                    {
                        error!("Failed storing CLIP embedding for {}: {:?}", picture_id, e);
                    }
                }
                Err(e) => error!(
                    "Failed computing CLIP embedding for {}: {:?}",
                    picture_id, e
                ),
            }
            self.progress_monitor.emit(ProgressMonitorInput::Advance);
        }

        self.progress_monitor.emit(ProgressMonitorInput::Complete);

        info!(
            "Computed CLIP embeddings for {} photos in {} seconds.",
            count,
            start.elapsed().as_secs()
        );

        let _ = sender.output(PhotoClipEmbedTaskOutput::Completed);
        Ok(())
    }
}

impl Worker for PhotoClipEmbedTask {
    type Init = (
        Arc<AtomicBool>,
        PathBuf,
        Thumbnailer,
        photo::Repository,
        search::Repository,
        Arc<Reducer<ProgressMonitor>>,
    );
    type Input = PhotoClipEmbedTaskInput;
    type Output = PhotoClipEmbedTaskOutput;

    fn init(
        (stop, cache_dir, thumbnailer, photo_repo, search_repo, progress_monitor): Self::Init,
        _sender: ComponentSender<Self>,
    ) -> Self {
        PhotoClipEmbedTask {
            stop,
            cache_dir,
            thumbnailer,
            photo_repo,
            search_repo,
            progress_monitor,
        }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        match msg {
            PhotoClipEmbedTaskInput::Start => {
                info!("Computing CLIP embeddings for photos...");
                let this = self.clone();
                rayon::spawn(move || {
                    if let Err(e) = this.embed(sender.clone()) {
                        error!("Failed to compute CLIP embeddings: {}", e);
                        let _ = sender.output(PhotoClipEmbedTaskOutput::Completed);
                    }
                });
            }
        };
    }
}
