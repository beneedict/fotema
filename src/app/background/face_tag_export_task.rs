// SPDX-FileCopyrightText: © 2026 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Writes the confirmed person names of a picture into the XMP sidecar file of
// that picture. A file sync tool then carries the names to another computer,
// where the import assigns the same people again. The user therefore names each
// person once, not once for each computer.
//
// The task keeps every other part of a sidecar. `fotema_core::photo::face_tags`
// protects the regions of another program: it replaces only the regions that
// this computer's own faces cover, and leaves the rest of the sidecar as it was.

use anyhow::*;
use relm4::Worker;
use relm4::prelude::*;
use std::result::Result::Ok;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{error, info, warn};

use fotema_core::photo::detection_size::DetectionSize;
use fotema_core::photo::face_tags::{self, FaceTag, TagArea, WriteOutcome};
use fotema_core::thumbnailify::Thumbnailer;

#[derive(Debug)]
pub enum FaceTagExportTaskInput {
    Start,
}

#[derive(Debug)]
pub enum FaceTagExportTaskOutput {
    Started,
    /// The run is over. A sidecar changes nothing that the library shows, so the
    /// output carries no count.
    Completed,
}

pub struct FaceTagExportTask {
    // Stop flag
    stop: Arc<AtomicBool>,

    repo: fotema_core::photo::Repository,

    thumbnailer: Thumbnailer,
}

impl FaceTagExportTask {
    fn export(&self, sender: &ComponentSender<Self>) -> Result<usize> {
        let start = std::time::Instant::now();

        let pending = self.repo.find_pictures_for_face_tag_export()?;
        if pending.is_empty() {
            return Ok(0);
        }

        let total = pending.len();
        info!("Found {} pictures to write face tags for", total);
        let _ = sender.output(FaceTagExportTaskOutput::Started);

        let mut repo = self.repo.clone();
        let mut written = 0usize;

        for pic in pending {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }

            if !pic.path.sandbox_path.exists() {
                // The picture is gone. Mark it, so the export does not look at
                // it again on the next run.
                if let Err(e) = repo.mark_face_tags_exported(&[pic.picture_id]) {
                    error!("Failed to mark {:?} as exported: {}", pic.path.host_path, e);
                }
                continue;
            }

            let size = DetectionSize::new(&self.thumbnailer, &pic.path, pic.orientation);

            let mut missing_size = false;
            let mut detected: Vec<TagArea> = Vec::with_capacity(pic.faces.len());
            let mut named: Vec<FaceTag> = Vec::new();

            for face in &pic.faces {
                let Some((width, height)) = size.size_of(face.is_source_original) else {
                    missing_size = true;
                    continue;
                };

                let Some(area) = TagArea::from_pixel_bounds(
                    face.bounds.x,
                    face.bounds.y,
                    face.bounds.width,
                    face.bounds.height,
                    width,
                    height,
                )
                .and_then(TagArea::clamped) else {
                    continue;
                };

                detected.push(area);
                if let Some(name) = face.name.clone() {
                    named.push(FaceTag {
                        name,
                        center_x: area.center_x,
                        area: Some(area),
                    });
                }
            }

            if missing_size {
                warn!(
                    "No detection size for one or more faces of {:?}, those faces were skipped",
                    pic.path.host_path
                );
            }

            match face_tags::write_face_tags(&pic.path.sandbox_path, &named, &detected) {
                Ok(WriteOutcome::Written) => written += 1,
                Ok(WriteOutcome::Unchanged) => {}
                Err(e) => {
                    // A single picture must not stop the whole export, and the
                    // sidecar is still out of date, so leave it unmarked.
                    error!(
                        "Failed to write face tags for {:?}: {}",
                        pic.path.host_path, e
                    );
                    continue;
                }
            }

            if let Err(e) = repo.mark_face_tags_exported(&[pic.picture_id]) {
                error!("Failed to mark {:?} as exported: {}", pic.path.host_path, e);
            }
        }

        info!(
            "Wrote face tags for {} of {} pictures in {:?}",
            written,
            total,
            start.elapsed()
        );

        Ok(written)
    }
}

impl Worker for FaceTagExportTask {
    type Init = (Arc<AtomicBool>, fotema_core::photo::Repository, Thumbnailer);
    type Input = FaceTagExportTaskInput;
    type Output = FaceTagExportTaskOutput;

    fn init((stop, repo, thumbnailer): Self::Init, _sender: ComponentSender<Self>) -> Self {
        Self {
            stop,
            repo,
            thumbnailer,
        }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        if self.stop.load(Ordering::Relaxed) {
            let _ = sender.output(FaceTagExportTaskOutput::Completed);
            return;
        }

        match msg {
            FaceTagExportTaskInput::Start => {
                info!("Writing face tags to sidecar files...");

                if let Err(e) = self.export(&sender) {
                    error!("Failed to write face tags: {}", e);
                }

                let _ = sender.output(FaceTagExportTaskOutput::Completed);
            }
        }
    }
}
