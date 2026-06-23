// SPDX-FileCopyrightText: © 2024-2025 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

use fotema_core::photo::Repository as PhotoRepository;
use fotema_core::video::Repository as VideoRepository;
use fotema_core::{ScannedFile, Scanner};
use itertools::{Either, Itertools};
use relm4::Reducer;
use relm4::Worker;
use relm4::prelude::*;
use std::sync::Arc;
use tracing::{error, info};

use crate::app::components::progress_monitor::{ProgressMonitor, ProgressMonitorInput, TaskName};

#[derive(Debug)]
pub enum LibraryScanTaskInput {
    Start,
}

#[derive(Debug)]
pub enum LibraryScanTaskOutput {
    Started,
    Completed,
}

pub struct LibraryScanTask {
    scan: Scanner,
    photo_repo: PhotoRepository,
    video_repo: VideoRepository,
    progress_monitor: Arc<Reducer<ProgressMonitor>>,
}

impl Worker for LibraryScanTask {
    type Init = (
        Scanner,
        PhotoRepository,
        VideoRepository,
        Arc<Reducer<ProgressMonitor>>,
    );
    type Input = LibraryScanTaskInput;
    type Output = LibraryScanTaskOutput;

    fn init(
        (scan, photo_repo, video_repo, progress_monitor): Self::Init,
        _sender: ComponentSender<Self>,
    ) -> Self {
        Self {
            scan,
            photo_repo,
            video_repo,
            progress_monitor,
        }
    }

    fn update(&mut self, msg: LibraryScanTaskInput, sender: ComponentSender<Self>) {
        match msg {
            LibraryScanTaskInput::Start => {
                let result = self.scan_and_add(sender);
                if let Err(e) = result {
                    error!("Failed scan with: {}", e);
                }
            }
        };
    }
}

impl LibraryScanTask {
    fn scan_and_add(&mut self, sender: ComponentSender<Self>) -> std::result::Result<(), String> {
        let start = std::time::Instant::now();

        sender
            .output(LibraryScanTaskOutput::Started)
            .map_err(|e| format!("{:?}", e))?;

        info!("Scanning file system for pictures...");

        // The scan discovers files as it walks, so the total is unknown up front:
        // show a pulsing bar for the duration.
        self.progress_monitor
            .emit(ProgressMonitorInput::StartPulse(TaskName::Scan));

        let result = self.scan.scan_all().map_err(|e| e.to_string())?;

        let (photos, videos) =
            result
                .into_iter()
                .partition_map(|scanned_file| match scanned_file {
                    f @ ScannedFile::Photo(_) => Either::Left(f),
                    f @ ScannedFile::Video(_) => Either::Right(f),
                });

        self.photo_repo
            .add_all(&photos)
            .map_err(|e| e.to_string())?;
        self.video_repo
            .add_all(&videos)
            .map_err(|e| e.to_string())?;

        info!(
            "Scanned {} photos and {} videos in {} seconds.",
            photos.len(),
            videos.len(),
            start.elapsed().as_secs()
        );

        self.progress_monitor
            .emit(ProgressMonitorInput::Complete);

        sender
            .output(LibraryScanTaskOutput::Completed)
            .map_err(|e| format!("{:?}", e))
    }
}
