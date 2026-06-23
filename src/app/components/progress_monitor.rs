// SPDX-FileCopyrightText: © 2024 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

use relm4::Reducible;

/// Media types
#[derive(Debug, Clone, Copy)]
pub enum MediaType {
    Photo,
    Video,
}

#[derive(Debug, Clone, Copy)]
pub enum ThumbnailType {
    Photo,
    Video,
    Face,
}

/// Different kinds of background task that have a progress bar
/// Note that some background tasks just have the banner and spinner.
#[derive(Debug, Clone, Copy)]
pub enum TaskName {
    Scan,
    Enrich(MediaType),
    Thumbnail(ThumbnailType),
    Transcode,
    MotionPhoto,
    DetectFaces,
    RecognizeFaces,
    ClipEmbed,

    /// FIXME figure out if 'Idle' will be used.
    Idle,
}

#[derive(Debug)]
pub enum ProgressMonitorInput {
    /// Begin a task with a known number of items (determinate progress bar).
    Start(TaskName, usize),
    /// Begin a task whose total is unknown (indeterminate / pulsing progress bar),
    /// e.g. the filesystem scan, which discovers files as it goes.
    StartPulse(TaskName),
    Advance,
    Complete,
}

/// Monitors the progress of a task and informs subscribers about changes.
pub struct ProgressMonitor {
    // Background task progress is for. None if idle.
    pub task_name: TaskName,

    /// Current progress
    pub current_count: usize,

    // Final progress
    end_count: usize,

    /// Whether the current task has an unknown total (pulsing bar).
    indeterminate: bool,

    /// Whether the current task has finished (bar should hide).
    finished: bool,
}

impl ProgressMonitor {
    pub fn fraction(&self) -> f64 {
        if self.end_count == 0 {
            0.0
        } else {
            self.current_count as f64 / self.end_count as f64
        }
    }

    /// Task total is unknown — the bar should pulse rather than show a fraction.
    pub fn is_indeterminate(&self) -> bool {
        self.indeterminate
    }

    /// Task has finished — the bar should be hidden.
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

impl Reducible for ProgressMonitor {
    type Input = ProgressMonitorInput;

    fn init() -> Self {
        Self {
            task_name: TaskName::Idle,
            current_count: 0,
            end_count: 0,
            indeterminate: false,
            finished: true,
        }
    }

    fn reduce(&mut self, input: Self::Input) -> bool {
        match input {
            ProgressMonitorInput::Start(task_name, end_count) => {
                self.task_name = task_name;
                self.end_count = end_count;
                self.current_count = 0;
                self.indeterminate = false;
                self.finished = false;
            }
            ProgressMonitorInput::StartPulse(task_name) => {
                self.task_name = task_name;
                self.end_count = 0;
                self.current_count = 0;
                self.indeterminate = true;
                self.finished = false;
            }
            ProgressMonitorInput::Advance => {
                if self.current_count < self.end_count {
                    self.current_count += 1;
                }
            }
            ProgressMonitorInput::Complete => {
                self.current_count = self.end_count;
                self.finished = true;
            }
        }
        true // subscribers only notified if 'true' is returned
    }
}
