// SPDX-FileCopyrightText: © 2024 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

use relm4::gtk;
use relm4::gtk::glib;
use relm4::gtk::prelude::WidgetExt;
use relm4::shared_state::Reducer;
use relm4::*;

use std::sync::Arc;
use std::time::Duration;

use super::progress_monitor::{MediaType, ProgressMonitor, TaskName, ThumbnailType};
use crate::fl;

/// How often the pulsing (indeterminate) progress bar advances.
const PULSE_INTERVAL: Duration = Duration::from_millis(120);

#[derive(Debug)]
pub enum ProgressPanelInput {
    Update {
        task_name: TaskName,
        fraction: f64,
        indeterminate: bool,
        finished: bool,
    },
}

/// Shows progress of a background task
pub struct ProgressPanel {
    progress_bar: gtk::ProgressBar,
    /// Active timer driving the pulsing bar (for indeterminate tasks), if any.
    pulse_source: Option<glib::SourceId>,
}

impl ProgressPanel {
    /// Human-readable label for the task currently shown in the bar.
    fn label(task_name: TaskName) -> String {
        match task_name {
            TaskName::Scan => fl!("progress-scan-photos"),
            TaskName::Enrich(MediaType::Photo) => fl!("progress-metadata-photos"),
            TaskName::Enrich(MediaType::Video) => fl!("progress-metadata-videos"),
            TaskName::Thumbnail(ThumbnailType::Photo) => fl!("progress-thumbnails-photos"),
            TaskName::Thumbnail(ThumbnailType::Video) => fl!("progress-thumbnails-videos"),
            TaskName::Thumbnail(ThumbnailType::Face) => fl!("progress-thumbnails-faces"),
            TaskName::Transcode => fl!("progress-convert-videos"),
            TaskName::MotionPhoto => fl!("progress-motion-photo"),
            TaskName::DetectFaces => fl!("progress-detect-faces-photos"),
            TaskName::RecognizeFaces => fl!("progress-recognize-faces-photos"),
            TaskName::ClipEmbed => fl!("progress-clip-embed-photos"),
            TaskName::Idle => fl!("progress-idle"),
        }
    }

    /// Start the pulsing timer if it isn't already running.
    fn start_pulse(&mut self) {
        if self.pulse_source.is_none() {
            let bar = self.progress_bar.clone();
            let id = glib::timeout_add_local(PULSE_INTERVAL, move || {
                bar.pulse();
                glib::ControlFlow::Continue
            });
            self.pulse_source = Some(id);
        }
    }

    /// Stop the pulsing timer if it is running.
    fn stop_pulse(&mut self) {
        if let Some(id) = self.pulse_source.take() {
            id.remove();
        }
    }
}

#[relm4::component(pub)]
impl SimpleComponent for ProgressPanel {
    type Init = Arc<Reducer<ProgressMonitor>>;
    type Input = ProgressPanelInput;
    type Output = ();

    view! {
        gtk::ProgressBar {
            set_margin_all: 12,
            set_visible: false,
            set_show_text: true,
            set_pulse_step: 0.05,
        }
    }

    fn init(
        progress_monitor: Self::Init,
        progress_bar: Self::Root,
        sender: ComponentSender<Self>,
    ) -> ComponentParts<Self> {
        progress_monitor.subscribe(sender.input_sender(), |data| ProgressPanelInput::Update {
            task_name: data.task_name,
            fraction: data.fraction(),
            indeterminate: data.is_indeterminate(),
            finished: data.is_finished(),
        });

        let model = ProgressPanel {
            progress_bar: progress_bar.clone(),
            pulse_source: None,
        };

        let widgets = view_output!();

        ComponentParts { model, widgets }
    }

    fn update(&mut self, msg: Self::Input, _sender: ComponentSender<Self>) {
        match msg {
            ProgressPanelInput::Update {
                task_name,
                fraction,
                indeterminate,
                finished,
            } => {
                if finished {
                    self.stop_pulse();
                    self.progress_bar.set_visible(false);
                    self.progress_bar.set_text(None);
                    return;
                }

                self.progress_bar.set_visible(true);
                self.progress_bar.set_text(Some(&Self::label(task_name)));

                if indeterminate {
                    self.start_pulse();
                } else {
                    self.stop_pulse();
                    self.progress_bar.set_fraction(fraction);
                }
            }
        }
    }
}
