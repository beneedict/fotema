// SPDX-FileCopyrightText: © 2025 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Smart-search page: a search entry plus an embedded photo grid that shows the
// CLIP/person results in relevance order. The actual ranking happens in the
// background `ClipSearchTask` worker; this component just wires the entry to the
// worker and the worker's results to the grid.

use std::path::PathBuf;
use std::rc::Rc;

use relm4::adw;
use relm4::adw::prelude::*;
use relm4::gtk;
use relm4::prelude::*;
use relm4::*;

use crate::app::ActiveView;
use crate::app::SharedState;
use crate::app::ViewName;
use crate::app::background::clip_search_task::{
    ClipSearchTask, ClipSearchTaskInput, ClipSearchTaskOutput,
};
use crate::app::components::albums::{
    album::{Album, AlbumInput, AlbumOutput},
    album_filter::AlbumFilter,
};
use crate::fl;

use fotema_core::PictureId;
use fotema_core::VisualId;
use fotema_core::people;
use fotema_core::search;
use fotema_core::thumbnailify::Thumbnailer;

#[derive(Debug)]
pub enum SearchAlbumInput {
    /// Page became visible.
    Activate,
    /// The search entry text changed (already debounced by gtk::SearchEntry).
    Search(String),
    /// Ranked results for `query` arrived from the worker.
    Results(String, Vec<PictureId>),
    /// A photo in the grid was selected.
    Selected(VisualId),
    /// Export the current search results to a folder.
    Export,
}

#[derive(Debug)]
pub enum SearchAlbumOutput {
    /// Open the selected photo, carrying the current filter so the viewer can
    /// page through the search results.
    Selected(VisualId, AlbumFilter),
    /// Export these pictures (the current search results) to a folder.
    Export(Vec<PictureId>),
}

pub struct SearchAlbum {
    album: Controller<Album>,
    search_task: WorkerController<ClipSearchTask>,
    /// The most recent query, used to drop stale (out-of-order) worker results.
    current_query: String,
    /// The currently-displayed ranked results, so the viewer can page through them.
    current_results: Vec<PictureId>,
}

#[relm4::component(pub)]
impl SimpleComponent for SearchAlbum {
    type Init = (
        SharedState,
        ActiveView,
        Rc<Thumbnailer>,
        PathBuf,
        search::Repository,
        people::Repository,
    );
    type Input = SearchAlbumInput;
    type Output = SearchAlbumOutput;

    view! {
        adw::ToolbarView {
            add_top_bar = &adw::HeaderBar {
                #[wrap(Some)]
                #[local_ref]
                set_title_widget = &search_entry -> gtk::SearchEntry {
                    set_hexpand: true,
                    set_placeholder_text: Some(&fl!("search-placeholder")),
                    connect_search_changed[sender] => move |entry| {
                        sender.input(SearchAlbumInput::Search(entry.text().to_string()));
                    },
                },

                pack_end = &gtk::Button {
                    set_icon_name: "document-save-symbolic",
                    set_tooltip_text: Some(&fl!("export-tooltip")),
                    connect_clicked => SearchAlbumInput::Export,
                },
            },

            #[wrap(Some)]
            set_content = &gtk::Box {
                set_orientation: gtk::Orientation::Vertical,
                set_vexpand: true,

                model.album.widget(),
            }
        }
    }

    fn init(
        (state, active_view, thumbnailer, cache_dir, search_repo, people_repo): Self::Init,
        root: Self::Root,
        sender: ComponentSender<Self>,
    ) -> ComponentParts<Self> {
        let album = Album::builder()
            .launch((
                state.clone(),
                active_view.clone(),
                ViewName::Search,
                AlbumFilter::None,
                thumbnailer,
            ))
            .forward(sender.input_sender(), |msg| match msg {
                AlbumOutput::Selected(id, _) => SearchAlbumInput::Selected(id),
                AlbumOutput::SecondaryClick(_) => SearchAlbumInput::Activate,
                AlbumOutput::ScrollOffset(_) => SearchAlbumInput::Activate,
            });

        let search_task = ClipSearchTask::builder()
            .detach_worker((cache_dir, search_repo, people_repo))
            .forward(sender.input_sender(), |msg| match msg {
                ClipSearchTaskOutput::Results(query, results) => SearchAlbumInput::Results(
                    query,
                    results.into_iter().map(|(id, _score)| id).collect(),
                ),
            });

        let search_entry = gtk::SearchEntry::builder().build();

        let model = SearchAlbum {
            album,
            search_task,
            current_query: String::new(),
            current_results: Vec::new(),
        };

        let widgets = view_output!();
        ComponentParts { model, widgets }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        match msg {
            SearchAlbumInput::Activate => {
                self.album.emit(AlbumInput::Activate);
            }
            SearchAlbumInput::Search(query) => {
                self.current_query = query.clone();
                if query.trim().is_empty() {
                    // Clear results when the entry is emptied.
                    self.current_results.clear();
                    self.album
                        .emit(AlbumInput::Filter(AlbumFilter::SearchResults(Vec::new())));
                } else {
                    self.search_task.emit(ClipSearchTaskInput::Query(query));
                }
            }
            SearchAlbumInput::Results(query, ids) => {
                // Ignore results for a query the user has since changed.
                if query == self.current_query {
                    self.current_results = ids.clone();
                    self.album
                        .emit(AlbumInput::Filter(AlbumFilter::SearchResults(ids)));
                }
            }
            SearchAlbumInput::Selected(visual_id) => {
                let _ = sender.output(SearchAlbumOutput::Selected(
                    visual_id,
                    AlbumFilter::SearchResults(self.current_results.clone()),
                ));
            }
            SearchAlbumInput::Export => {
                if !self.current_results.is_empty() {
                    let _ = sender.output(SearchAlbumOutput::Export(self.current_results.clone()));
                }
            }
        }
    }
}
