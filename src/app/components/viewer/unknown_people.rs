// SPDX-FileCopyrightText: © 2024 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

use gtk::prelude::OrientableExt;
use relm4::RelmObjectExt;
use relm4::adw::{self, prelude::*};
use relm4::binding::*;
use relm4::gtk::{self, gdk};
use relm4::prelude::*;
use relm4::typed_view::grid::{RelmGridItem, TypedGridView};

use crate::adaptive;
use crate::fl;
use fotema_core::people;

use super::person_select::{PersonSelect, PersonSelectInput, PersonSelectOutput};

use tracing::{debug, error};

const NARROW_EDGE_LENGTH: i32 = 170;
const WIDE_EDGE_LENGTH: i32 = 200;

pub struct UnknownFaceItem {
    face: people::Face,
    edge_length: I32Binding,
}

pub struct UnknownFaceWidgets {
    avatar: adw::Avatar,
    is_bound: bool,
}

impl RelmGridItem for UnknownFaceItem {
    type Root = gtk::Box;
    type Widgets = UnknownFaceWidgets;

    fn setup(_item: &gtk::ListItem) -> (Self::Root, Self::Widgets) {
        relm4::view! {
            root = gtk::Box {
                set_orientation: gtk::Orientation::Vertical,
                set_margin_all: 4,
                set_halign: gtk::Align::Center,
                set_valign: gtk::Align::Center,

                #[name(avatar)]
                adw::Avatar {
                    set_size: NARROW_EDGE_LENGTH,
                    set_show_initials: false,
                }
            }
        }

        (root, UnknownFaceWidgets { avatar, is_bound: false })
    }

    fn bind(&mut self, widgets: &mut Self::Widgets, _root: &mut Self::Root) {
        if !widgets.is_bound {
            widgets
                .avatar
                .add_write_only_binding(&self.edge_length, "size");
            widgets.is_bound = true;
        }

        let img = gdk::Texture::from_filename(&self.face.thumbnail_path).ok();
        widgets.avatar.set_custom_image(img.as_ref());
    }

    fn unbind(&mut self, widgets: &mut Self::Widgets, _root: &mut Self::Root) {
        widgets.avatar.set_custom_image(None::<&gdk::Paintable>);
    }
}

#[derive(Debug)]
pub enum UnknownPeopleInput {
    Refresh,
    SelectionChanged,
    PersonSelected,
    Adapt(adaptive::Layout),
}

#[derive(Debug)]
pub enum UnknownPeopleOutput {
    /// A face was named; the people overview should refresh.
    Changed,
}

pub struct UnknownPeople {
    people_repo: people::Repository,
    face_grid: TypedGridView<UnknownFaceItem, gtk::MultiSelection>,
    avatars: gtk::ScrolledWindow,
    status: adw::StatusPage,
    edge_length: I32Binding,
    person_select: AsyncController<PersonSelect>,
}

#[relm4::component(pub async)]
impl SimpleAsyncComponent for UnknownPeople {
    type Init = people::Repository;
    type Input = UnknownPeopleInput;
    type Output = UnknownPeopleOutput;

    view! {
        adw::OverlaySplitView {
            set_collapsed: false,
            set_sidebar_position: gtk::PackType::End,
            set_min_sidebar_width: 320.0,
            set_max_sidebar_width: 360.0,

            #[wrap(Some)]
            set_content = &gtk::Box {
                set_orientation: gtk::Orientation::Vertical,

                #[local_ref]
                avatars -> gtk::ScrolledWindow {
                    set_hexpand: true,
                    set_vexpand: true,

                    #[local_ref]
                    grid_view -> gtk::GridView {
                        remove_css_class: "view",
                        set_orientation: gtk::Orientation::Vertical,
                    },
                },

                #[local_ref]
                status -> adw::StatusPage {
                    set_valign: gtk::Align::Start,
                    set_vexpand: true,
                    set_visible: false,
                    set_icon_name: Some("sentiment-very-satisfied-symbolic"),
                },
            },

            #[wrap(Some)]
            set_sidebar = &gtk::Box {
                set_orientation: gtk::Orientation::Vertical,

                #[local_ref]
                person_select_widget -> gtk::Box {},
            },
        }
    }

    async fn init(
        people_repo: Self::Init,
        root: Self::Root,
        sender: AsyncComponentSender<Self>,
    ) -> AsyncComponentParts<Self> {
        let face_grid: TypedGridView<UnknownFaceItem, gtk::MultiSelection> = TypedGridView::new();
        let grid_view = &face_grid.view.clone();

        {
            let sender = sender.clone();
            face_grid
                .selection_model
                .connect_selection_changed(move |_, _, _| {
                    sender.input(UnknownPeopleInput::SelectionChanged);
                });
        }

        let avatars = gtk::ScrolledWindow::builder().build();
        let status = adw::StatusPage::new();
        status.set_title(&fl!("faces-page-empty", "title"));
        status.set_description(Some(&fl!("faces-page-empty", "description")));

        let person_select = PersonSelect::builder()
            .launch(people_repo.clone())
            .forward(sender.input_sender(), |msg| match msg {
                PersonSelectOutput::Done => UnknownPeopleInput::PersonSelected,
            });
        let person_select_widget = person_select.widget().clone();

        let widgets = view_output!();

        let model = Self {
            people_repo,
            face_grid,
            avatars,
            status,
            edge_length: I32Binding::new(NARROW_EDGE_LENGTH),
            person_select,
        };

        sender.input(UnknownPeopleInput::Refresh);
        let _ = root;
        AsyncComponentParts { model, widgets }
    }

    async fn update(&mut self, msg: Self::Input, sender: AsyncComponentSender<Self>) {
        match msg {
            UnknownPeopleInput::Refresh => {
                self.face_grid.clear();

                let repo = self.people_repo.clone();
                let faces = relm4::spawn_blocking(move || {
                    repo.find_unnamed_faces()
                        .unwrap_or_default()
                        .into_iter()
                        .filter(|face| face.thumbnail_path.exists())
                        .collect::<Vec<_>>()
                })
                .await
                .unwrap_or_else(|e| {
                    error!("Unknown faces task failed: {}", e);
                    Vec::new()
                });

                debug!("Found {} unnamed faces", faces.len());
                let edge = self.edge_length.clone();
                self.face_grid
                    .extend_from_iter(faces.into_iter().map(|face| UnknownFaceItem {
                        face,
                        edge_length: edge.clone(),
                    }));

                let is_empty = self.face_grid.len() == 0;
                self.avatars.set_visible(!is_empty);
                self.status.set_visible(is_empty);
            }
            UnknownPeopleInput::SelectionChanged => {
                for position in 0..self.face_grid.len() {
                    if self.face_grid.selection_model.is_selected(position) {
                        if let Some(item) = self.face_grid.get(position) {
                            let face = &item.borrow().face;
                            self.person_select.emit(PersonSelectInput::Activate(
                                face.face_id,
                                face.thumbnail_path.clone(),
                            ));
                        }
                        break;
                    }
                }
            }
            UnknownPeopleInput::PersonSelected => {
                sender.input(UnknownPeopleInput::Refresh);
                let _ = sender.output(UnknownPeopleOutput::Changed);
            }
            UnknownPeopleInput::Adapt(adaptive::Layout::Narrow) => {
                self.edge_length.set_value(NARROW_EDGE_LENGTH);
            }
            UnknownPeopleInput::Adapt(adaptive::Layout::Wide) => {
                self.edge_length.set_value(WIDE_EDGE_LENGTH);
            }
        }
    }
}
