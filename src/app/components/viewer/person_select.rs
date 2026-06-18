// SPDX-FileCopyrightText: © 2024 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

use relm4::adw::{self, prelude::*};
use relm4::gtk::{self, gdk};
use relm4::prelude::*;
use relm4::*;

use crate::fl;
use fotema_core::FaceId;
use fotema_core::PersonId;
use fotema_core::people;

use tracing::{debug, error};

use std::path::PathBuf;

#[derive(Debug)]
pub enum PersonSelectInput {
    /// Present the person selector for a single face.
    Activate(FaceId, PathBuf),

    /// Present the person selector for several faces at once; naming applies to
    /// all of them. The path is a representative thumbnail (the first face).
    ActivateMany(Vec<FaceId>, PathBuf),

    /// Create (or reuse) a person with the typed name for the selected face(s).
    NewPerson,

    /// "Assign person" button: assign the selected face(s) to the highlighted
    /// person in the list, or — if none is highlighted — to the typed name.
    AssignSelected,

    /// Complete the name entry to the best-matching known name (Tab key).
    Autocomplete,
}

#[derive(Debug)]
pub enum PersonSelectOutput {
    /// Face and person association either completed or dismissed.
    Done,
}

pub struct PersonSelect {
    people_repo: people::Repository,

    /// Avatar for face to associate with person.
    avatar: adw::Avatar,

    /// Shows how many faces will be named when more than one is selected.
    count_label: gtk::Label,

    /// Input box for new person name / search.
    face_name: gtk::Entry,

    /// List of avatars for people.
    people_list: gtk::ListBox,

    /// "Assign person" button (only shown in the unknown-people sidebar).
    assign_button: gtk::Button,

    /// Person ids in the same order as the rows in `people_list`, so the
    /// highlighted row can be mapped back to a person.
    all_people: Vec<PersonId>,

    /// Names of people shown in `people_list`. Used for Tab autocomplete.
    all_names: Vec<String>,

    /// Faces to associate with a person (one, or several when multi-selected).
    face_ids: Vec<FaceId>,
}

#[relm4::component(pub async)]
impl SimpleAsyncComponent for PersonSelect {
    type Init = (people::Repository, bool);
    type Input = PersonSelectInput;
    type Output = PersonSelectOutput;

    view! {
        gtk::Box {
            set_orientation: gtk::Orientation::Vertical,
            set_spacing: 8,
            set_margin_all: 12,

            #[local_ref]
            avatar -> adw::Avatar,

            #[local_ref]
            count_label -> gtk::Label {
                add_css_class: "dim-label",
                set_visible: false,
            },

            #[local_ref]
            face_name -> gtk::Entry,

            gtk::ScrolledWindow {
                set_vexpand: true,

                #[local_ref]
                people_list -> gtk::ListBox,
            },

            // Always-visible, mouse-friendly alternative to double-click/Enter:
            // assign the selected face(s) to the highlighted person (or typed name).
            // Only shown in the unknown-people sidebar (hidden in the viewer).
            #[local_ref]
            assign_button -> gtk::Button {
                set_label: &fl!("people-assign-button"),
                add_css_class: "suggested-action",
                connect_clicked => PersonSelectInput::AssignSelected,
            },
        }
    }

    async fn init(
        (people_repo, show_assign_button): Self::Init,
        root: Self::Root,
        sender: AsyncComponentSender<Self>,
    ) -> AsyncComponentParts<Self> {
        let avatar = adw::Avatar::builder()
            .size(100)
            .show_initials(false)
            .build();

        let count_label = gtk::Label::builder().build();

        let face_name = gtk::Entry::builder()
            .placeholder_text(fl!("people-person-search", "placeholder"))
            .input_purpose(gtk::InputPurpose::Name)
            .build();

        // Double-click (not single) to assign a known person: in the persistent
        // "Unknown People" sidebar the grid is clicked constantly, so a stray
        // single-click must not assign. A single click only highlights the row;
        // a double-click activates it → assigns (no Enter needed).
        let people_list = gtk::ListBox::builder()
            .css_classes(["boxed-list"])
            .activate_on_single_click(false)
            .build();

        // Double-click activates the selected row → assign that person. Use the
        // list box's reliable `row-activated` signal (a per-row `activate` on an
        // adw::ActionRow does NOT fire dependably with single-click-activate off,
        // which is why double-click appeared to do nothing).
        {
            let sender = sender.clone();
            people_list.connect_row_activated(move |_, _| {
                sender.input(PersonSelectInput::AssignSelected);
            });
        }

        // Suggest already-known names: live-filter the people list to those
        // whose name contains what the user is typing.
        people_list.set_filter_func({
            let face_name = face_name.clone();
            move |row| {
                let query = face_name.text().to_lowercase();
                let query = query.trim();
                if query.is_empty() {
                    return true;
                }
                row.downcast_ref::<adw::ActionRow>()
                    .map(|r| r.title().to_lowercase().contains(query))
                    .unwrap_or(true)
            }
        });
        {
            let people_list = people_list.clone();
            face_name.connect_changed(move |_| people_list.invalidate_filter());
        }

        // Enter creates a person with the typed name (or reuses a matching one).
        {
            let sender = sender.clone();
            face_name.connect_activate(move |_| {
                sender.input(PersonSelectInput::NewPerson);
            });
        }

        // Tab completes the entry to the best-matching known name (autocomplete),
        // instead of moving focus.
        {
            let key_controller = gtk::EventControllerKey::new();
            let sender = sender.clone();
            key_controller.connect_key_pressed(move |_, key, _, _| {
                if key == gdk::Key::Tab || key == gdk::Key::ISO_Left_Tab {
                    sender.input(PersonSelectInput::Autocomplete);
                    gtk::glib::Propagation::Stop
                } else {
                    gtk::glib::Propagation::Proceed
                }
            });
            face_name.add_controller(key_controller);
        }

        // Mouse-friendly "Assign person" button; only shown in the unknown-people
        // sidebar (the viewer's bottom-sheet selector stays as upstream).
        let assign_button = gtk::Button::new();
        assign_button.set_visible(show_assign_button);

        let widgets = view_output!();

        let model = Self {
            people_repo,
            avatar,
            count_label,
            face_name,
            people_list,
            assign_button: assign_button.clone(),
            all_people: vec![],
            all_names: vec![],
            face_ids: vec![],
        };

        AsyncComponentParts { model, widgets }
    }

    async fn update(&mut self, msg: Self::Input, sender: AsyncComponentSender<Self>) {
        match msg {
            PersonSelectInput::Activate(face_id, thumbnail) => {
                self.face_ids = vec![face_id];
                let people = self.ranked_people(Some(face_id)).await;
                self.populate(&thumbnail, people, &sender);
            }
            PersonSelectInput::ActivateMany(face_ids, thumbnail) => {
                self.face_ids = face_ids;
                let rep = self.face_ids.first().copied();
                let people = self.ranked_people(rep).await;
                self.populate(&thumbnail, people, &sender);
            }
            PersonSelectInput::NewPerson => {
                self.assign_typed_name(&sender);
            }
            PersonSelectInput::AssignSelected => {
                // Prefer the highlighted known person in the list.
                if let Some(row) = self.people_list.selected_row() {
                    let idx = row.index();
                    if idx >= 0 {
                        if let Some(person_id) = self.all_people.get(idx as usize).copied() {
                            if !self.face_ids.is_empty() {
                                self.assign_all(person_id);
                            }
                            self.finish(&sender);
                            return;
                        }
                    }
                }
                // Nobody highlighted: fall back to the typed name (like Enter).
                self.assign_typed_name(&sender);
            }
            PersonSelectInput::Autocomplete => {
                let text = self.face_name.text().to_string();
                let query = text.trim().to_lowercase();
                if query.is_empty() {
                    return;
                }
                // Prefer a name that starts with what's typed; otherwise the
                // first that contains it (matches the visible filtered list).
                let best = self
                    .all_names
                    .iter()
                    .find(|n| n.to_lowercase().starts_with(&query))
                    .or_else(|| self.all_names.iter().find(|n| n.to_lowercase().contains(&query)));
                if let Some(name) = best {
                    self.face_name.set_text(name);
                    self.face_name.set_position(-1);
                }
            }
        }
    }
}

impl PersonSelect {
    /// Rank known people by similarity to a face off the UI thread (the
    /// comparison scales with the number of named faces).
    async fn ranked_people(&self, face_id: Option<FaceId>) -> Vec<people::Person> {
        let repo = self.people_repo.clone();
        relm4::spawn_blocking(move || match face_id {
            Some(fid) => repo.people_by_similarity(fid).unwrap_or_default(),
            None => repo.all_people().unwrap_or_default(),
        })
        .await
        .unwrap_or_default()
    }

    /// Rebuild the selector for the current `face_ids`: set the preview avatar
    /// and the selection count, and fill the people list (already ranked by
    /// similarity to the first selected face).
    fn populate(
        &mut self,
        thumbnail: &PathBuf,
        people: Vec<people::Person>,
        sender: &AsyncComponentSender<Self>,
    ) {
        self.people_list.remove_all();
        self.all_people.clear();
        self.all_names.clear();
        self.face_name.set_text("");

        let n = self.face_ids.len();
        self.count_label.set_visible(n > 1);
        if n > 1 {
            self.count_label
                .set_label(&fl!("people-selected-count", count = n.to_string()));
        }

        let img = gdk::Texture::from_filename(thumbnail).ok();
        self.avatar.set_custom_image(img.as_ref());

        for person in people {
            let avatar = adw::Avatar::builder().size(50).name(&person.name).build();

            if let Some(thumbnail_path) = person.small_thumbnail_path {
                let img = gdk::Texture::from_filename(&thumbnail_path).ok();
                avatar.set_custom_image(img.as_ref());
            }

            self.all_names.push(person.name.clone());
            self.all_people.push(person.person_id);

            let row = adw::ActionRow::builder()
                .title(person.name)
                .activatable(true)
                .build();

            row.add_prefix(&avatar);

            // Activation (double-click / Enter) is handled at the list-box level
            // via `connect_row_activated` → AssignSelected (assigns the selected
            // row's person).

            self.people_list.append(&row);
        }
    }

    /// Create or reuse a person with the typed name and assign the selected
    /// face(s) to them — the Enter path, also used as the button's fallback.
    fn assign_typed_name(&mut self, sender: &AsyncComponentSender<Self>) {
        let name = self.face_name.text().to_string();
        let trimmed = name.trim();
        if trimmed.is_empty() || self.face_ids.is_empty() {
            // Nothing typed / nothing selected: keep the selector open.
            return;
        }

        // Reuse a person with this exact name, else create one.
        let person_id = match self.people_repo.find_person_id_by_name(trimmed) {
            Ok(Some(pid)) => Some(pid),
            Ok(None) => {
                let first = self.face_ids[0];
                if let Err(e) = self.people_repo.add_person(first, trimmed) {
                    error!("Failed adding new person: {:?}", e);
                    None
                } else {
                    self.people_repo
                        .find_person_id_by_name(trimmed)
                        .ok()
                        .flatten()
                }
            }
            Err(e) => {
                error!("Failed looking up person '{}': {:?}", trimmed, e);
                None
            }
        };

        if let Some(person_id) = person_id {
            self.assign_all(person_id);
        }
        self.finish(sender);
    }

    /// Associate every currently selected face with `person_id`.
    fn assign_all(&mut self, person_id: PersonId) {
        for face_id in self.face_ids.clone() {
            debug!("Associating face {} with person {}", face_id, person_id);
            if let Err(e) = self.people_repo.mark_as_person(face_id, person_id) {
                error!("Failed associating face {} with person: {:?}", face_id, e);
            }
        }
    }

    /// Reset the selector and notify the parent that naming is done.
    fn finish(&mut self, sender: &AsyncComponentSender<Self>) {
        self.people_list.remove_all();
        self.all_people.clear();
        self.all_names.clear();
        self.face_ids.clear();
        self.count_label.set_visible(false);
        let _ = sender.output(PersonSelectOutput::Done);
    }
}
