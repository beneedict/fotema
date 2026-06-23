// SPDX-FileCopyrightText: © 2024 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

use fotema_core::PictureId;
use fotema_core::VisualId;
use fotema_core::YearMonth;
use fotema_core::thumbnailify::{ThumbnailSize, Thumbnailer};

use gtk::prelude::OrientableExt;
use relm4::binding::*;
use relm4::gtk;
use relm4::gtk::gdk;
use relm4::gtk::gdk_pixbuf;
use relm4::gtk::prelude::AdjustmentExt;
use relm4::gtk::prelude::*;
use relm4::typed_view::grid::{RelmGridItem, TypedGridView};
use relm4::*;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use super::album_filter::AlbumFilter;
use super::album_sort::AlbumSort;
use crate::app::ActiveView;
use crate::app::SharedState;
use crate::app::ViewName;
use crate::app::adaptive;

use tracing::{debug, info};

const NARROW_EDGE_LENGTH: i32 = 112;
const WIDE_EDGE_LENGTH: i32 = 200;

#[derive(Debug)]
pub enum AlbumInput {
    /// Album is visible
    Activate,

    // State has been updated
    Refresh,

    /// User has selected photo in grid view
    Selected(u32), // Index into a Vec

    /// User right-clicked a picture in the grid.
    SecondaryClick(PictureId),

    /// Selection mode: when on, a single click selects (multi-select) instead of
    /// opening the photo — used for the person album's bulk review/confirm.
    SetSelectMode(bool),

    // Scroll to first photo of year/month.
    GoToMonth(YearMonth),

    // I'd like to pass a closure of Fn(Picture)->bool for the filter... but Rust
    // is making that too hard.

    // Show no photos
    Filter(AlbumFilter),

    // Sort
    Sort(AlbumSort),

    // Adapt to layout
    Adapt(adaptive::Layout),

    // Scroll offset, in pixels.
    ScrollOffset(f64),

    // Scroll to top of photo grid, regardless of sort order
    ScrollToTop,
}

#[derive(Debug)]
pub enum AlbumOutput {
    /// User has selected photo or video in grid view
    Selected(VisualId, AlbumFilter),

    /// User right-clicked a picture in the grid. Carries the current multi-
    /// selection (the right-clicked picture is always included) so the consumer
    /// can act on one or many photos at once.
    SecondaryClick(Vec<PictureId>),

    // Scroll offset, in pixels.
    ScrollOffset(f64),
}

#[derive(Debug)]
struct PhotoGridItem {
    visual: Arc<fotema_core::visual::Visual>,

    // Length of thumbnail edge to allow for resizing when layout changes.
    edge_length: I32Binding,

    thumbnailer: Rc<Thumbnailer>,

    // Channel back to the Album, used to report right-clicks.
    album_sender: relm4::Sender<AlbumInput>,
}

struct PhotoGridItemWidgets {
    picture: gtk::Picture,
    status_overlay: gtk::Frame,
    motion_type_icon: gtk::Image,
    duration_overlay: gtk::Frame,
    duration_label: gtk::Label,

    // If the gtk::Picture has been bound to edge_length.
    is_bound: bool,

    // Current right-click target (sender + picture). Updated on each bind so the
    // gesture created once in setup() always acts on the cell's current item.
    secondary: Rc<RefCell<Option<(relm4::Sender<AlbumInput>, PictureId)>>>,
}

impl RelmGridItem for PhotoGridItem {
    type Root = gtk::Frame;
    type Widgets = PhotoGridItemWidgets;

    fn setup(_item: &gtk::ListItem) -> (Self::Root, Self::Widgets) {
        relm4::view! {
            root = gtk::Frame {
                gtk::Overlay {
                    #[name(status_overlay)]
                    add_overlay =  &gtk::Frame {
                        set_halign: gtk::Align::End,
                        set_valign: gtk::Align::End,
                        set_margin_all: 8,
                        add_css_class: "photo-grid-photo-status-frame",

                        #[wrap(Some)]
                        #[name(motion_type_icon)]
                        set_child = &gtk::Image {
                            set_width_request: 16,
                            set_height_request: 16,
                            add_css_class: "photo-grid-photo-status-label",
                        },
                    },

                    #[name(duration_overlay)]
                    add_overlay =  &gtk::Frame {
                        set_halign: gtk::Align::End,
                        set_valign: gtk::Align::End,
                        set_margin_all: 8,
                        add_css_class: "photo-grid-photo-status-frame",

                        #[wrap(Some)]
                        #[name(duration_label)]
                        set_child = &gtk::Label{
                            add_css_class: "photo-grid-photo-status-label",
                        },
                    },

                    #[wrap(Some)]
                    #[name(picture)]
                    set_child = &gtk::Picture {
                        set_content_fit: gtk::ContentFit::Cover,
                        set_width_request: NARROW_EDGE_LENGTH,
                        set_height_request: NARROW_EDGE_LENGTH,
                    }
                }
            }
        }

        // Right-click a picture to act on it (e.g. set as a person's avatar).
        // The gesture is added once here; bind() points `secondary` at the
        // current item, so recycling never stacks gestures.
        let secondary: Rc<RefCell<Option<(relm4::Sender<AlbumInput>, PictureId)>>> =
            Rc::new(RefCell::new(None));
        {
            let secondary = secondary.clone();
            let gesture = gtk::GestureClick::builder()
                .button(gdk::BUTTON_SECONDARY)
                .build();
            gesture.connect_released(move |_, _, _, _| {
                if let Some((sender, picture_id)) = secondary.borrow().clone() {
                    let _ = sender.send(AlbumInput::SecondaryClick(picture_id));
                }
            });
            root.add_controller(gesture);
        }

        let widgets = PhotoGridItemWidgets {
            picture,
            status_overlay,
            motion_type_icon,
            duration_overlay,
            duration_label,
            is_bound: false,
            secondary,
        };

        (root, widgets)
    }

    fn bind(&mut self, widgets: &mut Self::Widgets, _root: &mut Self::Root) {
        // Point the right-click gesture at this item (pictures only).
        *widgets.secondary.borrow_mut() = self
            .visual
            .picture_id
            .clone()
            .map(|picture_id| (self.album_sender.clone(), picture_id));

        // Bindings to allow dynamic update of thumbnail width and height
        // when layout changes between wide and narrow

        // If we repeatedly bind, then Fotema will die with the following error:
        // (fotema:2): GLib-GObject-CRITICAL **: 13:26:14.297: Too many GWeakRef registered
        // GLib-GObject:ERROR:../gobject/gbinding.c:805:g_binding_constructed: assertion failed: (source != NULL)
        // Bail out! GLib-GObject:ERROR:../gobject/gbinding.c:805:g_binding_constructed: assertion failed: (source != NULL)

        if !widgets.is_bound {
            widgets
                .picture
                .add_write_only_binding(&self.edge_length, "width-request");
            widgets
                .picture
                .add_write_only_binding(&self.edge_length, "height-request");
            widgets.is_bound = true;
        }

        let thumbnail_size = if self.edge_length.value() == NARROW_EDGE_LENGTH {
            ThumbnailSize::Normal
        } else {
            ThumbnailSize::Large
        };

        let thumbnail_path = self
            .thumbnailer
            .nearest_thumbnail(&self.visual.thumbnail_hash(), thumbnail_size);

        if thumbnail_path.is_some() {
            widgets.picture.set_filename(thumbnail_path);

            widgets.picture.set_content_fit(gtk::ContentFit::Cover);
        } else {
            let pb = gdk_pixbuf::Pixbuf::from_resource_at_scale(
                "/app/fotema/Fotema/icons/scalable/actions/image-missing-symbolic.svg",
                200,
                200,
                true,
            )
            .unwrap();
            let img = gdk::Texture::for_pixbuf(&pb);
            widgets.picture.set_paintable(Some(&img));
            widgets.picture.set_content_fit(gtk::ContentFit::Contain);
        }

        if self.visual.is_motion_photo() {
            widgets.status_overlay.set_visible(true);
            widgets.duration_overlay.set_visible(false);
            widgets.duration_label.set_label("");
            widgets.motion_type_icon.set_icon_name(Some("cd-symbolic"));
        } else if self.visual.is_video_only() && self.visual.video_duration.is_some() {
            widgets.status_overlay.set_visible(false);
            widgets.duration_overlay.set_visible(true);

            let hhmmss = self
                .visual
                .video_duration
                .map(|ref x| fotema_core::time::format_hhmmss(x))
                .unwrap_or(String::from("—"));

            widgets.duration_label.set_label(&hhmmss);
        } else if self.visual.is_video_only() {
            widgets.status_overlay.set_visible(true);
            widgets.duration_overlay.set_visible(false);
            widgets
                .motion_type_icon
                .set_icon_name(Some("play-symbolic"));
        } else {
            // is_photo_only()
            widgets.status_overlay.set_visible(false);
            widgets.motion_type_icon.set_icon_name(None);
            widgets.duration_overlay.set_visible(false);
            widgets.duration_label.set_label("");
        }
    }

    fn unbind(&mut self, widgets: &mut Self::Widgets, _root: &mut Self::Root) {
        *widgets.secondary.borrow_mut() = None;
        widgets.picture.set_filename(None::<&Path>);
        widgets.motion_type_icon.set_icon_name(None);
        widgets.status_overlay.set_visible(false);
        widgets.duration_overlay.set_visible(false);
        widgets.duration_label.set_label("");
    }
}

pub struct Album {
    state: SharedState,
    active_view: ActiveView,
    view_name: ViewName,
    photo_grid: TypedGridView<PhotoGridItem, gtk::MultiSelection>,
    filter: AlbumFilter,
    sort: AlbumSort,
    edge_length: I32Binding,
    thumbnailer: Rc<Thumbnailer>,

    // Cloned into each grid item so it can report right-clicks.
    input_sender: relm4::Sender<AlbumInput>,
}

#[relm4::component(pub)]
impl SimpleComponent for Album {
    type Init = (
        SharedState,
        ActiveView,
        ViewName,
        AlbumFilter,
        Rc<Thumbnailer>,
    );
    type Input = AlbumInput;
    type Output = AlbumOutput;

    view! {
        gtk::ScrolledWindow {
            set_vexpand: true,

            #[local_ref]
            grid_view -> gtk::GridView {
                set_orientation: gtk::Orientation::Vertical,
                set_single_click_activate: true,

                connect_activate[sender] => move |_, idx| {
                    sender.input(AlbumInput::Selected(idx))
                },
            },

            #[wrap(Some)]
            set_vadjustment = &gtk::Adjustment {
                // Emit scroll events so PersonAlbum can determine when to hide avatar.
                // FIXME maybe just emit one event at a boundary, instead of emitting an
                // event for every scroll?
                connect_value_changed[sender] => move |v| sender.input(AlbumInput::ScrollOffset(v.value())),
            },

        }
    }

    fn init(
        (state, active_view, view_name, filter, thumbnailer): Self::Init,
        root: Self::Root,
        sender: ComponentSender<Self>,
    ) -> ComponentParts<Self> {
        let photo_grid = TypedGridView::new();
        let grid_view = &photo_grid.view.clone();

        let mut model = Album {
            state,
            active_view,
            view_name,
            photo_grid,
            filter,
            sort: AlbumSort::default(),
            edge_length: I32Binding::new(NARROW_EDGE_LENGTH),
            thumbnailer,
            input_sender: sender.input_sender().clone(),
        };

        model.update_filter();

        let widgets = view_output!();
        ComponentParts { model, widgets }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        match msg {
            AlbumInput::Activate => {
                *self.active_view.write() = self.view_name;
                if self.photo_grid.is_empty() {
                    self.refresh();
                }
            }
            AlbumInput::Refresh => {
                if *self.active_view.read() == self.view_name {
                    info!("{:?} view is active so refreshing", self.view_name);
                    self.refresh();
                } else {
                    info!("{:?} view is inactive so clearing", self.view_name);
                    self.photo_grid.clear();
                }
            }
            AlbumInput::Filter(filter) => {
                self.filter = filter;
                self.update_filter();
                // Search results must be rebuilt in relevance order (not the
                // album's date order), so rebuild the grid immediately rather than
                // only re-applying the visibility filter over date-sorted items.
                if matches!(self.filter, AlbumFilter::SearchResults(_)) {
                    self.refresh();
                }
                //self.scroll();
            }
            AlbumInput::Sort(sort) => {
                if self.sort != sort {
                    info!("Sort order is now {:?}", sort);
                    self.sort = sort;
                    sender.input(AlbumInput::Refresh);
                }
            }
            AlbumInput::Selected(index) => {
                // Albums are filters so must use get_visible(...) over get(...), otherwise
                // wrong photo is displayed.
                if let Some(item) = self.photo_grid.get_visible(index) {
                    let visual_id = item.borrow().visual.visual_id.clone();
                    debug!("index {} has visual_id {}", index, visual_id);
                    let _ = sender.output(AlbumOutput::Selected(visual_id, self.filter.clone()));
                }
            }
            AlbumInput::SecondaryClick(picture_id) => {
                // Act on the whole multi-selection when the right-clicked picture
                // is part of it; otherwise just the right-clicked one.
                let mut ids = self.selected_picture_ids();
                if !ids.contains(&picture_id) {
                    ids = vec![picture_id];
                }
                let _ = sender.output(AlbumOutput::SecondaryClick(ids));
            }
            AlbumInput::SetSelectMode(select) => {
                // In select mode a single click selects (for multi-select);
                // otherwise it opens the photo.
                self.photo_grid.view.set_single_click_activate(!select);
                if !select {
                    self.photo_grid.selection_model.unselect_all();
                }
            }
            AlbumInput::GoToMonth(ym) => {
                info!("Showing for month: {}", ym);
                let index_opt = self.photo_grid.find(|p| p.visual.year_month() == ym);
                if let Some(index) = index_opt {
                    let flags = gtk::ListScrollFlags::SELECT;
                    debug!("Scrolling to {}", index);
                    self.photo_grid.view.scroll_to(index, flags, None);
                }
            }
            AlbumInput::ScrollToTop => {
                // Hmm... not sure I like this...
                if !self.photo_grid.is_empty() {
                    self.photo_grid
                        .view
                        .scroll_to(0, gtk::ListScrollFlags::SELECT, None);
                }
            }
            AlbumInput::Adapt(adaptive::Layout::Narrow) => {
                self.edge_length.set_value(NARROW_EDGE_LENGTH);
            }
            AlbumInput::Adapt(adaptive::Layout::Wide) => {
                self.edge_length.set_value(WIDE_EDGE_LENGTH);
            }
            AlbumInput::ScrollOffset(offset) => {
                let _ = sender.output(AlbumOutput::ScrollOffset(offset));
            }
        }
    }
}

impl Album {
    /// Picture ids of the currently multi-selected grid items. Reads the
    /// selection bitset (visible positions) and maps via `get_visible`, because
    /// the album filters its backing store (store positions != visible ones).
    fn selected_picture_ids(&self) -> Vec<PictureId> {
        let mut ids = Vec::new();
        let selection = self.photo_grid.selection_model.selection();
        for i in 0..selection.size() {
            let pos = selection.nth(i as u32);
            if let Some(item) = self.photo_grid.get_visible(pos) {
                if let Some(picture_id) = item.borrow().visual.picture_id.clone() {
                    ids.push(picture_id);
                }
            }
        }
        ids
    }

    fn refresh(&mut self) {
        let mut all = {
            let data = self.state.read();
            data.iter()
                .map(|visual| PhotoGridItem {
                    visual: visual.clone(),
                    edge_length: self.edge_length.clone(),
                    thumbnailer: self.thumbnailer.clone(),
                    album_sender: self.input_sender.clone(),
                })
                .collect::<Vec<PhotoGridItem>>()
        };

        if let AlbumFilter::SearchResults(ref ids) = self.filter {
            // Relevance order: keep only matched pictures and order them by their
            // position in `ids` (best match first). PictureId isn't Hash, so key
            // the rank lookup by the inner i64.
            let rank: std::collections::HashMap<i64, usize> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| (id.id(), i))
                .collect();
            all.retain(|item| {
                item.visual
                    .picture_id
                    .as_ref()
                    .is_some_and(|pid| rank.contains_key(&pid.id()))
            });
            all.sort_by_key(|item| {
                item.visual
                    .picture_id
                    .as_ref()
                    .and_then(|pid| rank.get(&pid.id()))
                    .copied()
                    .unwrap_or(usize::MAX)
            });
        } else {
            // State is always in ascending time order
            self.sort.sort(&mut all);
        }

        self.photo_grid.clear();

        //self.photo_grid.add_filter(move |item| (self.photo_grid_filter)(&item.picture));
        self.photo_grid.extend_from_iter(all);

        info!("{} items added to album", self.photo_grid.len());

        if matches!(self.filter, AlbumFilter::SearchResults(_)) {
            // Best match first, so show the top of the list.
            if !self.photo_grid.is_empty() {
                self.photo_grid
                    .view
                    .scroll_to(0, gtk::ListScrollFlags::SELECT, None);
            }
        } else {
            // NOTE person album will in effect overide scrolling to the end
            // by sending a ScrollToTop command.
            self.scroll_to_end();
        }
    }

    /// Scroll to the last item of the *filtered* view.
    ///
    /// Must index into the filtered/selection model (visible items), NOT the
    /// backing store: on a filtered album like Videos the store holds all 33k
    /// visuals but only the videos are visible, so scrolling to `store.len()-1`
    /// targets an out-of-range index and the view stays at the top showing just
    /// the first matching item.
    ///
    /// The filtered count is only settled *after* this update returns (reading it
    /// synchronously is stale and scrolls to the top), so defer the scroll to the
    /// next idle and read the visible count then.
    fn scroll_to_end(&mut self) {
        let view = self.photo_grid.view.clone();
        let selection = self.photo_grid.selection_model.clone();
        let sort = self.sort;
        gtk::glib::idle_add_local_once(move || {
            let visible = selection.n_items();
            info!("scroll_to_end: {} visible items", visible);
            if visible == 0 {
                return;
            }
            let index = match sort {
                AlbumSort::Ascending => visible - 1,
                AlbumSort::Descending => 0,
            };
            view.scroll_to(index, gtk::ListScrollFlags::SELECT, None);
        });
    }

    fn update_filter(&mut self) {
        self.photo_grid.clear_filters();
        let filter = self.filter.clone();
        self.photo_grid
            .add_filter(move |item| filter.clone().filter(&item.visual));
    }
}
