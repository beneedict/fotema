// SPDX-FileCopyrightText: © 2026 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The pixel size of the image that the face detector looked at. The bounds of a
// detected face are in the pixels of that image. To compare a face with a
// region of an XMP file, or to write it as a region, the face must be divided by
// this size.

use std::cell::OnceCell;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::FlatpakPathBuf;
use crate::photo::model::Orientation;
use crate::thumbnailify::{ThumbnailSize, Thumbnailer};

/// The face detector looks at the XLarge thumbnail of a picture. An older build
/// looked at the original picture; such a face has `is_source_original` set.
///
/// The struct reads each size only when a face needs it, and only once. Most
/// pictures need no size at all, because the import has no region to compare.
pub struct DetectionSize {
    original_path: PathBuf,
    orientation: Option<Orientation>,
    thumbnail_path: Option<PathBuf>,
    original: OnceCell<Option<(f32, f32)>>,
    thumbnail: OnceCell<Option<(f32, f32)>>,
}

impl DetectionSize {
    /// `orientation` is the EXIF orientation of the original picture, as the
    /// database holds it. The detector looked at the picture after the
    /// orientation was applied, so a turned picture swaps width and height.
    pub fn new(
        thumbnailer: &Thumbnailer,
        path: &FlatpakPathBuf,
        orientation: Option<Orientation>,
    ) -> Self {
        Self {
            original_path: path.sandbox_path.clone(),
            orientation,
            thumbnail_path: thumbnailer
                .existing_thumbnail(&path.thumbnail_hash(), ThumbnailSize::XLarge),
            original: OnceCell::new(),
            thumbnail: OnceCell::new(),
        }
    }

    /// The width and the height of the image that the detector looked at for a
    /// face. None when the file cannot be read.
    pub fn size_of(&self, is_source_original: bool) -> Option<(f32, f32)> {
        if is_source_original {
            *self.original.get_or_init(|| {
                let (w, h) = image_dimensions(&self.original_path)?;
                Some(if self.orientation.is_some_and(Orientation::swaps_sides) {
                    (h, w)
                } else {
                    (w, h)
                })
            })
        } else {
            *self
                .thumbnail
                .get_or_init(|| image_dimensions(self.thumbnail_path.as_deref()?))
        }
    }
}

/// Read the width and the height from the header of an image file.
///
/// The function reads the first part of the file only. The decoder for JPEG
/// otherwise takes the whole file into memory to find the size. When the header
/// is longer than that part, the function reads the whole file.
pub fn image_dimensions(path: &Path) -> Option<(f32, f32)> {
    const HEADER_CAP: u64 = 512 * 1024;

    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(HEADER_CAP)
        .read_to_end(&mut bytes)
        .ok()?;

    let from_header = image::ImageReader::new(std::io::Cursor::new(&bytes))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok();

    let (w, h) = match from_header {
        Some(size) => size,
        None => image::image_dimensions(path).ok()?,
    };
    Some((w as f32, h as f32))
}
