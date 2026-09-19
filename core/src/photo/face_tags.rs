// SPDX-FileCopyrightText: © 2025 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Reads person-name face-region tags embedded in photo XMP metadata so existing
// names written by other software (digiKam, Picasa, Windows Photo Gallery, Apple
// Photos, ...) can be imported. Two common schemas are supported:
//
//  - MWG Regions (`mwg-rs:Regions` / `mwg-rs:Name` + `mwg-rs:Area` whose `x` is
//    the region centre, normalized 0..1).
//  - Microsoft People Tags (`MPReg:PersonDisplayName` + `MPReg:Rectangle`
//    "x, y, w, h", top-left, normalized 0..1).
//
// The parser does the best that it can: it gives no tags for input that it
// cannot understand. It always extracts the horizontal centre, which is
// sufficient to sort the regions from left to right and to match them to the
// detected faces. It also extracts the full area when the file supplies one.
//
// This module also WRITES a sidecar file. Fotema writes the confirmed person
// names of a photo to "<photo>.jpg.xmp" beside the photo, as digiKam and
// darktable do it. A file sync tool then carries the names to another computer,
// where the import assigns them again. Fotema does not change the photo itself.
//
// Each region that Fotema writes carries the attribute `fotema:Origin`. The
// writer uses it to tell its own regions from the regions of another program.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use quick_xml::events::Event;
use quick_xml::reader::Reader;

use anyhow::Result;

/// The area of a face region. All values are normalized to 0..1. The centre
/// values give the middle of the area, as the MWG schema specifies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TagArea {
    pub center_x: f32,
    pub center_y: f32,
    pub width: f32,
    pub height: f32,
}

impl TagArea {
    /// Convert the area into (left, top, width, height). Use this form to
    /// compare an area with the bounds of a detected face.
    pub fn to_rect(self) -> (f32, f32, f32, f32) {
        (
            self.center_x - self.width / 2.0,
            self.center_y - self.height / 2.0,
            self.width,
            self.height,
        )
    }

    /// Build a normalized area from the pixel bounds of a face and the pixel
    /// size of the image that the face detector looked at. The function gives
    /// None when the image size is not positive.
    pub fn from_pixel_bounds(
        left: f32,
        top: f32,
        width: f32,
        height: f32,
        image_width: f32,
        image_height: f32,
    ) -> Option<TagArea> {
        if image_width <= 0.0 || image_height <= 0.0 {
            return None;
        }
        Some(TagArea {
            center_x: (left + width / 2.0) / image_width,
            center_y: (top + height / 2.0) / image_height,
            width: width / image_width,
            height: height / image_height,
        })
    }

    /// Cut the area down to the picture (0..1 on both axes). A face detector can
    /// give a box that reaches over the edge of the picture. The function gives
    /// None when nothing of the area lies inside the picture.
    pub fn clamped(self) -> Option<TagArea> {
        let (left, top, width, height) = self.to_rect();
        let left2 = left.max(0.0);
        let top2 = top.max(0.0);
        let right = (left + width).min(1.0);
        let bottom = (top + height).min(1.0);
        if right <= left2 || bottom <= top2 {
            return None;
        }
        Some(TagArea {
            center_x: (left2 + right) / 2.0,
            center_y: (top2 + bottom) / 2.0,
            width: right - left2,
            height: bottom - top2,
        })
    }

    /// The intersection over union of two areas. The value is 0.0 when the areas
    /// do not touch, and 1.0 when they are equal.
    pub fn iou(self, other: TagArea) -> f32 {
        let (ax, ay, aw, ah) = self.to_rect();
        let (bx, by, bw, bh) = other.to_rect();

        let left = ax.max(bx);
        let top = ay.max(by);
        let right = (ax + aw).min(bx + bw);
        let bottom = (ay + ah).min(by + bh);

        if right <= left || bottom <= top {
            return 0.0;
        }

        let intersection = (right - left) * (bottom - top);
        let union = aw * ah + bw * bh - intersection;
        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }
}

/// A named face region read from a photo's XMP.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceTag {
    /// Person name.
    pub name: String,
    /// Horizontal centre of the region, normalized to 0..1.
    pub center_x: f32,
    /// The full area of the region, if the file supplies one. A sidecar that
    /// Fotema wrote always supplies it. Other software often supplies only the
    /// centre.
    pub area: Option<TagArea>,
}

/// The path of the XMP sidecar file that Fotema writes. The name keeps the
/// extension of the photo, as digiKam and darktable do it: "photo.jpg" gives
/// "photo.jpg.xmp". Thus "photo.jpg" and "photo.heic" do not share one file,
/// and Fotema does not write into the sidecar of a raw file "photo.nef".
pub fn sidecar_path(photo_path: &Path) -> PathBuf {
    let mut name = photo_path.as_os_str().to_os_string();
    name.push(".xmp");
    PathBuf::from(name)
}

/// The path of a sidecar in the form of Adobe: "photo.jpg" gives "photo.xmp".
/// Fotema reads this form, but does not write it.
pub fn legacy_sidecar_path(photo_path: &Path) -> PathBuf {
    photo_path.with_extension("xmp")
}

/// Read named face-region tags for a photo. The function reads the sidecar files
/// first, because a photo manager keeps the sidecar current and the photo itself
/// unchanged. It reads the photo only when no sidecar gives tags.
pub fn read_face_tags(path: &Path) -> Vec<FaceTag> {
    for sidecar in [sidecar_path(path), legacy_sidecar_path(path)] {
        if sidecar == path {
            continue;
        }
        let tags = read_tags_from_file(&sidecar);
        if !tags.is_empty() {
            return tags;
        }
    }
    read_tags_from_file(path)
}

/// Read the first megabyte of a file and parse the XMP packet in it.
fn read_tags_from_file(path: &Path) -> Vec<FaceTag> {
    let file = match std::fs::File::open(path) {
        std::result::Result::Ok(f) => f,
        std::result::Result::Err(_) => return Vec::new(),
    };
    let mut buf = Vec::new();
    if file.take(1024 * 1024).read_to_end(&mut buf).is_err() {
        return Vec::new();
    }
    face_tags_from_bytes(&buf)
}

/// Parse named face-region tags from already-read header bytes (the part of the
/// file that holds the XMP packet). Lets EXIF and XMP be extracted from a single
/// file read.
pub fn face_tags_from_bytes(bytes: &[u8]) -> Vec<FaceTag> {
    match extract_xmp(bytes) {
        Some(xmp) => parse_face_tags(&xmp),
        None => Vec::new(),
    }
}

/// A region as the parser found it, with the facts that the writer needs to
/// merge it with the regions of this computer.
#[derive(Debug, Clone)]
struct ParsedRegion {
    tag: FaceTag,
    /// True when Fotema wrote the region. Such a region carries the attribute
    /// `fotema:Origin`.
    is_fotema: bool,
    /// True for a MWG region, false for a Microsoft People Tag region.
    is_mwg: bool,
}

/// The attribute that marks a region that Fotema wrote.
const ORIGIN_ATTRIBUTE: &str = "fotema:Origin";
const ORIGIN_VALUE: &str = "Fotema";

/// Pull the `<x:xmpmeta>...</x:xmpmeta>` packet out of the given bytes.
fn extract_xmp(bytes: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(bytes);
    let start = text.find("<x:xmpmeta")?;
    let rest = &text[start..];
    let end = rest.find("</x:xmpmeta>")? + "</x:xmpmeta>".len();
    Some(rest[..end].to_string())
}

#[derive(Clone, Copy)]
enum Field {
    Name,
    Type,
    Rect,
}

/// Parse the XMP XML for MWG / Microsoft face regions.
fn parse_face_tags(xmp: &str) -> Vec<FaceTag> {
    parse_regions(xmp).into_iter().map(|r| r.tag).collect()
}

/// Parse the XMP XML for MWG / Microsoft face regions, and keep the origin of
/// each region. digiKam writes each face in both schemas. The function then
/// keeps one region for the face, and prefers the MWG region, because Fotema
/// writes that schema.
fn parse_regions(xmp: &str) -> Vec<ParsedRegion> {
    let mut reader = Reader::from_str(xmp);
    let mut out: Vec<ParsedRegion> = Vec::new();

    let mut cur_name: Option<String> = None;
    let mut cur_area = PartialArea::default();
    let mut is_face = true;
    let mut capture: Option<Field> = None;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                match e.local_name().as_ref() {
                    b"li" | b"Description" => {
                        cur_name = None;
                        cur_area = PartialArea::default();
                        is_face = true;
                        // The compact form keeps the name and the area in the
                        // attributes of the li element.
                        scan_region_attrs(e.attributes(), &mut cur_name, &mut cur_area);
                    }
                    b"Area" => scan_region_attrs(e.attributes(), &mut cur_name, &mut cur_area),
                    b"Name" | b"PersonDisplayName" => capture = Some(Field::Name),
                    b"Type" => capture = Some(Field::Type),
                    b"Rectangle" => capture = Some(Field::Rect),
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => match e.local_name().as_ref() {
                b"li" | b"Description" => {
                    cur_name = None;
                    cur_area = PartialArea::default();
                    is_face = true;
                    scan_region_attrs(e.attributes(), &mut cur_name, &mut cur_area);
                    flush(&mut out, &mut cur_name, &mut cur_area, &mut is_face);
                }
                b"Area" => scan_region_attrs(e.attributes(), &mut cur_name, &mut cur_area),
                _ => {}
            },
            Ok(Event::Text(e)) => {
                if let Some(field) = capture.take() {
                    let txt = e.unescape().unwrap_or_default().trim().to_string();
                    match field {
                        Field::Name => {
                            if !txt.is_empty() {
                                cur_name = Some(txt);
                            }
                        }
                        Field::Type => {
                            if !txt.eq_ignore_ascii_case("Face") {
                                is_face = false;
                            }
                        }
                        Field::Rect => ms_rectangle_area(&txt, &mut cur_area),
                    }
                }
            }
            Ok(Event::End(e)) => {
                capture = None;
                if matches!(e.local_name().as_ref(), b"li" | b"Description") {
                    flush(&mut out, &mut cur_name, &mut cur_area, &mut is_face);
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    out
}

/// The area values of the region that the parser reads at present. A file can
/// supply the centre only, so each value is separate until the region ends.
#[derive(Default)]
struct PartialArea {
    center_x: Option<f32>,
    center_y: Option<f32>,
    width: Option<f32>,
    height: Option<f32>,
    /// True when the region has the attribute `fotema:Origin`.
    is_fotema: bool,
    /// True when the values came from a Microsoft Rectangle element.
    is_microsoft: bool,
}

impl PartialArea {
    /// Return the full area when the file supplied every value.
    fn complete(&self) -> Option<TagArea> {
        Some(TagArea {
            center_x: self.center_x?,
            center_y: self.center_y?,
            width: self.width?,
            height: self.height?,
        })
    }
}

/// Push the current region if it is a complete face region, and reset state.
///
/// digiKam writes the same face as a MWG region and as a Microsoft region. The
/// function keeps one region for a face: the MWG region replaces a Microsoft
/// region at the same place, and a Microsoft region does not add a face that a
/// MWG region already gives.
fn flush(
    out: &mut Vec<ParsedRegion>,
    cur_name: &mut Option<String>,
    cur_area: &mut PartialArea,
    is_face: &mut bool,
) {
    if *is_face && let (Some(name), Some(center_x)) = (cur_name.take(), cur_area.center_x) {
        let region = ParsedRegion {
            tag: FaceTag {
                name,
                center_x,
                area: cur_area.complete(),
            },
            is_fotema: cur_area.is_fotema,
            is_mwg: !cur_area.is_microsoft,
        };
        match out.iter().position(|r| same_place(r, &region)) {
            Some(i) if region.is_mwg && !out[i].is_mwg => out[i] = region,
            Some(_) => {}
            None => out.push(region),
        }
    }
    *cur_name = None;
    *cur_area = PartialArea::default();
    *is_face = true;
}

/// Examine if two regions describe the same face. With two full areas the
/// function compares the overlap. With a centre only, it compares the centre.
fn same_place(a: &ParsedRegion, b: &ParsedRegion) -> bool {
    match (a.tag.area, b.tag.area) {
        (Some(x), Some(y)) => x.iou(y) >= SAME_FACE_OVERLAP,
        _ => (a.tag.center_x - b.tag.center_x).abs() < 0.01,
    }
}

/// Two areas that overlap this much describe the same face.
const SAME_FACE_OVERLAP: f32 = 0.3;

/// Scan the attributes of an element for a region name and an area. This covers
/// the compact form and the MWG form.
fn scan_region_attrs(
    attrs: quick_xml::events::attributes::Attributes,
    cur_name: &mut Option<String>,
    cur_area: &mut PartialArea,
) {
    for attr in attrs.flatten() {
        let value = || {
            String::from_utf8_lossy(&attr.value)
                .trim()
                .parse::<f32>()
                .ok()
        };
        match attr.key.local_name().as_ref() {
            b"Name" | b"PersonDisplayName" => {
                // The writer escapes "&" and the quotes, so the reader must
                // undo that. Otherwise "Müller &amp; Sohn" comes back changed.
                let v = attr
                    .unescape_value()
                    .map(|v| v.trim().to_string())
                    .unwrap_or_default();
                if !v.is_empty() {
                    *cur_name = Some(v);
                }
            }
            b"Origin" if attr.key.as_ref().starts_with(b"fotema:") => {
                cur_area.is_fotema = true;
            }
            // The MWG attributes stArea:x and stArea:y give the centre of the
            // region. The attributes stArea:w and stArea:h give its size.
            b"x" if cur_area.center_x.is_none() => cur_area.center_x = value(),
            b"y" if cur_area.center_y.is_none() => cur_area.center_y = value(),
            b"w" if cur_area.width.is_none() => cur_area.width = value(),
            b"h" if cur_area.height.is_none() => cur_area.height = value(),
            _ => {}
        }
    }
}

/// The Microsoft Rectangle element holds "x, y, w, h". The first two values give
/// the top left corner. All values are normalized.
fn ms_rectangle_area(txt: &str, cur_area: &mut PartialArea) {
    let parts: Vec<f32> = txt
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    if parts.len() == 4 {
        cur_area.center_x = Some(parts[0] + parts[2] / 2.0);
        cur_area.center_y = Some(parts[1] + parts[3] / 2.0);
        cur_area.width = Some(parts[2]);
        cur_area.height = Some(parts[3]);
        cur_area.is_microsoft = true;
    }
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------
//
// Fotema writes the confirmed person names of a photo into the XMP sidecar of
// that photo. A file sync tool then carries the names to another computer.
//
// A sidecar of another program holds much more than face regions: tags,
// categories, ratings and edit history. Fotema must not destroy that data.
// Therefore the writer does NOT serialise the XML again. It replaces only the
// region blocks in the byte stream and keeps every other byte unchanged.

/// The XML namespaces that a region block needs. The writer adds a declaration
/// to the rdf:Description element only when the file does not already have it.
const REQUIRED_NS: [(&str, &str); 3] = [
    (
        "xmlns:mwg-rs",
        "http://www.metadataworkinggroup.com/schemas/regions/",
    ),
    ("xmlns:stArea", "http://ns.adobe.com/xmp/sType/Area#"),
    ("xmlns:fotema", "https://fotema.app/ns/xmp/1.0/"),
];

/// The extension of the copy that the writer makes before it changes a sidecar
/// of another program for the first time.
const BACKUP_SUFFIX: &str = ".fotema-bak";

/// An empty sidecar, used when a photo has no sidecar yet.
const EMPTY_SIDECAR: &str = concat!(
    "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n",
    "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\" x:xmptk=\"Fotema\">\n",
    " <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n",
    "  <rdf:Description rdf:about=\"\">\n",
    "  </rdf:Description>\n",
    " </rdf:RDF>\n",
    "</x:xmpmeta>\n",
    "<?xpacket end=\"w\"?>\n",
);

/// What `write_face_tags` did to the file on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    /// The sidecar changed.
    Written,
    /// The sidecar already held the given regions, so the file stayed as it was.
    Unchanged,
}

/// Write the confirmed face tags of a photo into its XMP sidecar.
///
/// `named` gives the confirmed faces with a name. `detected` gives the areas of
/// ALL faces that this computer found in the photo, with or without a name. Each
/// area must be a full area; the function ignores a tag without one.
///
/// The function merges the new regions with the regions in the file:
///
///  - A region that Fotema wrote before is replaced when a detected or a named
///    face covers it. Thus a name that the user took away leaves the file. A
///    Fotema region that no face of this computer covers stays, because another
///    computer found that face and this computer did not.
///  - A region of another program stays, unless a named face covers it. Fotema
///    then writes its own name for that face.
///  - A region without a full area cannot be written again, so it leaves.
///
/// The function keeps all other content of the sidecar. It writes the file
/// through a temporary file, so an interruption cannot leave a part of a file.
/// Before it changes a sidecar for the first time, it makes a copy with the
/// suffix ".fotema-bak". An empty `named` list with no region to keep removes
/// the region block, but the file stays.
pub fn write_face_tags(
    photo_path: &Path,
    named: &[FaceTag],
    detected: &[TagArea],
) -> Result<WriteOutcome> {
    let path = sidecar_path(photo_path);

    let on_disk = read_sidecar(&path)?;
    let had_file = on_disk.is_some();
    let existing = on_disk.or_else(|| adopt_legacy_sidecar(photo_path));
    let source = existing.unwrap_or_else(|| EMPTY_SIDECAR.to_string());

    let regions = merge_regions(parse_regions(&source), named, detected);
    if !had_file && regions.is_empty() {
        // There is no sidecar and nothing to write, so do not make one.
        return Ok(WriteOutcome::Unchanged);
    }

    let updated = replace_regions(&source, &regions);
    if had_file && updated == source {
        return Ok(WriteOutcome::Unchanged);
    }

    let mode = if had_file {
        make_backup(&path)?;
        std::fs::metadata(&path).ok().map(|m| m.permissions())
    } else {
        None
    };

    write_atomically(&path, &updated, mode)?;
    Ok(WriteOutcome::Written)
}

/// Read a sidecar. A missing file gives None. Any other problem, for example a
/// file that is not UTF-8 or that the user cannot read, is an error. The writer
/// must not replace such a file with a new one.
fn read_sidecar(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow::anyhow!("Cannot read sidecar {:?}: {}", path, e)),
    }
}

/// An older build of Fotema wrote "photo.xmp". Take that content over into the
/// new file, but only when Fotema made the old file. The sidecar of another
/// program can belong to a different photo with the same stem.
fn adopt_legacy_sidecar(photo_path: &Path) -> Option<String> {
    let legacy = legacy_sidecar_path(photo_path);
    if legacy == photo_path {
        return None;
    }
    let text = std::fs::read_to_string(legacy).ok()?;
    text.contains("x:xmptk=\"Fotema\"").then_some(text)
}

/// A region that the writer puts into the file.
#[derive(Debug, Clone)]
struct Region {
    name: String,
    area: TagArea,
    /// True when the region carries the Fotema origin mark.
    is_fotema: bool,
}

/// Merge the regions in the file with the faces of this computer. Refer to
/// `write_face_tags` for the rules.
fn merge_regions(
    existing: Vec<ParsedRegion>,
    named: &[FaceTag],
    detected: &[TagArea],
) -> Vec<Region> {
    let named: Vec<Region> = named
        .iter()
        .filter_map(|t| {
            Some(Region {
                name: t.name.clone(),
                area: t.area?,
                is_fotema: true,
            })
        })
        .collect();

    let covers = |area: TagArea| move |other: &TagArea| other.iou(area) >= SAME_FACE_OVERLAP;

    let mut out: Vec<Region> = Vec::new();
    for region in existing {
        let Some(area) = region.tag.area else {
            continue;
        };
        let by_named = named.iter().map(|n| &n.area).any(covers(area));
        let covered = if region.is_fotema {
            by_named || detected.iter().any(covers(area))
        } else {
            by_named
        };
        if !covered {
            out.push(Region {
                name: region.tag.name,
                area,
                is_fotema: region.is_fotema,
            });
        }
    }
    out.extend(named);
    out
}

/// Copy a sidecar once, before Fotema changes it for the first time. The
/// function does nothing when a copy already exists.
fn make_backup(path: &Path) -> Result<()> {
    let mut backup = path.as_os_str().to_os_string();
    backup.push(BACKUP_SUFFIX);
    let backup = PathBuf::from(backup);

    if !backup.exists() {
        std::fs::copy(path, &backup)?;
    }
    Ok(())
}

/// Write `content` to `path` through a temporary file in the same directory.
/// The move into position is one operation, so a reader sees either the old file
/// or the new file, and never a part of one.
///
/// `mode` gives the permissions of the file that the new file replaces. The
/// temporary file starts with permissions for the owner only, so without this
/// step a sidecar that other accounts could read would become private.
fn write_atomically(path: &Path, content: &str, mode: Option<std::fs::Permissions>) -> Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Sidecar path has no parent directory"))?;

    let mut temp = tempfile::Builder::new()
        .prefix(".fotema-xmp-")
        .suffix(".tmp")
        .tempfile_in(dir)?;

    temp.write_all(content.as_bytes())?;
    temp.flush()?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = mode.unwrap_or_else(|| std::fs::Permissions::from_mode(0o644));
        temp.as_file().set_permissions(perms)?;
    }
    #[cfg(not(unix))]
    let _ = mode;

    temp.persist(path)?;
    Ok(())
}

/// Replace the region blocks of an XMP document with the given regions.
///
/// The function removes each existing `mwg-rs:Regions` and `MPRI:Regions` block.
/// It then puts one new `mwg-rs:Regions` block before the end of the first
/// `rdf:Description` element. An empty list of regions removes the blocks and
/// adds nothing.
///
/// The function changes no other byte of the document.
fn replace_regions(source: &str, regions: &[Region]) -> String {
    let mut out = source.to_string();

    // Take out the region blocks of every program. Fotema writes one block, so
    // two blocks with different names cannot disagree with each other. The
    // merge has already taken the regions of those blocks over.
    for name in ["mwg-rs:Regions", "MPRI:Regions"] {
        while let Some(range) = find_element(&out, name) {
            let range = expand_to_whole_lines(&out, range);
            out.replace_range(range, "");
        }
    }

    if regions.is_empty() {
        return out;
    }

    let Some(desc) = find_description(&out) else {
        return out;
    };

    out = ensure_namespaces(out, desc.tag_end);

    // The namespaces can have made the start tag longer, so find it again.
    let Some(desc) = find_description(&out) else {
        return out;
    };

    let block = render_regions(regions);

    if desc.self_closing {
        // "<rdf:Description ... />" holds all its data in attributes and has no
        // end tag. Open the element, put the block in it, and close it.
        let close = format!(">\n{}{}</rdf:Description>", block, desc.indent);
        out.replace_range(desc.tag_end - 2..desc.tag_end, &close);
    } else {
        // Put the block before the end tag of THIS element. A struct inside the
        // element can hold its own rdf:Description, so the first end tag in the
        // document is not always the right one.
        let Some(end) = find_description_end(&out, desc.tag_end) else {
            return out;
        };
        let line_start = out[..end].rfind('\n').map_or(0, |i| i + 1);
        if out[line_start..end].trim().is_empty() {
            // The end tag has a line of its own. Put the block on its own lines
            // before it, so a second write gives the same file as the first.
            out.insert_str(line_start, &block);
        } else {
            // The end tag shares its line with other content. Start a new line.
            let insert = format!("\n{}{}", block, desc.indent);
            out.insert_str(end, &insert);
        }
    }

    out
}

/// Where the first rdf:Description element starts and ends.
struct Description {
    /// The position just after the ">" that ends the start tag.
    tag_end: usize,
    /// True when the element has no end tag, as in "<rdf:Description ... />".
    self_closing: bool,
    /// The whitespace at the start of the line that holds the element.
    indent: String,
}

/// Find the first rdf:Description element of the document.
fn find_description(xml: &str) -> Option<Description> {
    let start = xml.find("<rdf:Description")?;
    let rel = xml[start..].find('>')?;
    let tag_end = start + rel + 1;
    let self_closing = xml[start..tag_end].trim_end_matches('>').ends_with('/');

    let line_start = xml[..start].rfind('\n').map_or(0, |i| i + 1);
    let indent = xml[line_start..start]
        .chars()
        .take_while(|c| c.is_whitespace())
        .collect();

    Some(Description {
        tag_end,
        self_closing,
        indent,
    })
}

/// Find the position of the end tag that closes the rdf:Description element
/// whose start tag ends at `tag_end`. The function counts the nested
/// rdf:Description elements, so it does not stop at the end tag of a struct
/// inside the element.
fn find_description_end(xml: &str, tag_end: usize) -> Option<usize> {
    const OPEN: &str = "<rdf:Description";
    const CLOSE: &str = "</rdf:Description>";

    let mut pos = tag_end;
    let mut depth = 0usize;
    loop {
        let rest = &xml[pos..];
        let next_open = rest.find(OPEN);
        let next_close = rest.find(CLOSE)?;

        match next_open {
            Some(open) if open < next_close => {
                let open_end = pos + open + rest[open..].find('>')? + 1;
                if !xml[pos + open..open_end]
                    .trim_end_matches('>')
                    .ends_with('/')
                {
                    depth += 1;
                }
                pos = open_end;
            }
            _ => {
                if depth == 0 {
                    return Some(pos + next_close);
                }
                depth -= 1;
                pos += next_close + CLOSE.len();
            }
        }
    }
}

/// Grow a range so that it covers whole lines. The range takes in the
/// whitespace before the element and the line break after it. Without this an
/// empty, indented line stays behind, and each write makes the file longer.
fn expand_to_whole_lines(xml: &str, range: std::ops::Range<usize>) -> std::ops::Range<usize> {
    let line_start = xml[..range.start].rfind('\n').map_or(0, |i| i + 1);
    let start = if xml[line_start..range.start].trim().is_empty() {
        line_start
    } else {
        range.start
    };

    let mut end = range.end;
    if xml[end..].starts_with('\n') {
        end += 1;
    }

    start..end
}

/// Find the byte range of an element, from its start tag to the end of its end
/// tag. The function also finds an element that closes itself.
fn find_element(xml: &str, name: &str) -> Option<std::ops::Range<usize>> {
    let open = format!("<{}", name);
    let start = xml.find(&open)?;

    // The character after the name must end the name, so "mwg-rs:RegionList"
    // does not match a search for "mwg-rs:Regions".
    let after = xml[start + open.len()..].chars().next()?;
    if !after.is_whitespace() && after != '>' && after != '/' {
        return None;
    }

    let tag_end = xml[start..].find('>')? + start + 1;
    if xml[start..tag_end].trim_end_matches('>').ends_with('/') {
        return Some(start..tag_end);
    }

    let close = format!("</{}>", name);
    let end = xml[tag_end..].find(&close)? + tag_end + close.len();
    Some(start..end)
}

/// Add each missing namespace declaration to the rdf:Description element that
/// holds the position `before`. Without the declaration the new region block
/// would make the document invalid.
fn ensure_namespaces(mut xml: String, before: usize) -> String {
    let Some(start) = xml[..before].rfind("<rdf:Description") else {
        return xml;
    };
    let Some(rel) = xml[start..].find('>') else {
        return xml;
    };
    let tag_end = start + rel;

    let mut additions = String::new();
    for (prefix, uri) in REQUIRED_NS {
        if !xml[start..tag_end].contains(prefix) {
            additions.push_str(&format!("\n    {}=\"{}\"", prefix, uri));
        }
    }

    if !additions.is_empty() {
        // Put the declarations before the ">" that ends the start tag, and keep
        // a "/" of an element that closes itself.
        let insert_at = if xml[..tag_end].ends_with('/') {
            tag_end - 1
        } else {
            tag_end
        };
        xml.insert_str(insert_at, &additions);
    }

    xml
}

/// Build an MWG region block. The layout follows what digiKam writes, so other
/// programs read it without difficulty. A region of Fotema carries the origin
/// mark; a region of another program that the merge kept does not.
fn render_regions(regions: &[Region]) -> String {
    let mut out = String::from("   <mwg-rs:Regions rdf:parseType=\"Resource\">\n");
    out.push_str("    <mwg-rs:RegionList>\n     <rdf:Bag>\n");

    for region in regions {
        let area = region.area;
        out.push_str("      <rdf:li>\n       <rdf:Description\n");
        out.push_str(&format!(
            "        mwg-rs:Name=\"{}\"\n",
            quick_xml::escape::escape(region.name.as_str())
        ));
        if region.is_fotema {
            out.push_str(&format!(
                "        {}=\"{}\"\n",
                ORIGIN_ATTRIBUTE, ORIGIN_VALUE
            ));
        }
        out.push_str("        mwg-rs:Type=\"Face\">\n");
        out.push_str("       <mwg-rs:Area\n");
        out.push_str(&format!("        stArea:x=\"{:.6}\"\n", area.center_x));
        out.push_str(&format!("        stArea:y=\"{:.6}\"\n", area.center_y));
        out.push_str(&format!("        stArea:w=\"{:.6}\"\n", area.width));
        out.push_str(&format!("        stArea:h=\"{:.6}\"\n", area.height));
        out.push_str("        stArea:unit=\"normalized\"/>\n");
        out.push_str("       </rdf:Description>\n      </rdf:li>\n");
    }

    out.push_str("     </rdf:Bag>\n    </mwg-rs:RegionList>\n   </mwg-rs:Regions>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(cx: f32, cy: f32, w: f32, h: f32) -> TagArea {
        TagArea {
            center_x: cx,
            center_y: cy,
            width: w,
            height: h,
        }
    }

    fn tag(name: &str, cx: f32) -> FaceTag {
        FaceTag {
            name: name.into(),
            center_x: cx,
            area: Some(area(cx, 0.5, 0.1, 0.2)),
        }
    }

    /// Merge and write, as `write_face_tags` does it, but on a string.
    fn write(source: &str, named: &[FaceTag], detected: &[TagArea]) -> String {
        let regions = merge_regions(parse_regions(source), named, detected);
        replace_regions(source, &regions)
    }

    /// A sidecar of digiKam, cut down but with the parts that matter.
    const DIGIKAM: &str = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:digiKam="http://www.digikam.org/ns/1.0/"
    xmlns:MPRI="http://ns.microsoft.com/photo/1.2/t/RegionInfo#"
    xmlns:MPReg="http://ns.microsoft.com/photo/1.2/t/Region#"
    xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
    xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#"
    xmp:Rating="4">
   <digiKam:TagsList>
    <rdf:Seq>
     <rdf:li>st/playground</rdf:li>
     <rdf:li>Personen/Lukas Nimm</rdf:li>
    </rdf:Seq>
   </digiKam:TagsList>
    <MPRI:Regions>
     <rdf:Bag>
      <rdf:li
       MPReg:PersonDisplayName="Lukas Nimm"
       MPReg:Rectangle="0.51473, 0.419757, 0.0544794, 0.10086"/>
     </rdf:Bag>
    </MPRI:Regions>
   <mwg-rs:Regions rdf:parseType="Resource">
    <mwg-rs:RegionList>
     <rdf:Bag>
      <rdf:li>
       <rdf:Description
        mwg-rs:Name="Lukas Nimm"
        mwg-rs:Type="Face">
       <mwg-rs:Area
        stArea:x="0.541969"
        stArea:y="0.470187"
        stArea:w="0.0544794"
        stArea:h="0.10086"
        stArea:unit="normalized"/>
       </rdf:Description>
      </rdf:li>
     </rdf:Bag>
    </mwg-rs:RegionList>
   </mwg-rs:Regions>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;

    /// The area of the face "Lukas Nimm" in the digiKam sidecar.
    fn lukas() -> TagArea {
        area(0.541969, 0.470187, 0.0544794, 0.10086)
    }

    #[test]
    fn keeps_foreign_data_when_it_replaces_regions() {
        let out = write(DIGIKAM, &[tag("Anna", 0.25)], &[]);

        // The data of the other program must survive.
        assert!(out.contains("<rdf:li>st/playground</rdf:li>"));
        assert!(out.contains("<rdf:li>Personen/Lukas Nimm</rdf:li>"));
        assert!(out.contains("xmp:Rating=\"4\""));
        assert!(out.contains("digiKam:TagsList"));
        assert!(out.contains("<?xpacket end=\"w\"?>"));

        // The Microsoft block is gone, but the face of the other program stays
        // as a MWG region without the origin mark. The new region is there and
        // carries the mark.
        assert!(!out.contains("MPRI:Regions"));
        assert!(out.contains("mwg-rs:Name=\"Lukas Nimm\""));
        assert!(out.contains("mwg-rs:Name=\"Anna\""));
        assert_eq!(out.matches("fotema:Origin=\"Fotema\"").count(), 1);
        assert!(out.contains("xmlns:fotema="));

        // Exactly one region block.
        assert_eq!(out.matches("<mwg-rs:Regions").count(), 1);
    }

    #[test]
    fn replaces_a_foreign_region_that_a_named_face_covers() {
        let mut renamed = tag("Lukas Nimmervoll", 0.54);
        renamed.area = Some(lukas());
        let out = write(DIGIKAM, &[renamed], &[]);

        assert!(!out.contains("mwg-rs:Name=\"Lukas Nimm\""));
        assert!(out.contains("mwg-rs:Name=\"Lukas Nimmervoll\""));
        assert_eq!(out.matches("<rdf:li>\n").count(), 1);
    }

    #[test]
    fn keeps_a_foreign_region_that_no_named_face_covers() {
        // No name, but a detected face on top of the foreign region: the region
        // belongs to another program, so it stays.
        let out = write(DIGIKAM, &[], &[lukas()]);
        assert!(out.contains("mwg-rs:Name=\"Lukas Nimm\""));
        assert!(!out.contains("fotema:Origin"));
    }

    #[test]
    fn removes_its_own_region_when_the_name_is_gone() {
        let first = write(DIGIKAM, &[tag("Anna", 0.25)], &[area(0.25, 0.5, 0.1, 0.2)]);
        assert!(first.contains("mwg-rs:Name=\"Anna\""));

        // The user took the name away. The face is still detected, so the
        // region of Fotema leaves, and the foreign region stays.
        let second = write(&first, &[], &[area(0.25, 0.5, 0.1, 0.2)]);
        assert!(!second.contains("mwg-rs:Name=\"Anna\""));
        assert!(second.contains("mwg-rs:Name=\"Lukas Nimm\""));
        assert_eq!(second.matches("<mwg-rs:Regions").count(), 1);
    }

    #[test]
    fn keeps_its_own_region_when_this_computer_did_not_detect_the_face() {
        let first = write(DIGIKAM, &[tag("Anna", 0.25)], &[]);
        // Another computer wrote Anna; this computer found no face there.
        let second = write(&first, &[tag("Bob", 0.75)], &[area(0.75, 0.5, 0.1, 0.2)]);
        assert!(second.contains("mwg-rs:Name=\"Anna\""));
        assert!(second.contains("mwg-rs:Name=\"Bob\""));
    }

    #[test]
    fn removes_the_block_when_no_region_is_left() {
        let plain = write(EMPTY_SIDECAR, &[tag("Anna", 0.25)], &[]);
        let gone = write(&plain, &[], &[area(0.25, 0.5, 0.1, 0.2)]);
        assert!(!gone.contains("mwg-rs:Regions"));
        assert!(gone.contains("<?xpacket end=\"w\"?>"));
    }

    #[test]
    fn writes_a_document_that_it_can_read_again() {
        let out = write(EMPTY_SIDECAR, &[tag("Anna", 0.25), tag("Bob", 0.75)], &[]);
        let tags = parse_face_tags(&out);

        assert_eq!(tags.len(), 2);
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"Anna"));
        assert!(names.contains(&"Bob"));

        let anna = tags.iter().find(|t| t.name == "Anna").unwrap();
        let a = anna.area.expect("the writer always writes an area");
        assert!((a.center_x - 0.25).abs() < 1e-4);
        assert!((a.width - 0.1).abs() < 1e-4);
    }

    #[test]
    fn a_name_with_special_characters_comes_back_unchanged() {
        let name = "Müller & Söhne <\"Junior\">";
        let out = write(EMPTY_SIDECAR, &[tag(name, 0.25)], &[]);
        assert!(out.contains("&amp;"));
        let tags = parse_face_tags(&out);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name, name);
    }

    #[test]
    fn a_second_write_gives_the_same_file() {
        let first = write(DIGIKAM, &[tag("Anna", 0.25)], &[]);
        let second = write(&first, &[tag("Anna", 0.25)], &[]);
        assert_eq!(first, second);
    }

    #[test]
    fn adds_a_namespace_when_the_file_has_none() {
        let plain = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmp:Rating="2">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"#;
        let out = write(plain, &[tag("Anna", 0.25)], &[]);

        assert!(out.contains("xmlns:mwg-rs="));
        assert!(out.contains("xmlns:stArea="));
        assert!(out.contains("xmlns:fotema="));
        assert!(out.contains("xmp:Rating=\"2\""));
        assert_eq!(parse_face_tags(&out).len(), 1);
    }

    #[test]
    fn puts_the_block_after_a_nested_description() {
        // Lightroom writes structs as nested rdf:Description elements.
        let lightroom = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
   <crs:Look>
    <rdf:Description crs:Name="Adobe Color">
     <crs:Parameters>
      <rdf:Description crs:Version="15.0"/>
     </crs:Parameters>
    </rdf:Description>
   </crs:Look>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"#;
        let out = write(lightroom, &[tag("Anna", 0.25)], &[]);

        // The block must come after the crs:Look struct, not inside it.
        let look_end = out.find("</crs:Look>").unwrap();
        let block = out.find("<mwg-rs:Regions").unwrap();
        assert!(block > look_end);
        assert_eq!(parse_face_tags(&out).len(), 1);
    }

    #[test]
    fn handles_an_end_tag_that_shares_its_line() {
        let compact = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""><dc:title>x</dc:title></rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"#;
        let out = write(compact, &[tag("Anna", 0.25)], &[]);
        assert!(out.contains("<dc:title>x</dc:title>"));
        assert_eq!(parse_face_tags(&out).len(), 1);
        // The start tag must be whole.
        assert!(out.contains("rdf:about=\"\"\n    xmlns:mwg-rs"));
    }

    #[test]
    fn does_not_confuse_region_list_with_regions() {
        // "mwg-rs:RegionList" must not match a search for "mwg-rs:Regions".
        let xml = "<mwg-rs:RegionList>x</mwg-rs:RegionList>";
        assert!(find_element(xml, "mwg-rs:Regions").is_none());
    }

    #[test]
    fn ignores_a_tag_without_an_area() {
        let no_area = FaceTag {
            name: "Anna".into(),
            center_x: 0.25,
            area: None,
        };
        let out = write(DIGIKAM, &[no_area], &[]);
        assert!(!out.contains("mwg-rs:Name=\"Anna\""));
    }

    #[test]
    fn computes_the_overlap_of_two_areas() {
        let a = area(0.5, 0.5, 0.2, 0.2);
        assert!((a.iou(a) - 1.0).abs() < 1e-6);

        let far = area(0.9, 0.9, 0.05, 0.05);
        assert_eq!(a.iou(far), 0.0);

        let half = area(0.6, 0.5, 0.2, 0.2);
        let overlap = a.iou(half);
        assert!(overlap > 0.0 && overlap < 1.0);
    }

    #[test]
    fn clamps_an_area_to_the_picture() {
        let edge = area(0.0, 0.5, 0.2, 0.2).clamped().unwrap();
        assert!((edge.center_x - 0.05).abs() < 1e-6);
        assert!((edge.width - 0.1).abs() < 1e-6);

        assert!(area(1.5, 0.5, 0.2, 0.2).clamped().is_none());
        assert!(area(0.5, 0.5, 0.0, 0.2).clamped().is_none());
    }

    #[test]
    fn builds_an_area_from_pixel_bounds() {
        let a = TagArea::from_pixel_bounds(100.0, 50.0, 200.0, 100.0, 1000.0, 500.0).unwrap();
        assert!((a.center_x - 0.2).abs() < 1e-6);
        assert!((a.center_y - 0.2).abs() < 1e-6);
        assert!((a.width - 0.2).abs() < 1e-6);
        assert!(TagArea::from_pixel_bounds(0.0, 0.0, 1.0, 1.0, 0.0, 10.0).is_none());
    }

    #[test]
    fn sidecar_keeps_the_extension() {
        assert_eq!(
            sidecar_path(Path::new("/photos/a.jpg")),
            PathBuf::from("/photos/a.jpg.xmp")
        );
        assert_eq!(
            legacy_sidecar_path(Path::new("/photos/a.jpg")),
            PathBuf::from("/photos/a.xmp")
        );
    }

    #[test]
    fn keeps_one_region_for_a_face_that_digikam_wrote_twice() {
        let tags = parse_face_tags(DIGIKAM);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name, "Lukas Nimm");
        // The MWG region wins, so the full area is there.
        assert!(tags[0].area.is_some());
    }

    #[test]
    fn parses_mwg_regions() {
        let xmp = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
          <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
                             xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#">
              <mwg-rs:Regions>
                <mwg-rs:RegionList>
                  <rdf:Bag>
                    <rdf:li>
                      <mwg-rs:Name>Bob</mwg-rs:Name>
                      <mwg-rs:Type>Face</mwg-rs:Type>
                      <mwg-rs:Area stArea:x="0.8" stArea:y="0.4" stArea:w="0.1" stArea:h="0.2"/>
                    </rdf:li>
                    <rdf:li>
                      <mwg-rs:Name>Alice</mwg-rs:Name>
                      <mwg-rs:Type>Face</mwg-rs:Type>
                      <mwg-rs:Area stArea:x="0.2" stArea:y="0.4" stArea:w="0.1" stArea:h="0.2"/>
                    </rdf:li>
                  </rdf:Bag>
                </mwg-rs:RegionList>
              </mwg-rs:Regions>
            </rdf:Description>
          </rdf:RDF>
        </x:xmpmeta>"#;

        let tags = parse_face_tags(xmp);
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].name, "Bob");
        assert!((tags[0].center_x - 0.8).abs() < 1e-6);
        assert_eq!(tags[1].name, "Alice");
        assert!((tags[1].center_x - 0.2).abs() < 1e-6);
    }

    #[test]
    fn ignores_non_face_regions() {
        let xmp = r#"<x:xmpmeta><rdf:li>
            <mwg-rs:Name>Pet</mwg-rs:Name>
            <mwg-rs:Type>Pet</mwg-rs:Type>
            <mwg-rs:Area stArea:x="0.5"/>
          </rdf:li></x:xmpmeta>"#;
        assert!(parse_face_tags(xmp).is_empty());
    }

    #[test]
    fn parses_microsoft_people_tags() {
        let xmp = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
          <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description xmlns:MPReg="http://ns.microsoft.com/photo/1.2/t/Region#">
              <MPRI:Regions xmlns:MPRI="http://ns.microsoft.com/photo/1.2/t/RegionInfo#">
                <rdf:Bag>
                  <rdf:li>
                    <MPReg:Rectangle>0.10, 0.20, 0.20, 0.30</MPReg:Rectangle>
                    <MPReg:PersonDisplayName>Carol</MPReg:PersonDisplayName>
                  </rdf:li>
                </rdf:Bag>
              </MPRI:Regions>
            </rdf:Description>
          </rdf:RDF>
        </x:xmpmeta>"#;

        let tags = parse_face_tags(xmp);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name, "Carol");
        // centre x = 0.10 + 0.20/2 = 0.20
        assert!((tags[0].center_x - 0.20).abs() < 1e-6);
    }

    #[test]
    fn empty_when_no_xmp() {
        assert!(parse_face_tags("<html>no xmp here</html>").is_empty());
    }

    #[test]
    fn does_not_replace_a_sidecar_that_it_cannot_read() {
        let dir = tempfile::tempdir().unwrap();
        let photo = dir.path().join("a.jpg");
        std::fs::write(&photo, b"").unwrap();
        let sidecar = sidecar_path(&photo);
        // Not UTF-8, so the writer cannot understand the file.
        std::fs::write(&sidecar, [0xff, 0xfe, 0x00, 0x41]).unwrap();

        let result = write_face_tags(&photo, &[tag("Anna", 0.25)], &[]);
        assert!(result.is_err());
        assert_eq!(std::fs::read(&sidecar).unwrap(), [0xff, 0xfe, 0x00, 0x41]);
    }

    #[test]
    fn writes_a_new_sidecar_and_reads_it_back() {
        let dir = tempfile::tempdir().unwrap();
        let photo = dir.path().join("a.jpg");
        std::fs::write(&photo, b"").unwrap();

        assert_eq!(
            write_face_tags(&photo, &[], &[]).unwrap(),
            WriteOutcome::Unchanged
        );
        assert!(!sidecar_path(&photo).exists());

        assert_eq!(
            write_face_tags(&photo, &[tag("Anna", 0.25)], &[]).unwrap(),
            WriteOutcome::Written
        );
        assert_eq!(
            write_face_tags(&photo, &[tag("Anna", 0.25)], &[]).unwrap(),
            WriteOutcome::Unchanged
        );
        let tags = read_face_tags(&photo);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name, "Anna");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(sidecar_path(&photo))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o644);
        }
    }

    #[test]
    fn keeps_the_permissions_of_an_existing_sidecar() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let photo = dir.path().join("a.jpg");
            std::fs::write(&photo, b"").unwrap();
            let sidecar = sidecar_path(&photo);
            std::fs::write(&sidecar, DIGIKAM).unwrap();
            std::fs::set_permissions(&sidecar, std::fs::Permissions::from_mode(0o664)).unwrap();

            write_face_tags(&photo, &[tag("Anna", 0.25)], &[]).unwrap();

            let mode = std::fs::metadata(&sidecar).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o664);
            // The copy of the original file is there.
            assert!(dir.path().join("a.jpg.xmp.fotema-bak").exists());
        }
    }

    #[test]
    fn takes_over_a_sidecar_of_an_older_build() {
        let dir = tempfile::tempdir().unwrap();
        let photo = dir.path().join("a.jpg");
        std::fs::write(&photo, b"").unwrap();
        let old = write(EMPTY_SIDECAR, &[tag("Anna", 0.25)], &[]);
        std::fs::write(legacy_sidecar_path(&photo), old).unwrap();

        write_face_tags(&photo, &[tag("Bob", 0.75)], &[]).unwrap();

        let tags = read_face_tags(&photo);
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"Anna"));
        assert!(names.contains(&"Bob"));
    }
}
