-- SPDX-FileCopyrightText: © 2026 David Bliss
--
-- SPDX-License-Identifier: GPL-3.0-or-later

-- Marks a picture whose confirmed person names Fotema has written to the XMP
-- sidecar of that picture. A file sync tool carries the sidecar to another
-- computer, where the import assigns the same people again.
ALTER TABLE pictures ADD COLUMN face_tags_exported INTEGER NOT NULL DEFAULT 0;

-- A picture without a confirmed person needs no sidecar. Mark it as done, so the
-- first run writes only the pictures that hold a person. Without this the first
-- run would open the sidecar of every picture in the library for no result.
--
-- A later change to a face clears the marker of its picture again. The export
-- then removes the regions when the user has taken the person away.
UPDATE pictures
SET face_tags_exported = 1
WHERE picture_id NOT IN (
    SELECT picture_id FROM pictures_faces WHERE is_confirmed = 1
);
