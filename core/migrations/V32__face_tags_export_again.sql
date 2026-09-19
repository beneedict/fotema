-- SPDX-FileCopyrightText: © 2026 David Bliss
--
-- SPDX-License-Identifier: GPL-3.0-or-later

-- An earlier build wrote the sidecar as "photo.xmp" and without the origin
-- mark on each region. The writer now uses "photo.jpg.xmp" and marks its
-- regions, so it can tell them from the regions of another program. Clear
-- the marker of every picture with a confirmed person, so the next run writes
-- the sidecar of that picture again in the new form.
UPDATE pictures
SET face_tags_exported = 0
WHERE picture_id IN (
    SELECT picture_id FROM pictures_faces WHERE is_confirmed = 1
);
