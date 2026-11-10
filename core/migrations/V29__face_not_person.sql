-- SPDX-FileCopyrightText: © 2026 Benedikt Nimmervoll
--
-- SPDX-License-Identifier: GPL-3.0-or-later

-- "This face is not this person" corrections. Recognition consults this table
-- so a corrected mistake does not come back with the next run.
CREATE TABLE face_not_person (
    face_id   INTEGER NOT NULL REFERENCES pictures_faces (face_id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES people (person_id) ON DELETE CASCADE,
    PRIMARY KEY (face_id, person_id)
);
