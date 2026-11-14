-- SPDX-FileCopyrightText: © 2026 Benedikt Nimmervoll
--
-- SPDX-License-Identifier: GPL-3.0-or-later

-- The detector that produced a picture's face scan. Used to decide whether a
-- picture must be scanned again after the user switches face detectors.
-- Existing scans on this branch were produced by BlazeFace.
ALTER TABLE pictures_face_scans ADD COLUMN detector TEXT NOT NULL DEFAULT 'blazeface';
