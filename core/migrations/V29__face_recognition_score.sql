-- Cosine similarity of an auto-recognised face to its best matching reference,
-- stored when a face is auto-assigned to a person (is_confirmed = FALSE). Used to
-- group unconfirmed suggestions into confidence tiers for review. NULL for faces
-- assigned before this column existed, or from sources without a similarity score
-- (e.g. imported XMP name tags).
ALTER TABLE pictures_faces ADD COLUMN recognition_score REAL;
