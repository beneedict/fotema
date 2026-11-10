-- Allow a whole person to be hidden without deleting their data. Reversible:
-- clearing the flag restores the person and all face assignments.
ALTER TABLE people ADD COLUMN is_ignored BOOLEAN NOT NULL DEFAULT 0;
