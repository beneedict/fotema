// SPDX-FileCopyrightText: © 2024 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

use anyhow::*;
pub use rusqlite::Connection;
use rusqlite::params;
use std::collections::HashMap;
use std::path;

// Embed migration SQL in executable.
refinery::embed_migrations!("migrations");

/// Migrations that the fork applied under version numbers that upstream later
/// uses for different migrations. The pair is (version, fork name).
const FORK_MARKER_TABLE: &str = "fotema_fork_schema";

const LEGACY_FORK_MIGRATIONS: &[(i32, &str)] = &[
    (25, "face_tags_imported"),
    (26, "pictures_face_tags"),
    (27, "face_not_person"),
    (28, "people_is_ignored"),
];

pub fn setup(database_path: &path::Path) -> Result<Connection> {
    let mut con = Connection::open(database_path)?;
    run_migrations(&mut con)?;
    Ok(con)
}

// for testing
pub fn setup_in_memory() -> Result<Connection> {
    let mut con = Connection::open_in_memory()?;
    run_migrations(&mut con)?;
    Ok(con)
}

/// Apply pending migrations. Backward compatibility is a hard requirement, so
/// the runner is tolerant: it must never refuse to start on a database written
/// by an older build. It still applies any new (pending) migrations, but does
/// not abort on bookkeeping discrepancies between the database's recorded
/// migrations and the embedded set (which would otherwise crash an upgrade).
fn run_migrations(con: &mut Connection) -> Result<()> {
    let runner = migrations::runner();
    reconcile_legacy_fork_history(con, &runner)?;
    runner
        .set_abort_divergent(false)
        .set_abort_missing(false)
        .run(con)?;
    Ok(())
}

/// An existing fork database recorded the fork's own migrations under versions
/// 24 and above. When the embedded set uses those numbers for different
/// migrations, the tolerant runner would treat them as already applied and
/// would never run any later migration, because the database's highest version
/// is above the embedded series.
///
/// Rewrite the bookkeeping so that it follows the embedded migrations. A
/// migration counts as applied when its schema is already present, so no
/// migration whose schema is missing is skipped. Records of migrations that
/// this build does not know (the fork-only ones) are removed; their schema
/// stays in place and is harmless.
///
/// A small marker table remembers that this database came from the fork, so a
/// partial upgrade through an intermediate set of migrations is also handled
/// when a later build adds more migrations.
fn reconcile_legacy_fork_history(con: &mut Connection, runner: &refinery::Runner) -> Result<()> {
    if !table_exists(con, "refinery_schema_history")? {
        return Ok(());
    }

    let applied: HashMap<i32, (String, String)> = {
        let mut stmt =
            con.prepare("SELECT version, name, applied_on FROM refinery_schema_history")?;
        let rows = stmt.query_map([], |row| {
            std::result::Result::Ok((
                row.get::<_, i32>(0)?,
                (row.get::<_, String>(1)?, row.get::<_, String>(2)?),
            ))
        })?;
        rows.flatten().collect()
    };

    let is_legacy_fork = LEGACY_FORK_MIGRATIONS
        .iter()
        .all(|(version, name)| applied.get(version).is_some_and(|(got, _)| got == name))
        || table_exists(con, FORK_MARKER_TABLE)?
        || table_exists(con, "clip_embeddings")?;
    if !is_legacy_fork {
        return Ok(());
    }

    let embedded: HashMap<i32, &refinery::Migration> = runner
        .get_migrations()
        .iter()
        .map(|migration| (migration.version(), migration))
        .collect();

    let now = chrono::Utc::now().to_rfc3339();
    let mut to_keep: Vec<(i32, String, String, String)> = Vec::new();
    for migration in embedded.values().filter(|m| m.version() >= 24) {
        let existing = applied.get(&migration.version());
        // Keep an already applied migration under the embedded checksum. This
        // covers migrations whose schema check this function does not know.
        let applied_on = match existing {
            Some((name, applied_on)) if name == migration.name() => applied_on.clone(),
            _ if schema_present(con, migration.name())? => now.clone(),
            // Not applied yet: the runner must run it.
            _ => continue,
        };
        to_keep.push((
            migration.version(),
            migration.name().to_string(),
            applied_on,
            migration.checksum().to_string(),
        ));
    }

    let tx = con.transaction()?;
    tx.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS {FORK_MARKER_TABLE} (id INTEGER PRIMARY KEY CHECK (id = 1));
         INSERT OR IGNORE INTO {FORK_MARKER_TABLE} (id) VALUES (1);"
    ))?;
    tx.execute(
        "DELETE FROM refinery_schema_history WHERE version >= 24",
        [],
    )?;
    {
        let mut insert = tx.prepare(
            "INSERT INTO refinery_schema_history (version, name, applied_on, checksum)
             VALUES (?1, ?2, ?3, ?4)",
        )?;
        for (version, name, applied_on, checksum) in &to_keep {
            insert.execute(params![version, name, applied_on, checksum])?;
        }
    }
    tx.commit()?;

    tracing::info!(
        "Reconciled fork migration history with the embedded migration series ({} record(s))",
        to_keep.len()
    );
    Ok(())
}

/// Whether the schema change of a migration is already present. Only the
/// migrations of the known fork series have a check; any other migration is
/// left for the runner to apply.
fn schema_present(con: &Connection, migration_name: &str) -> Result<bool> {
    match migration_name {
        "face_embeddings" => column_exists(con, "pictures_faces", "embedding"),
        "people_is_ignored" => column_exists(con, "people", "is_ignored"),
        "face_tags_imported" => column_exists(con, "pictures", "face_tags_imported"),
        "pictures_face_tags" => table_exists(con, "pictures_face_tags"),
        "face_tags_exported" => column_exists(con, "pictures", "face_tags_exported"),
        "face_not_person" => table_exists(con, "face_not_person"),
        _ => Ok(false),
    }
}

fn table_exists(con: &Connection, table: &str) -> Result<bool> {
    let exists: bool = con.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1
         )",
        params![table],
        |row| row.get(0),
    )?;
    Ok(exists)
}

fn column_exists(con: &Connection, table: &str, column: &str) -> Result<bool> {
    let mut stmt = con.prepare(&format!("PRAGMA table_info({table})"))?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let name: String = row.get(1)?;
        if name == column {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Opening the same database twice must not apply anything a second time.
    /// This is the property that lets an existing library survive an upgrade.
    #[test]
    fn setup_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pictures.sqlite");

        let first = setup(&path).unwrap();
        let first_count: i64 = first
            .query_row("SELECT COUNT(*) FROM refinery_schema_history", [], |row| {
                row.get(0)
            })
            .unwrap();
        drop(first);

        let second = setup(&path).unwrap();
        let second_count: i64 = second
            .query_row("SELECT COUNT(*) FROM refinery_schema_history", [], |row| {
                row.get(0)
            })
            .unwrap();

        assert_eq!(first_count, second_count);
        assert!(first_count >= 32);
    }
}
