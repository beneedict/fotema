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
/// is above the embedded series. Rewrite the bookkeeping to the embedded names
/// and checksums for the schema that the fork database already has. The schema
/// checks decide whether a migration counts as applied, so this cannot skip a
/// migration whose schema is missing. Extra fork-only columns and tables stay
/// in place; they are harmless.
fn reconcile_legacy_fork_history(con: &mut Connection, runner: &refinery::Runner) -> Result<()> {
    if !table_exists(con, "refinery_schema_history")? {
        return Ok(());
    }

    let applied: HashMap<i32, String> = {
        let mut stmt = con.prepare("SELECT version, name FROM refinery_schema_history")?;
        let rows = stmt.query_map([], |row| {
            std::result::Result::Ok((row.get::<_, i32>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.flatten().collect()
    };

    let embedded: HashMap<i32, &refinery::Migration> = runner
        .get_migrations()
        .iter()
        .map(|migration| (migration.version(), migration))
        .collect();

    // Nothing to do when the recorded names already match this build.
    let divergent = applied.iter().any(|(version, name)| {
        embedded
            .get(version)
            .is_some_and(|migration| migration.name() != name)
    });
    if !divergent {
        return Ok(());
    }

    // Only rewrite the known fork series. Any other divergence is left to the
    // tolerant runner, which reports it in the log and keeps the library usable.
    let is_legacy_fork = LEGACY_FORK_MIGRATIONS
        .iter()
        .all(|(version, name)| applied.get(version).is_some_and(|got| got == name));
    if !is_legacy_fork {
        return Ok(());
    }

    let mut to_mark: Vec<(i32, String, String)> = Vec::new();
    for migration in embedded.values() {
        if migration.version() < 24 || !schema_present(con, migration.name())? {
            continue;
        }
        to_mark.push((
            migration.version(),
            migration.name().to_string(),
            migration.checksum().to_string(),
        ));
    }

    let tx = con.transaction()?;
    tx.execute(
        "DELETE FROM refinery_schema_history WHERE version >= 24",
        [],
    )?;
    let applied_on = chrono::Utc::now().to_rfc3339();
    {
        let mut insert = tx.prepare(
            "INSERT INTO refinery_schema_history (version, name, applied_on, checksum)
             VALUES (?1, ?2, ?3, ?4)",
        )?;
        for (version, name, checksum) in &to_mark {
            insert.execute(params![version, name, applied_on, checksum])?;
        }
    }
    tx.commit()?;

    tracing::info!(
        "Reconciled {} fork migration record(s) with the embedded migration series",
        to_mark.len()
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
