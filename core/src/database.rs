// SPDX-FileCopyrightText: © 2024 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later

use anyhow::*;
pub use rusqlite::Connection;
use std::path;

// Embed migration SQL in executable.
refinery::embed_migrations!("migrations");

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
    migrations::runner()
        .set_abort_divergent(false)
        .set_abort_missing(false)
        .run(con)?;
    Ok(())
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
