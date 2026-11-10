// SPDX-FileCopyrightText: © 2024 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// ONNX Runtime allows exactly one environment per process. Both the face
// detector (rust-faces) and the CLIP smart search create sessions, and they can
// run at the same time. This module owns the single environment and hands out
// clones of it. Creating a second environment would fail with "only one
// environment is allowed per process".
//
// The environment must be created after `src/main.rs` has pointed
// `ORT_DYLIB_PATH` at the bundled onnxruntime library.

use std::sync::{Mutex, OnceLock};

use anyhow::{Result, anyhow};
use ort::environment::Environment;

static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Return the process-wide ONNX Runtime environment, creating it on first use.
///
/// The mutex serialises the first creation. Without it two background tasks
/// could call `ort::init().build()` at the same time, and the second call would
/// fail because ONNX Runtime permits only one environment.
pub fn shared_environment() -> Result<&'static Environment> {
    if let Some(env) = ENVIRONMENT.get() {
        return Ok(env);
    }

    let _guard = INIT_LOCK
        .lock()
        .map_err(|_| anyhow!("ONNX Runtime environment lock is poisoned"))?;

    if let Some(env) = ENVIRONMENT.get() {
        return Ok(env);
    }

    // ort::Error is not Send + Sync, so stringify it instead of converting it
    // into anyhow::Error with `?`.
    let env = ort::init()
        .with_name("fotema")
        .build()
        .map_err(|e| anyhow!("Failed to create ONNX Runtime environment: {e}"))?;

    let _ = ENVIRONMENT.set(env);
    ENVIRONMENT
        .get()
        .ok_or_else(|| anyhow!("ONNX Runtime environment was not stored"))
}
