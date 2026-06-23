// SPDX-FileCopyrightText: © 2025 David Bliss
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// CLIP-based "smart search" (à la Immich): a multilingual CLIP model embeds every
// photo into a vector, and a free-text query is embedded into the same space by a
// text encoder. Ranking is plain cosine similarity, exactly like the ArcFace face
// embeddings in `face_recognizer.rs`.
//
// The model is a split ONNX export from the `immich-app` HuggingFace org
// (`nllb-clip-base-siglip__v1`): a SigLIP image tower and an NLLB multilingual
// text tower. Both are downloaded on first use into the cache dir, mirroring how
// ArcFace is fetched. Inference uses `ort` (onnxruntime); the dylib is bundled in
// the Flatpak and located via `ORT_DYLIB_PATH` (set up in `src/main.rs`), so we
// build `ort` with `load-dynamic` and never link a second copy.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

use ort::execution_providers::{CPU, CUDA, CoreML, WebGPU};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use reqwest::header::{ACCEPT, HeaderMap, HeaderValue};
use tokenizers::Tokenizer;
use tracing::info;

/// Model identifier stored alongside each embedding in the DB. If the model ever
/// changes, bumping this string invalidates old embeddings (they get recomputed),
/// the same trick the ArcFace migration uses with the embedding byte length.
pub const MODEL_NAME: &str = "nllb-clip-base-siglip-v1";

const HF_BASE: &str =
    "https://huggingface.co/immich-app/nllb-clip-base-siglip__v1/resolve/main";

const VISUAL_URL: &str = "https://huggingface.co/immich-app/nllb-clip-base-siglip__v1/resolve/main/visual/model.onnx";
const VISUAL_FILE: &str = "clip_visual.onnx";

const TEXTUAL_URL: &str = "https://huggingface.co/immich-app/nllb-clip-base-siglip__v1/resolve/main/textual/model.onnx";
const TEXTUAL_FILE: &str = "clip_textual.onnx";

const TOKENIZER_URL: &str = "https://huggingface.co/immich-app/nllb-clip-base-siglip__v1/resolve/main/textual/tokenizer.json";
const TOKENIZER_FILE: &str = "clip_tokenizer.json";

/// SigLIP image preprocessing constants. The image tower of this model expects
/// 384x384 RGB (verified from the ONNX input shape `image [1,3,384,384]`), scaled
/// to [0,1] then normalised to [-1,1] with mean/std 0.5. Hardcoded so we don't
/// need a JSON parser dependency. If the default model is ever changed, update
/// these together with the URLs above (and re-check the input shape).
const IMAGE_SIZE: u32 = 384;
const IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Text tower context length and pad token. The ONNX text input is `text [1,77]`
/// of int32; the NLLB tokenizer emits `[eng_Latn, …tokens…, </s>]` with no
/// padding, so we pad/truncate to 77 with `<pad>` (id 1).
const TEXT_CONTEXT_LEN: usize = 77;
const TEXT_PAD_ID: i32 = 1;

fn models_dir(cache_dir: &Path) -> Result<PathBuf> {
    let base = cache_dir.join("clip_models");
    std::fs::create_dir_all(&base)?;
    Ok(base)
}

/// Ensure the image encoder is present, returning its path. Used by the
/// background embedding task (it does not need the text tower).
pub fn ensure_visual_model(cache_dir: &Path) -> Result<PathBuf> {
    let _ = HF_BASE; // documents the source repo
    let base = models_dir(cache_dir)?;
    let visual = base.join(VISUAL_FILE);
    download_model(VISUAL_URL, &visual, "CLIP image encoder (~0.4GB)")?;
    Ok(visual)
}

/// Ensure the text encoder and its tokenizer are present, returning their paths
/// (textual model, tokenizer). Used by the runtime search worker.
pub fn ensure_text_model(cache_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let base = models_dir(cache_dir)?;
    let textual = base.join(TEXTUAL_FILE);
    download_model(TEXTUAL_URL, &textual, "CLIP text encoder")?;
    let tokenizer = base.join(TOKENIZER_FILE);
    download_model(TOKENIZER_URL, &tokenizer, "CLIP tokenizer")?;
    Ok((textual, tokenizer))
}

/// Per-thread image embedder. The `ort::Session` is not cheap to create (loads the
/// ~0.4GB image tower), so build one per worker thread and reuse it across photos,
/// exactly like `FaceEmbedder`.
pub struct ClipImageEmbedder {
    session: Session,
}

impl ClipImageEmbedder {
    pub fn new(visual_path: &Path) -> Result<Self> {
        // Image embedding is the heavy, batch workload → use the GPU EP chain.
        let session = build_session(visual_path, true)?;
        Ok(Self { session })
    }

    /// Compute the L2-normalised CLIP embedding for an image file (a thumbnail).
    pub fn embedding(&mut self, image_path: &Path) -> Result<Vec<f32>> {
        let img = image::open(image_path)?.to_rgb8();
        let resized = image::imageops::resize(
            &img,
            IMAGE_SIZE,
            IMAGE_SIZE,
            image::imageops::FilterType::CatmullRom,
        );

        // Build a planar CHW float tensor: data[c][y][x] = (pixel/255 - mean)/std.
        let side = IMAGE_SIZE as usize;
        let mut data = vec![0f32; 3 * side * side];
        for (i, px) in resized.pixels().enumerate() {
            let x = i % side;
            let y = i / side;
            for c in 0..3 {
                let v = px.0[c] as f32 / 255.0;
                data[c * side * side + y * side + x] = (v - IMAGE_MEAN[c]) / IMAGE_STD[c];
            }
        }

        let shape = vec![1_i64, 3, IMAGE_SIZE as i64, IMAGE_SIZE as i64];
        // ort::Error is not Send+Sync, so it can't convert into anyhow::Error via
        // `?`; stringify it instead.
        let tensor =
            Tensor::from_array((shape, data)).map_err(|e| anyhow!("CLIP image tensor: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .map_err(|e| anyhow!("CLIP image inference: {e}"))?;
        let (_, raw) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("CLIP image output: {e}"))?;
        let mut v = raw.to_vec();
        normalize(&mut v);
        Ok(v)
    }
}

/// Text embedder for search queries. Loads the NLLB text tower and its tokenizer
/// once and keeps them in memory (the runtime search worker holds a single
/// instance so each keystroke-triggered query is fast).
pub struct ClipTextEmbedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl ClipTextEmbedder {
    pub fn new(textual_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("Loading tokenizer: {e}"))?;
        // Text embedding is a single fast inference per query → run it on the CPU.
        // This also avoids a SECOND concurrent WebGPU/Dawn session racing the
        // background image-embedding task (which crashed Dawn with "Command
        // encoding already finished"). Only the image task uses the GPU.
        let session = build_session(textual_path, false)?;
        Ok(Self { session, tokenizer })
    }

    /// Compute the L2-normalised CLIP embedding for a search query. The text model
    /// has a single fixed-size int32 input `text [1,77]`.
    pub fn embedding(&mut self, query: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(query, true)
            .map_err(|e| anyhow!("Tokenizing query: {e}"))?;

        // Truncate to the context length and pad with <pad> (id 1) to exactly 77.
        let mut ids: Vec<i32> = encoding
            .get_ids()
            .iter()
            .take(TEXT_CONTEXT_LEN)
            .map(|&x| x as i32)
            .collect();
        ids.resize(TEXT_CONTEXT_LEN, TEXT_PAD_ID);

        // ort::Error is not Send+Sync, so stringify rather than convert via `?`.
        let tensor = Tensor::from_array((vec![1_i64, TEXT_CONTEXT_LEN as i64], ids))
            .map_err(|e| anyhow!("CLIP text ids: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .map_err(|e| anyhow!("CLIP text inference: {e}"))?;

        let (_, raw) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("CLIP text output: {e}"))?;
        let mut v = raw.to_vec();
        normalize(&mut v);
        Ok(v)
    }
}

/// Build an `ort` session for an ONNX file. The execution provider (CUDA/CoreML/
/// CPU) is whatever the bundled onnxruntime offers — the same fp32 model runs on
/// GPU or CPU, so there is one universal file regardless of hardware.
fn build_session(model_path: &Path, gpu: bool) -> Result<Session> {
    // Vendor-neutral acceleration: when `gpu`, register GPU execution providers in
    // preference order, then CPU. ort registers them in order and an EP that isn't
    // available (feature off, not compiled into the loaded onnxruntime, or no
    // matching hardware/driver) is skipped with a warning — CPU is always the final
    // fallback, so this never regresses. To actually use a GPU the bundled
    // onnxruntime must include the matching EP:
    //   - WebGPU (Dawn→Vulkan/Metal/D3D): cross-vendor (Intel/AMD/NVIDIA/ARM),
    //     needs a WebGPU-enabled onnxruntime build.
    //   - CUDA: NVIDIA; CoreML: Apple. (OpenVINO for Intel can be added when an
    //     OpenVINO-enabled runtime is bundled.)
    //
    // `gpu == false` forces CPU. Used for the text encoder so it never spins up a
    // second concurrent WebGPU/Dawn session alongside the background image task
    // (concurrent Dawn devices corrupt command encoding / the heap).
    //
    // ort::Error is not Send+Sync, so it can't convert into anyhow::Error via `?`.
    let eps = if gpu {
        vec![
            CUDA::default().build(),
            WebGPU::default().build(),
            CoreML::default().build(),
            CPU::default().build(),
        ]
    } else {
        vec![CPU::default().build()]
    };
    let session = Session::builder()
        .map_err(|e| anyhow!("CLIP session builder: {e}"))?
        .with_execution_providers(eps)
        .map_err(|e| anyhow!("CLIP execution providers: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("CLIP optimization level: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("CLIP load model {:?}: {e}", model_path))?;
    Ok(session)
}

/// L2-normalise a vector in place (so cosine similarity is a plain dot product).
fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity of two equal-length, L2-normalised vectors (= dot product).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return -1.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Download a model file if it isn't already present. Mirrors the helper in
/// `face_recognizer.rs` (atomic via a temp file + rename).
fn download_model(url: &str, destination: &Path, description: &str) -> Result<()> {
    if destination.exists() {
        return Ok(());
    }

    info!("Downloading CLIP model ({}) from {}", description, url);

    let headers = {
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("*/*"));
        headers
    };

    let client = reqwest::blocking::Client::new();
    let mut response = client.get(url).headers(headers).send()?;

    if response.status().is_success() {
        let tmp_path = destination.with_extension("tmp");
        let tmp_file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(tmp_file);
        while let Ok(bytes_read) = response.copy_to(&mut writer) {
            if bytes_read == 0 {
                break;
            }
        }
        info!("CLIP model ({}) downloaded.", description);
        std::fs::rename(tmp_path, destination)?;
        Ok(())
    } else {
        Err(anyhow!(
            "Failed to download CLIP model ({}): {}",
            description,
            response.status()
        ))
    }
}
