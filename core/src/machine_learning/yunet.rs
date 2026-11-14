// SPDX-FileCopyrightText: © 2026 Benedikt Nimmervoll
//
// SPDX-License-Identifier: GPL-3.0-or-later

//! YuNet face detector via OpenCV's FaceDetectorYN.
//!
//! YuNet is a small, fast single-stage detector. It is trained on faces of
//! roughly 10x10 to 300x300 pixels. Fotema runs it on the x-large thumbnail,
//! which keeps faces in that range for almost all pictures.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use ndarray::ArrayViewD;
use opencv::core::{CV_8UC3, Mat, Size};
use opencv::objdetect::{FaceDetectorYN, FaceDetectorYNTrait};
use opencv::prelude::*;
use rust_faces::{Face, FaceDetector, Rect, RustFacesError, RustFacesResult};
use tracing::info;

use super::model_download::{ModelFile, ensure_model};

const MODEL_FILE: &str = "face_detection_yunet_2023mar.onnx";
const MODEL_URL: &str = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx";
const MODEL_SHA256: &str = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4";
const MODEL_SIZE: u64 = 232_589;

/// Score threshold. YuNet's own default is 0.9; 0.6 matches the operating
/// point used by the other detectors and the benchmark.
const SCORE_THRESHOLD: f32 = 0.6;
const NMS_THRESHOLD: f32 = 0.3;
const TOP_K: i32 = 5000;

pub struct YuNetDetector {
    // FaceDetectorYN is Send but not Sync. The Mutex makes the detector Sync
    // so it satisfies rust_faces::FaceDetector, and each worker still owns its
    // own detector instance.
    detector: Mutex<opencv::core::Ptr<FaceDetectorYN>>,
}

impl YuNetDetector {
    pub fn new(cache_dir: &Path) -> Result<Self> {
        let model_path = cache_dir.join("opencv_models").join(MODEL_FILE);
        if let Some(parent) = model_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        ensure_model(
            &ModelFile {
                url: MODEL_URL,
                sha256: MODEL_SHA256,
                size: MODEL_SIZE,
                description: "YuNet face detector",
            },
            &model_path,
        )?;

        let detector = FaceDetectorYN::create(
            &model_path.to_string_lossy(),
            "",
            Size::new(640, 640),
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            TOP_K,
            0,
            0,
        )
        .map_err(|e| anyhow!("Failed to create YuNet face detector: {e}"))?;

        info!("Loaded YuNet face detector from {:?}", model_path);
        Ok(Self {
            detector: Mutex::new(detector),
        })
    }

    fn detect_mat(&self, image: &Mat) -> Result<Vec<Face>, opencv::Error> {
        let size = image.size()?;
        let mut detector = self.detector.lock().unwrap();
        detector.set_input_size(Size::new(size.width, size.height))?;

        let mut faces = Mat::default();
        detector.detect(image, &mut faces)?;

        let rows = faces.rows();
        let cols = faces.cols();
        if rows <= 0 || cols < 15 {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(rows as usize);
        for row in 0..rows {
            let x = *faces.at_2d::<f32>(row, 0)?;
            let y = *faces.at_2d::<f32>(row, 1)?;
            let width = *faces.at_2d::<f32>(row, 2)?;
            let height = *faces.at_2d::<f32>(row, 3)?;
            let confidence = *faces.at_2d::<f32>(row, 14)?;

            let mut landmarks = Vec::with_capacity(5);
            for i in 0..5 {
                let lx = *faces.at_2d::<f32>(row, 4 + i * 2)?;
                let ly = *faces.at_2d::<f32>(row, 5 + i * 2)?;
                landmarks.push((lx, ly));
            }

            out.push(Face {
                rect: Rect {
                    x,
                    y,
                    width,
                    height,
                },
                confidence,
                landmarks: Some(landmarks),
            });
        }

        Ok(out)
    }
}

impl FaceDetector for YuNetDetector {
    fn detect(&mut self, image: ArrayViewD<u8>) -> RustFacesResult<Vec<Face>> {
        let shape = image.shape().to_vec();
        if shape.len() != 3 || shape[2] != 3 {
            return Err(RustFacesError::Other(format!(
                "YuNet expects an HxWx3 RGB image, got shape {shape:?}"
            )));
        }
        let (height, width) = (shape[0], shape[1]);
        let rgb = image
            .as_slice()
            .ok_or_else(|| RustFacesError::Other("YuNet input is not contiguous".into()))?;

        // OpenCV expects BGR.
        let mut bgr = Vec::with_capacity(width * height * 3);
        for px in rgb.chunks_exact(3) {
            bgr.push(px[2]);
            bgr.push(px[1]);
            bgr.push(px[0]);
        }

        let mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                height as i32,
                width as i32,
                CV_8UC3,
                bgr.as_mut_ptr() as *mut _,
                width * 3,
            )
        }
        .map_err(|e| RustFacesError::Other(format!("Failed to wrap image for YuNet: {e}")))?;

        self.detect_mat(&mat)
            .map_err(|e| RustFacesError::Other(format!("YuNet inference failed: {e}")))
    }
}
