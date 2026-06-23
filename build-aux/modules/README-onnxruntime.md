<!--
SPDX-FileCopyrightText: © 2025 David Bliss
SPDX-License-Identifier: GFDL-1.3-or-later
-->

# onnxruntime variants (CPU vs GPU/WebGPU)

Fotema runs its ONNX models (CLIP smart-search; face detection/recognition use it
too) through `ort`, which loads `libonnxruntime.so` dynamically at runtime
(`src/main.rs::setup_onnxruntime` → `/app/lib/libonnxruntime.so`). Three
interchangeable module files provide that library — all install the same
`name: "libonnxruntime"`, so exactly **one** is referenced by the manifest:

| Module | What it ships | Flathub-safe | Notes |
|---|---|---|---|
| `libonnxruntime.json` | **CPU** stock prebuilt (default) | ✅ | Reference this in the committed manifest. |
| `libonnxruntime-webgpu.json` | **GPU** prebuilt (WebGPU/Dawn→Vulkan/Metal/D3D) | ✅ | Downloads a release artifact; fast; one-line opt-in. |
| `libonnxruntime-webgpu-source.json` | GPU built **from source** | ❌ (build-time network, ~1h) | Only to *regenerate* the prebuilt artifact. |

The GPU build is **vendor-neutral**: the same library accelerates on Intel / AMD /
NVIDIA / ARM GPUs via Vulkan (the GNOME runtime ships Mesa Vulkan; `--device=dri`
is already granted). NPUs are not covered. Fotema's EP selection
(`core/src/machine_learning/clip.rs::build_session`) registers WebGPU then falls
back to CPU, so the **same Fotema binary** works with either library.

## Enable GPU (the easy way)

Point the manifest at the prebuilt GPU module — one line in
`build-aux/app.fotema.Fotema.json` (and `.Devel.json`):

```diff
-        "modules/libonnxruntime.json",
+        "modules/libonnxruntime-webgpu.json",
```

…or, with the local build script, just:

```sh
FOTEMA_ORT=webgpu        build-aux/flatpak-build.sh   # prebuilt GPU artifact
FOTEMA_ORT=webgpu-source build-aux/flatpak-build.sh   # build GPU from source
FOTEMA_ORT=cpu           build-aux/flatpak-build.sh   # CPU (default committed)
```

(`flatpak-build.sh` defaults to `webgpu-source` so local installs keep GPU; it
restores the committed CPU manifest on exit.)

## Regenerate / host the prebuilt GPU artifact

When bumping the onnxruntime version, rebuild the artifact and re-host it, then
update the URL + `sha256` in `libonnxruntime-webgpu.json`:

```sh
# 1. Build the GPU library from source (produces /app/lib/libonnxruntime.so*)
FOTEMA_ORT=webgpu-source build-aux/flatpak-build.sh

# 2. Package it as <top>/lib/libonnxruntime.so* and publish as a release asset, e.g.
gh release create onnxruntime-webgpu-1.24.4 onnxruntime-webgpu-linux-x64-1.24.4.tgz

# 3. Put the asset URL + sha256 into libonnxruntime-webgpu.json
```

The artifact is built inside the GNOME SDK, so it is ABI-compatible with the
runtime. WebGPU needs Node.js+npm at build time (bundled as a source), build-time
network for Dawn/FetchContent, and `CMAKE_POLICY_VERSION_MINIMUM=3.5` for cmake 4
— all encoded in `libonnxruntime-webgpu-source.json`.
