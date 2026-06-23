#!/bin/bash
# Build the release Flatpak (app.fotema.Fotema), keeping the upstream Fotema
# version and appending an incrementing build suffix (-01, -02, ...) so each
# build is uniquely identifiable. The suffix is applied only for the build; the
# tracked files are restored afterwards, so the committed version stays at the
# upstream value.
#
# Everything stays inside the project folder (build dir, repo, bundle, state).
#
# Usage:  build-aux/flatpak-build.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

MANIFEST="build-aux/app.fotema.Fotema.json"
METAINFO="data/app.fotema.Fotema.metainfo.xml.in.in"
APPID="app.fotema.Fotema"
STATE="$ROOT/.flatpak"
COUNTER="$ROOT/build-aux/.build-number"
BUNDLE="$ROOT/fotema.flatpak"

# Persistent cargo cache OUTSIDE the (wiped) build dir, so the ~400 Rust crates
# are not re-downloaded/recompiled every build. Injected into the fotema module
# below; honoured by src/meson.build via FOTEMA_CARGO_TARGET_DIR / _HOME.
CACHE="$STATE/cargo-cache"
mkdir -p "$CACHE/target" "$CACHE/home"

mkdir -p "$STATE/backup"

# Keep the tracked files at their upstream value: back up, restore on exit.
cp meson.build "$STATE/backup/meson.build"
cp "$METAINFO" "$STATE/backup/metainfo.xml"
cp "$MANIFEST" "$STATE/backup/manifest.json"
restore_versions() {
    cp "$STATE/backup/meson.build" meson.build
    cp "$STATE/backup/metainfo.xml" "$METAINFO"
    cp "$STATE/backup/manifest.json" "$MANIFEST"
}
trap restore_versions EXIT

# onnxruntime variant: cpu | webgpu | webgpu-source.
#   cpu           = stock prebuilt (Flathub default, no GPU)
#   webgpu        = prebuilt GPU artifact (fast download; needs the release uploaded)
#   webgpu-source = build the GPU runtime from source (~1h first time, then cached)
# Default to the from-source GPU build so local installs keep GPU acceleration even
# before the prebuilt artifact is published. The tracked manifest stays CPU-only
# (restored on exit), so upstream/Flathub builds are unaffected.
ORT_VARIANT="${FOTEMA_ORT:-webgpu-source}"
echo ">> onnxruntime variant: $ORT_VARIANT (override with FOTEMA_ORT=cpu|webgpu|webgpu-source)"

# Inject the persistent cargo cache into the fotema module (mount + env) and select
# the onnxruntime module variant. The tracked manifest stays generic; restored on
# exit by the trap above.
python3 - "$MANIFEST" "$CACHE" "$ORT_VARIANT" <<'PY'
import json, sys
path, cache, ort_variant = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(path))

# Swap the onnxruntime module include for the requested variant.
ort_module = {
    "cpu": "modules/libonnxruntime.json",
    "webgpu": "modules/libonnxruntime-webgpu.json",
    "webgpu-source": "modules/libonnxruntime-webgpu-source.json",
}.get(ort_variant, "modules/libonnxruntime.json")
m["modules"] = [
    ort_module if (isinstance(x, str) and x.endswith("libonnxruntime.json")) else x
    for x in m.get("modules", [])
]

for mod in m.get("modules", []):
    if isinstance(mod, dict) and mod.get("name") == "fotema":
        # Skip the ~8-minute test suite for local/personal builds.
        mod.pop("run-tests", None)
        bo = mod.setdefault("build-options", {})
        args = bo.setdefault("build-args", [])
        fs = "--filesystem=" + cache
        if fs not in args:
            args.append(fs)
        env = bo.setdefault("env", {})
        env["FOTEMA_CARGO_TARGET_DIR"] = cache + "/target"
        env["FOTEMA_CARGO_HOME"] = cache + "/home"
json.dump(m, open(path, "w"), indent=4)
PY

# Build number: -01 for the first build, then -02, ...
n=$(( $(cat "$COUNTER" 2>/dev/null || echo 0) + 1 ))
suffix=$(printf '%02d' "$n")

# Canonical version = the latest Fotema git release tag (e.g. v2.4.2). This is
# the authoritative current version; fall back to the meson project version if
# tags are unavailable. Used as the single base everywhere so the in-app
# VERSION, the Flatpak version and the update check all agree.
base=$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//')
[ -n "$base" ] || base=$(grep -m1 -oP "version: '\K[0-9.]+" meson.build)
echo ">> Build $base-$suffix (Fotema git version $base + build suffix -$suffix)"

# In-app VERSION = project_version + version_suffix. Sync the project version to
# the git base (meson.build's own value may lag) and set the suffix.
sed -i "0,/version: '[0-9.]*'/s//version: '$base'/" meson.build
sed -i "s/version_suffix = ''/version_suffix = '-$suffix'/" meson.build
# Flatpak/AppStream version: base + suffix on the top metainfo <release>
# (overrides the metainfo's own release number for the build).
sed -i "0,/<release version=\"[0-9.]*\"/s//<release version=\"$base-$suffix\"/" "$METAINFO"

REPO="$STATE/repo"
BUILDDIR="$STATE/build"
rm -rf "$BUILDDIR" "$REPO"

flatpak install -y --user --noninteractive flathub org.flatpak.Builder >/dev/null 2>&1 || true
flatpak run --filesystem="$ROOT" --share=network org.flatpak.Builder \
    --user --install-deps-from=flathub --force-clean --disable-rofiles-fuse \
    --state-dir="$STATE/builder" \
    --repo="$REPO" "$BUILDDIR" "$MANIFEST"
flatpak build-bundle "$REPO" "$BUNDLE" "$APPID" master

echo "$n" >"$COUNTER"
echo ">> Bundle: $BUNDLE  (build -$suffix)"
ls -lh "$BUNDLE"
