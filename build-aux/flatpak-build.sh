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

mkdir -p "$STATE/backup"

# Keep the tracked version files at their upstream value: back up, restore on exit.
cp meson.build "$STATE/backup/meson.build"
cp "$METAINFO" "$STATE/backup/metainfo.xml"
restore_versions() {
    cp "$STATE/backup/meson.build" meson.build
    cp "$STATE/backup/metainfo.xml" "$METAINFO"
}
trap restore_versions EXIT

# Build number: -01 for the first build, then -02, ...
n=$(( $(cat "$COUNTER" 2>/dev/null || echo 0) + 1 ))
suffix=$(printf '%02d' "$n")

# Canonical upstream version = the version of the latest metainfo <release>
# (this is what `flatpak info` shows). Use it as the single base for everything
# so the in-app VERSION, the Flatpak version and the update check all agree.
base=$(grep -m1 -oP '<release version="\K[0-9.]+' "$METAINFO")
echo ">> Build $base-$suffix (upstream version $base + build suffix -$suffix)"

# In-app VERSION = project_version + version_suffix. Sync the project version to
# the canonical base (meson.build's own version may lag) and set the suffix.
sed -i "0,/version: '[0-9.]*'/s//version: '$base'/" meson.build
sed -i "s/version_suffix = ''/version_suffix = '-$suffix'/" meson.build
# Flatpak/AppStream version: base + suffix on the top metainfo <release>.
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
