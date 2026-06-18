# SPDX-FileCopyrightText: © 2024 David Bliss
#
# SPDX-License-Identifier: GPL-3.0-or-later

fs = import('fs')

global_conf = configuration_data()
global_conf.set_quoted('APP_ID', application_id)
global_conf.set_quoted('PKGDATADIR', pkgdatadir)
global_conf.set_quoted('PROFILE', profile)
global_conf.set_quoted('VERSION', version + version_suffix)
global_conf.set_quoted('GETTEXT_PACKAGE', gettext_package)
global_conf.set_quoted('LOCALEDIR', localedir)
global_conf.set_quoted('I18NDIR', i18ndir)

config = configure_file(
  input: 'config.rs.in',
  output: 'config.rs',
  configuration: global_conf
)
# Copy the config.rs output to the source directory.
run_command(
  'cp',
  meson.project_build_root() / 'src' / 'config.rs',
  meson.project_source_root() / 'src' / 'config.rs',
  check: true
)

cargo_options = [ '--manifest-path', meson.project_source_root() / 'Cargo.toml' ]

# Persistent cargo target dir / CARGO_HOME for faster incremental Flatpak
# rebuilds: the fork's build script sets FOTEMA_CARGO_TARGET_DIR / FOTEMA_CARGO_HOME
# to a cache mounted OUTSIDE the (wiped) build dir, so unchanged crates are not
# recompiled or re-downloaded. Upstream/Flathub builds leave them unset and
# behave exactly as before.
cargo_target_dir = run_command('sh', '-c', 'printf %s "${FOTEMA_CARGO_TARGET_DIR:-}"', check: false).stdout().strip()
if cargo_target_dir == ''
  cargo_target_dir = meson.project_build_root() / 'src'
endif
cargo_options += [ '--target-dir', cargo_target_dir ]

cargo_home = run_command('sh', '-c', 'printf %s "${FOTEMA_CARGO_HOME:-}"', check: false).stdout().strip()
if cargo_home != ''
  cargo_env = [ 'CARGO_HOME=' + cargo_home ]
elif get_option('sandboxed')
  # This is the path used by flatpak-cargo-generator in flatpak-builder-tools
  # See https://github.com/flathub/app.fotema.Fotema
  cargo_env = [ 'CARGO_HOME=' + meson.project_source_root() / 'cargo' ]
else
  cargo_env = [ 'CARGO_HOME=' + meson.project_build_root() / 'cargo-home' ]
endif

if get_option('profile') == 'default'
  cargo_options += [ '--release' ]
  rust_target = 'release'
  message('Building in release mode')
else
  rust_target = 'debug'
  message('Building in debug mode')
endif

cargo_build = custom_target(
  'cargo-build',
  build_by_default: true,
  build_always_stale: true,
  output: meson.project_name(),
  console: true,
  install: true,
  install_dir: bindir,
  depends: resources,
  command: [
    'env',
    cargo_env,
    cargo, 'build',
    cargo_options,
    '&&',
    'cp', cargo_target_dir / rust_target / meson.project_name(), '@OUTPUT@',
  ]
)
