#!/bin/bash
# Build the vendored hdf5_ecf codec plugin (plugin/) into a prefix directory.
# Usage: build_scripts/build_plugin.sh [install_prefix]   (default: ./hdf5_plugin)
# Requires: cmake, a C++ compiler, libhdf5 dev headers
#   (Debian: apt-get install build-essential cmake libhdf5-dev)
#   (macOS:  brew install cmake hdf5)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${1:-${ROOT_DIR}/hdf5_plugin}"
BUILD_DIR="${ROOT_DIR}/plugin/build"

cmake -S "${ROOT_DIR}/plugin" -B "${BUILD_DIR}" \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF

cmake --build "${BUILD_DIR}" -j "$(getconf _NPROCESSORS_ONLN)"
cmake --install "${BUILD_DIR}"

# Flatten: the plugin loader wants the shared libs directly under the prefix
if [ -d "${PREFIX}/lib" ]; then
  cp -a "${PREFIX}/lib/hdf5/plugin/"* "${PREFIX}/" 2>/dev/null || true
  cp -a "${PREFIX}/lib/libhdf5_ecf_codec"* "${PREFIX}/" 2>/dev/null || true
  rm -rf "${PREFIX}/lib"
fi
rm -rf "${PREFIX}/include" "${PREFIX}/share" "${BUILD_DIR}"

echo "ECF plugin installed to: ${PREFIX}"
echo "Set: export HDF5_PLUGIN_PATH=\"${PREFIX}\""
ls -la "${PREFIX}"
