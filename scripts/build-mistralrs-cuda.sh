#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-v0.9.2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/.build/mistralrs-cuda"
SRC="$BUILD/src"
BVENV="$BUILD/venv"
RAW_OUT="$BUILD/wheels-raw"
FIXED_DIR="$ROOT/vendor/wheels"
REAL_LIBCUDA="/lib/x86_64-linux-gnu/libcuda.so.1"

mkdir -p "$BUILD" "$FIXED_DIR"

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/EricLBuehler/mistral.rs.git "$SRC"
else
  echo "==> reusing existing clone at $SRC (rm -rf it to re-clone fresh)"
fi

[ -d "$BVENV" ] || uv venv --python 3.12 "$BVENV"
source "$BVENV/bin/activate"
uv pip install "maturin[patchelf]"

cd "$SRC/mistralrs-pyo3"
cargo clean
maturin build --release --features cuda -o "$RAW_OUT"

RAW_WHEEL="$(find "$RAW_OUT" -name 'mistralrs-*.whl' | head -1)"
echo "==> built $RAW_WHEEL"

# --- patch: don't vendor libcuda.so.1, symlink to the real system driver instead ---
PATCH="$BUILD/patch-extract"
rm -rf "$PATCH" && mkdir -p "$PATCH"
cd "$PATCH"
unzip -q "$RAW_WHEEL"

VENDORED=$(find mistralrs.libs -maxdepth 1 -name 'libcuda-*.so.1')
[ -n "$VENDORED" ] || { echo "!! no vendored libcuda-*.so.1 found — wheel layout changed, check manually" >&2; exit 1; }
[ -e "$REAL_LIBCUDA" ] || { echo "!! $REAL_LIBCUDA missing — adjust path in this script" >&2; exit 1; }

rm "$VENDORED"
cp "$REAL_LIBCUDA" "$VENDORED"
echo "==> copied $REAL_LIBCUDA into $(basename "$VENDORED") (uv doesn't preserve symlinks on wheel install — see README)"

FIXED="$FIXED_DIR/$(basename "$RAW_WHEEL")"
zip -qr --symlinks "$FIXED" .
echo "==> wrote patched wheel: $FIXED"
echo "==> now run: uv add --group ask-sm86 '$FIXED'"
