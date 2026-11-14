#!/usr/bin/env bash
# lfm installer: python package + llama.cpp + the model, all in user space.
set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="$REPO_DIR/.venv"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
DOWNLOAD_MODEL=1

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }
die() { printf '\033[1merror:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: ./install.sh [--no-model]

  --no-model   skip the 1.13 GB model download; "lfm" then prints how to fetch it later
  -h, --help   this text
EOF
}

for arg in "$@"; do
    case "$arg" in
        --no-model) DOWNLOAD_MODEL=0 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $arg (try --help)" ;;
    esac
done

# ---------------------------------------------------------------- platform
system="$(uname -s)"
machine="$(uname -m)"
case "$system/$machine" in
    Linux/x86_64|Linux/amd64) ;;
    Darwin/arm64)
        bold "note: macOS is untested — continuing, but Linux x86_64 is the supported target"
        ;;
    *)
        die "unsupported platform: $system/$machine. lfm needs Linux x86_64 (macOS arm64 is untested)."
        ;;
esac

# ---------------------------------------------------------------- python
command -v "$PYTHON" >/dev/null 2>&1 \
    || die "$PYTHON not found. Install Python 3.11+ (Arch: sudo pacman -S python; Debian/Ubuntu: sudo apt install python3 python3-venv)"

"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || die "$("$PYTHON" -V 2>&1) is too old. lfm needs Python 3.11+."

# ---------------------------------------------------------------- venv
if [ ! -x "$VENV/bin/python" ]; then
    bold "creating $VENV"
    if ! "$PYTHON" -m venv "$VENV" 2>/tmp/lfm-venv-err; then
        cat /tmp/lfm-venv-err >&2 || true
        rm -f /tmp/lfm-venv-err
        die "could not create a virtualenv. On Debian/Ubuntu install python3-venv; on Arch python ships it."
    fi
    rm -f /tmp/lfm-venv-err
fi

bold "installing lfm into $VENV"

install_package() {
    if "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
        "$VENV/bin/python" -m pip install --quiet -e "$REPO_DIR" && return 0
    fi
    if "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
        "$VENV/bin/python" -m pip install --quiet -e "$REPO_DIR" && return 0
    fi
    if command -v uv >/dev/null 2>&1; then
        uv pip install --quiet --python "$VENV/bin/python" -e "$REPO_DIR" && return 0
    fi
    return 1
}

if ! install_package; then
    die "could not install the package — install pip or uv and rerun ./install.sh"
fi

# ---------------------------------------------------------------- launcher
mkdir -p "$BIN_DIR"
ln -sfn "$VENV/bin/lfm" "$BIN_DIR/lfm"
note "launcher: $BIN_DIR/lfm"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) bold "note: $BIN_DIR is not in PATH — add it to your shell profile to run 'lfm' from anywhere" ;;
esac

# ---------------------------------------------------------------- llama.cpp + model
bold "installing llama.cpp and the model (this can take a while)"
if [ "$DOWNLOAD_MODEL" -eq 1 ]; then
    "$BIN_DIR/lfm" --setup --download-model
else
    "$BIN_DIR/lfm" --setup
fi

bold "done. run: lfm"
