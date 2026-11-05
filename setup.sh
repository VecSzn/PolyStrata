#!/bin/sh
# PolyStrata for macOS and Linux, following START-HERE.bat's bootstrap flow:
# reuse .venv, find/install Python, create .venv, then choose model, quantization and context.
# The launcher hands the choices and original arguments to the Mac or Linux installer.
cd "$(dirname "$0")" || exit 1

finish() {
  status=$1
  # Keep errors readable in a terminal, like the batch launcher's pause. Automation never waits.
  # Ctrl-C is a normal way to stop the server, not an installation error.
  if [ "$status" -ne 0 ] && [ "$status" -ne 130 ] && [ "$status" -ne 143 ] && [ -t 0 ] && [ -t 1 ]; then
    printf '\nPress Enter to close ... '
    read -r answer || :
  fi
  exit "$status"
}

fail() {
  printf '\n%s\n' "$1" >&2
  finish 1
}

# Debian/Ubuntu's venv needs the separate python3-venv package to provide ensurepip.
ok_py() {
  "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)' >/dev/null 2>&1
}

venv_ready() {
  [ -x .venv/bin/python ] &&
    .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)' >/dev/null 2>&1 &&
    .venv/bin/python -m pip --version >/dev/null 2>&1
}

find_python() {
  PY=""
  for candidate in python3 python python3.13 python3.12 python3.11 python3.10 .python/python/bin/python3; do
    if command -v "$candidate" >/dev/null 2>&1 && ok_py "$candidate"; then
      PY=$candidate
      return 0
    fi
  done
  return 1
}

install_mac_python() {
  [ "$(uname -m)" = arm64 ] || fail "Automatic Mac setup needs Apple Silicon running a native terminal (not Rosetta)."
  for utility in curl shasum tar mktemp; do
    command -v "$utility" >/dev/null 2>&1 || fail "Python could not be installed: $utility is missing. Install 64-bit Python 3.10 or newer, then run ./setup.sh again."
  done
  printf '\nPython 3.10+ is needed; downloading a standalone Python 3.12 into .python (25 MB, no sudo) ...\n'
  # Extract in a temporary folder so an interrupted download/extraction is never reused as Python.
  (
    mkdir -p .python || exit 1
    download_dir=$(mktemp -d .python/bootstrap.XXXXXX) || exit 1
    trap 'rm -rf "$download_dir"' 0
    trap 'exit 130' INT
    trap 'exit 143' HUP TERM
    pbs_url="https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.12.15%2B20261009-aarch64-apple-darwin-install_only.tar.gz"
    curl -fL --retry 3 -o "$download_dir/python.tar.gz" "$pbs_url" || exit 1
    digest=$(shasum -a 256 "$download_dir/python.tar.gz") || exit 1
    if [ "${digest%% *}" != "83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8" ]; then
      echo "The Python download does not match its pinned SHA-256." >&2
      exit 1
    fi
    tar xzf "$download_dir/python.tar.gz" -C "$download_dir" || exit 1
    ok_py "$download_dir/python/bin/python3" || exit 1
    rm -rf .python/python || exit 1
    mv "$download_dir/python" .python/python || exit 1
  ) || fail "Python could not be installed automatically. Install 64-bit Python 3.10 or newer, then run ./setup.sh again."
}

as_root() {
  if [ "$(id -u)" = 0 ]; then
    "$@"
  else
    command -v sudo >/dev/null 2>&1 || fail "Installing Python needs sudo. Ask your administrator to install Python 3.10+ with venv and pip, then run ./setup.sh again."
    sudo "$@"
  fi
}

install_linux_python() {
  printf '\nPython 3.10+ with venv and pip is needed; installing it (sudo may ask for your password) ...\n'
  if command -v apt-get >/dev/null 2>&1; then
    as_root apt-get update && as_root apt-get install -y python3 python3-venv python3-pip
  elif command -v dnf >/dev/null 2>&1; then
    as_root dnf install -y python3 python3-pip
  elif command -v pacman >/dev/null 2>&1; then
    as_root pacman -S --noconfirm python python-pip
  else
    fail "No supported Python installer was found. Install 64-bit Python 3.10+ with venv and pip, then run ./setup.sh again. On NixOS, install uv."
  fi || fail "Python installation failed. Fix the package-manager error above, then run ./setup.sh again."
}

create_venv() {
  # Keep the standard venv on ordinary Linux and macOS. uv is opt-in, or used on NixOS where the system
  # interpreter cannot load pip wheels' libraries. A working .venv is always reused, as on Windows.
  # Remove a broken active venv before searching PATH, which may still start with .venv/bin.
  if [ -e .venv ] || [ -L .venv ]; then
    echo "Recreating the incomplete or incompatible Python environment in .venv ..."
    rm -rf .venv || fail "Could not remove the broken .venv. Check its permissions, then run ./setup.sh again."
  fi
  UV=""
  if [ "${STRATA_UV:-}" = 1 ] || { [ "$platform" = Linux ] && grep -qi '^ID=nixos' /etc/os-release 2>/dev/null; }; then
    UV=$(command -v uv 2>/dev/null) || fail "uv is needed for NixOS or STRATA_UV=1. Install uv, then run ./setup.sh again."
  fi
  if [ -z "$UV" ] && ! find_python; then
    case "$platform" in
      Darwin) install_mac_python ;;
      Linux) install_linux_python ;;
    esac
    find_python || fail "Python 3.10+ with venv and pip is still unavailable. On Ubuntu/Debian, install python3-venv; then run ./setup.sh again."
  fi

  echo "Creating the private Python environment in .venv ..."
  if [ -n "$UV" ]; then
    "$UV" venv --managed-python --seed --python 3.12 .venv
  else
    "$PY" -m venv .venv
  fi || fail "Could not create the Python environment in .venv. On Ubuntu/Debian, install python3-venv; then run ./setup.sh again."
  venv_ready || fail "The Python environment in .venv is incomplete (Python 3.10+, 64-bit, and pip are required). Run ./setup.sh again to repair it."
}

run_setup() {
  if [ ! -t 0 ] || [ ! -t 1 ]; then
    # Preserve exit codes, process identity and signals for scripts and services.
    exec .venv/bin/python tools/setup_launcher.py "$@"
  fi
  .venv/bin/python tools/setup_launcher.py "$@"
  finish "$?"
}

# Match START-HERE.bat: a usable environment goes straight to setup, without another install.
if ! venv_ready; then
  platform=$(uname -s)
  case "$platform" in
    Darwin|Linux) create_venv ;;
    *) fail "setup.sh supports macOS and Linux. On Windows, run START-HERE.bat." ;;
  esac
fi
run_setup "$@"
