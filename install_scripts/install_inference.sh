#!/usr/bin/env bash
# install_inference.sh
# Sets up the .venv_inference venv for running VLA inference with
# Isaac-GR00T PolicyClient against a remote or local policy server.
#
# Installs the local Isaac-GR00T checkout plus gear_sonic[inference], which pulls in
# PyZMQ, msgpack, Pinocchio, and other inference dependencies.
#
# Usage:  bash install_scripts/install_inference.sh   (run from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ISAAC_GROOT_ROOT="$(cd "$REPO_ROOT/../Isaac-GR00T" 2>/dev/null && pwd || true)"

# ── 0. System dependencies ────────────────────────────────────────────────────
ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

# ── 1. Ensure uv is installed and available ──────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found – installing via official installer …"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv installation succeeded but binary not found on PATH."
        echo "        Please add ~/.local/bin (or ~/.cargo/bin) to your PATH and re-run."
        exit 1
    fi
fi
echo "[OK] uv $(uv --version)"

# ── 2. Install a uv-managed Python 3.10 (includes dev headers / Python.h) ────
echo "[INFO] Installing uv-managed Python 3.10 (includes development headers) …"
uv python install 3.10
MANAGED_PY="$(uv python find --no-project 3.10)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv (if any) ──────────────────────────────────────────
cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_inference (if present) …"
rm -rf .venv_inference

# ── 4. Create venv & install inference extra ─────────────────────────────────
echo "[INFO] Creating .venv_inference with uv-managed Python 3.10 …"
uv venv .venv_inference --python "$MANAGED_PY" --prompt gear_sonic_inference
# shellcheck disable=SC1091
source .venv_inference/bin/activate
if [ -z "$ISAAC_GROOT_ROOT" ] || [ ! -f "$ISAAC_GROOT_ROOT/pyproject.toml" ]; then
    echo "[ERROR] Local Isaac-GR00T checkout not found at:"
    echo "        $REPO_ROOT/../Isaac-GR00T"
    echo "        Clone/install Isaac-GR00T there, then re-run this script."
    exit 1
fi
echo "[INFO] Installing local Isaac-GR00T from $ISAAC_GROOT_ROOT …"
uv pip install -e "$ISAAC_GROOT_ROOT"
echo "[INFO] Installing gear_sonic[inference] (this may take a few minutes) …"
uv pip install -e "gear_sonic[inference]"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the venv with:"
echo ""
echo "    source .venv_inference/bin/activate"
echo ""
echo "  You should see (gear_sonic_inference) in your prompt."
echo ""
echo "  Then run VLA inference with:"
echo "    python gear_sonic/scripts/run_vla_inference.py --help"
echo "══════════════════════════════════════════════════════════════"
