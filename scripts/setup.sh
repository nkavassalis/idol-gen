#!/usr/bin/env bash
# Environment setup for idol-gen: one image -> a 3D Gaussian splat.
#
#   scripts/setup.sh              create .venv, install everything, prefetch weights, check Hugging Face access
#   scripts/setup.sh --check      verify an existing setup and change nothing
#   scripts/setup.sh --rebuild    delete .venv and start over
#   scripts/setup.sh --no-weights skip the ~3.4 GB model weight download
#   scripts/setup.sh --cpu        install without requiring an NVIDIA GPU (generation still needs CUDA)
#
#   SDK_TAG=v0.1.0 scripts/setup.sh    pin the upstream SDK to a tag or commit
#
# Linux and macOS (and WSL). On Windows, use: pip install -e ".[view]"  — see README.md.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

SDK_REPO="${SDK_REPO:-https://github.com/speridlabs/idol-sdk.git}"
SDK_TAG="${SDK_TAG:-}"
MODE=install
WANT_WEIGHTS=1
REQUIRE_GPU=1

for arg in "$@"; do
  case "$arg" in
    --check) MODE=check ;;
    --rebuild) MODE=install; WANT_REBUILD=1 ;;
    --no-weights) WANT_WEIGHTS=0 ;;
    --cpu) REQUIRE_GPU=0 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$1"; }
die() { printf '\nerror: %s\n' "$1" >&2; exit "${2:-1}"; }

# ---------------------------------------------------------------- verify / rebuild

if [ "${MODE}" = check ]; then
  step "checking the existing environment"
  [ -x .venv/bin/python ] || die "no .venv — run scripts/setup.sh first"
else
  [ "${WANT_REBUILD:-0}" = 1 ] && rm -rf .venv
fi

# ---------------------------------------------------------------- virtualenv

step "python environment"
if command -v uv >/dev/null 2>&1; then
  [ -x .venv/bin/python ] || uv venv .venv --python 3.10 || uv venv .venv
  echo "  uv + .venv"
else
  if [ ! -x .venv/bin/python ]; then
    python=""
    for candidate in python3.12 python3.11 python3.10 python3; do
      if command -v "$candidate" >/dev/null 2>&1 &&
         "$candidate" -c 'import sys; raise SystemExit(not (3,10) <= sys.version_info[:2] <= (3,13))' 2>/dev/null; then
        python="$candidate"; break
      fi
    done
    [ -n "$python" ] || die "need Python 3.10-3.13 (or install uv: https://docs.astral.sh/uv/)"
    "$python" -m venv .venv
  fi
  PYTHONNOUSERSITE=1 .venv/bin/python -m pip install -q --upgrade pip
  echo "  venv + pip ($(.venv/bin/python --version 2>&1))"
fi
PY=.venv/bin/python
export PYTHONNOUSERSITE=1        # ~/.local/lib can shadow venv packages and break diffusers

# ---------------------------------------------------------------- GPU

step "GPU"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "  $(nvidia-smi -L | head -1)"
elif [ "$REQUIRE_GPU" = 1 ]; then
  echo "  no NVIDIA GPU detected."
  echo "  Idol needs CUDA to generate in reasonable time (~7 GB VRAM at default settings)."
  echo "  Install anyway? Re-run with --cpu, or try the hosted demo: https://huggingface.co/spaces/speridlabs/idol"
  die "no GPU" 3
else
  echo "  no GPU (--cpu: installing anyway; generation will be impractically slow)"
fi

# ---------------------------------------------------------------- install

if [ "$MODE" = install ]; then
  step "installing the upstream SDK (pinned: ${SDK_TAG:-main})"
  if [ ! -d idol-sdk/.git ]; then
    git clone --depth 1 ${SDK_TAG:+--branch "$SDK_TAG"} "$SDK_REPO" idol-sdk
  else
    echo "  idol-sdk/ already present, leaving it alone (rm -rf idol-sdk to re-clone)"
  fi

  step "installing dependencies and the idol-gen CLI"
  if command -v uv >/dev/null 2>&1; then
    VIRTUAL_ENV="$here/.venv" uv pip install -e ".[view]"
    VIRTUAL_ENV="$here/.venv" uv pip install -e ./idol-sdk      # local clone wins over the git dependency
  else
    .venv/bin/pip install -e ".[view]"
    .venv/bin/pip install -e ./idol-sdk
  fi
fi

step "imports"
$PY - <<'PY' || die "import check failed — see the traceback above"
import torch, transformers, diffusers, tensordict, plyfile, viser
from idol.pipeline import IdolPipeline
from idol_tools.generate import build_parser
from idol_tools.viewer import load_ply
from diffusers.models import AutoencoderKLFlux2
print(f'  torch {torch.__version__}  transformers {transformers.__version__}  '
      f'diffusers {diffusers.__version__}  viser {viser.__version__}')
print(f'  cuda available: {torch.cuda.is_available()}' +
      (f'  gpu: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else ''))
PY

# ---------------------------------------------------------------- weights

if [ "$MODE" = install ] && [ "$WANT_WEIGHTS" = 1 ]; then
  step "prefetching Idol weights (~3.4 GB, ungated) into the Hugging Face cache"
  $PY - <<'PY'
from huggingface_hub import snapshot_download
from idol.pipeline import WEIGHT_FILES
print('  ' + snapshot_download('speridlabs/idol', allow_patterns=WEIGHT_FILES))
PY
fi

# ---------------------------------------------------------------- Hugging Face access

step "access to the two gated image encoders"
if $PY - <<'PY'
from idol.conditioning import EncoderAccessError, require_encoder_access
try:
    require_encoder_access(['facebook/dinov3-vith16plus-pretrain-lvd1689m', 'black-forest-labs/FLUX.2-dev'])
except EncoderAccessError as error:
    raise SystemExit(str(error))
PY
then
  echo "  encoder access OK"
else
  cat <<'TXT'

  Not yet — Idol reads its input through two gated models. Accept the terms with your Hugging Face
  account, then log in:

    https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m
    https://huggingface.co/black-forest-labs/FLUX.2-dev          (only its VAE is used)

    hf auth login                     # or: export HF_TOKEN=hf_...

  Re-check any time:  ./idol-gen some.png --dry-run
TXT
fi

# ---------------------------------------------------------------- tests + summary

step "checks that need no GPU or weights"
$PY -m idol_tools.smoke_test --cli-only | tail -1

if [ "$MODE" = check ]; then
  echo
  echo "Environment looks good."
  exit 0
fi

cat <<TXT

Done. Next steps:

  ./idol-gen --help                                    all the quality levers
  ./idol-gen path/to/image.jpg --mode real --preset draft
  ./idol-gen path/to/image.jpg --serve --open          generate, then view in the browser
  ./idol-view outputs/                                 view .ply files you already have
  ./idol-test                                          full test (real weights, needs a GPU)

  No GPU? The hosted demo is at https://huggingface.co/spaces/speridlabs/idol
TXT
