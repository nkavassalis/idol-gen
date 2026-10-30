#!/usr/bin/env bash
# One-time setup for the Idol image-to-3D project: vendor the SDK, build .venv, install deps, prefetch weights.
#   scripts/setup.sh
set -euo pipefail
# this script lives in <root>/scripts/, everything else is created in <root>
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

SDK_TAG="${SDK_TAG:-}"                      # optional: pin a commit/tag of idol-sdk
SDK_REPO="${SDK_REPO:-https://github.com/speridlabs/idol-sdk.git}"

if [ ! -d idol-sdk/.git ]; then
  echo "==> cloning idol-sdk"
  git clone --depth 1 ${SDK_TAG:+--branch "$SDK_TAG"} "$SDK_REPO" idol-sdk
else
  echo "==> idol-sdk already present, not touching it (rm -rf idol-sdk to re-clone)"
fi

echo "==> creating .venv (isolated: no system / ~/.local site-packages)"
if [ -x .venv/bin/python ]; then
  echo "  .venv already exists, reusing it (rm -rf .venv to rebuild from scratch)"
elif command -v uv >/dev/null 2>&1; then
  uv venv .venv --python 3.10
else
  python3 -m venv .venv
fi

if command -v uv >/dev/null 2>&1; then
  VIRTUAL_ENV="$here/.venv" uv pip install -e ./idol-sdk
else
  PYTHONNOUSERSITE=1 .venv/bin/pip install -U pip
  PYTHONNOUSERSITE=1 .venv/bin/pip install -e ./idol-sdk
fi

# ~/.local/lib/python3.10/site-packages leaks an old peft into venvs that include system site-packages,
# which breaks diffusers' peft loader. Refuse to continue if that happens.
echo "==> checking imports"
PYTHONNOUSERSITE=1 .venv/bin/python - <<'PY'
import torch, transformers, diffusers
from idol.pipeline import IdolPipeline
from diffusers.models import AutoencoderKLFlux2
from transformers import AutoModel
assert torch.cuda.is_available(), 'no CUDA GPU visible: Idol needs an NVIDIA GPU with CUDA'
print(f'  torch {torch.__version__}  transformers {transformers.__version__}  '
      f'diffusers {diffusers.__version__}  gpu {torch.cuda.get_device_name(0)}')
PY

echo "==> prefetching Idol weights (~3.4 GB, ungated) into the Hugging Face cache"
PYTHONNOUSERSITE=1 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from idol.pipeline import WEIGHT_FILES
print('  ' + snapshot_download('speridlabs/idol', allow_patterns=WEIGHT_FILES))
PY

echo "==> checking access to the two gated image encoders"
if PYTHONNOUSERSITE=1 .venv/bin/python - <<'PY'
from idol.conditioning import EncoderAccessError, require_encoder_access
try:
    require_encoder_access(['facebook/dinov3-vith16plus-pretrain-lvd1689m', 'black-forest-labs/FLUX.2-dev'])
except EncoderAccessError as error:
    raise SystemExit(str(error))
PY
then
  echo "  encoder access OK"
else
  echo
  echo "  !! Not yet. Accept the terms, then log in:"
  echo "     https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m"
  echo "     https://huggingface.co/black-forest-labs/FLUX.2-dev   (only its VAE is used)"
  echo "     hf auth login        # or: export HF_TOKEN=hf_..."
  echo "  Re-check any time with: ./idol-gen some.png --dry-run"
fi

echo
echo "Done. Try:"
echo "  ./idol-gen idol-sdk/demo/examples/food-donut.webp --preset draft"
echo "  ./idol-gen --help"
