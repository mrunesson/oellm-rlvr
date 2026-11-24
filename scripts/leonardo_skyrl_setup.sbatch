#!/bin/bash
# One-time setup for SkyRL GRPO on Leonardo, on lrd_all_serial (internet): SkyRL v0.3.0 from its lockfile into a venv
# (CUDA 12.8 torch: runs on the R535 driver; v0.4+ needs CUDA 13), Prelude-base staged with the reason tokenizer and
# the OpenEuroLLM qwen3 chat template, and GSM8K in SkyRL's format. Idempotent; rerun after a failure.
#   sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_skyrl_setup.sbatch
# Needs uv in ~/.local/bin (curl -LsSf https://astral.sh/uv/install.sh | sh on a login node) and an Ubuntu 22.04+
# container: SkyRL's vllm-router wheel needs glibc 2.35 and Leonardo (RHEL 8) has 2.28, so the venv is built and run
# inside the vLLM image (scripts/leonardo_vllm_container_pull.sbatch). Only its OS is used; Python and packages come
# from the venv.
#SBATCH --job-name=skyrl-setup
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.%x.out
set -euo pipefail
: "${OELLM_SCRATCH:?set the OELLM_* variables first}"
: "${CONTROL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${ROOT:=$OELLM_SCRATCH/oellm-rl}"
: "${SKYRL_ROOT:=$ROOT/sources/SkyRL-v0.3.0}"
: "${VENV:=$ROOT/venvs/skyrl-v030}"
: "${MODEL:=$ROOT/models/prelude-base-reason}"
: "${DATA:=$ROOT/data/gsm8k-skyrl}"
: "${CONTAINER:=$ROOT/containers/vllm-openai-v0.32.0-x86_64-cu129.sif}"
SKYRL_COMMIT=f5bc3b78dfddfb352870d5d7430cd226e5785838   # skyrl-v0.3.0, the version qualified on LUMI
MODEL_REPO=openeurollm/prelude-base MODEL_REVISION=7ab2bc679c682dccd9852de688199505be58695f
TOKENIZER_REPO=openeurollm/tokenizer-256k-reason TOKENIZER_REVISION=baaafeb0375b5fa55fedeef8f5eedce7d4d242dd
# OpenEuroLLM/post-training src/post_training/chat_templates/templates/qwen3.jinja at 9ac9328
TEMPLATE_URL=https://raw.githubusercontent.com/OpenEuroLLM/post-training/9ac93288b9d15c6a7a9e58965fdabae73a1a2ba4/src/post_training/chat_templates/templates/qwen3.jinja
TEMPLATE_SHA256=9c0479e87a6a3445c3b2a0c11827a3fa2604175874f297aca9efa92101a8cd56

export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || { echo "uv not found; install it on a login node first" >&2; exit 1; }
[[ -r $CONTAINER ]] || { echo "no container at $CONTAINER (sbatch scripts/leonardo_vllm_container_pull.sbatch)" >&2; exit 1; }
# Caches on scratch (home is small).
export UV_CACHE_DIR=$ROOT/cache/uv UV_PYTHON_INSTALL_DIR=$ROOT/cache/uv-python HF_HOME=$ROOT/hf-cache
mkdir -p logs "$(dirname "$SKYRL_ROOT")" "$(dirname "$VENV")" "$DATA" "$UV_CACHE_DIR"
IN=(singularity exec -B /leonardo_work,/leonardo_scratch "$CONTAINER" env -u PYTHONPATH -u VIRTUAL_ENV)
"${IN[@]}" ldd --version | head -1

echo "=== 1. SkyRL $SKYRL_COMMIT -> $SKYRL_ROOT"
[[ -d $SKYRL_ROOT/.git ]] || git clone --filter=blob:none https://github.com/NovaSky-AI/SkyRL.git "$SKYRL_ROOT"
git -C "$SKYRL_ROOT" fetch --depth 1 origin "$SKYRL_COMMIT"
git -C "$SKYRL_ROOT" -c advice.detachedHead=false checkout --detach "$SKYRL_COMMIT"
[[ $(git -C "$SKYRL_ROOT" rev-parse HEAD) == "$SKYRL_COMMIT" ]]

echo "=== 2. venv from SkyRL's lockfile -> $VENV"
# The image sets its own UV_* (a read-only /opt/uv/cache) over ours: drop them all, then set the job's.
(cd "$SKYRL_ROOT" && "${IN[@]}" sh -c 'for v in $(env | sed -n "s/^\(UV_[A-Z_]*\)=.*/\1/p"); do unset "$v"; done
  unset VIRTUAL_ENV PYTHONPATH; exec env "$@"' sh UV_CACHE_DIR="$UV_CACHE_DIR" UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
  UV_PROJECT_ENVIRONMENT="$VENV" "$(command -v uv)" sync --frozen --extra fsdp --no-dev)
"${IN[@]}" "$VENV/bin/python" - <<'PY'
import importlib
for name in ("torch", "vllm", "transformers", "ray", "flash_attn", "skyrl", "skyrl_gym"):
    try:
        module = importlib.import_module(name)
        print(f"{name:13s}", getattr(module, "__version__", "ok"))
    except Exception as error:  # report every package, then fail below
        print(f"{name:13s} IMPORT FAILED: {type(error).__name__}: {error}")
import torch
print("torch cuda", torch.version.cuda)
PY

echo "=== 3. model -> $MODEL"
TEMPLATE=$ROOT/cache/qwen3-9ac9328.jinja
[[ -s $TEMPLATE ]] || curl -fsSL -o "$TEMPLATE" "$TEMPLATE_URL"
echo "$TEMPLATE_SHA256  $TEMPLATE" | sha256sum -c -
"${IN[@]}" "$VENV/bin/python" "$CONTROL_ROOT/scripts/stage_chat_model.py" \
  --model "$MODEL_REPO" --model-revision "$MODEL_REVISION" \
  --tokenizer "$TOKENIZER_REPO" --tokenizer-revision "$TOKENIZER_REVISION" \
  --chat-template "$TEMPLATE" --output "$MODEL"

echo "=== 4. GSM8K -> $DATA"
if [[ ! -s $DATA/train.parquet || ! -s $DATA/validation.parquet ]]; then
  (cd "$SKYRL_ROOT" && "${IN[@]}" "$VENV/bin/python" examples/train/gsm8k/gsm8k_dataset.py --output_dir "$DATA")
fi
"${IN[@]}" "$VENV/bin/python" -c 'import sys, pyarrow.parquet as pq
for split in ("train", "validation"):
    t = pq.read_table(f"{sys.argv[1]}/{split}.parquet"); print(split, t.num_rows, "rows; first prompt:", t.column("prompt")[0].as_py())' "$DATA"

echo "skyrl_root=$SKYRL_ROOT"
echo "venv=$VENV"
echo "container=$CONTAINER"
echo "model=$MODEL"
echo "data=$DATA"
