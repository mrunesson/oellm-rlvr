#!/bin/bash
# One-time setup for SkyRL + Harbor agentic training on CodeContests with Modal sandboxes, on lrd_all_serial
# (internet). Builds a separate venv from SkyRL v0.3.0's lockfile with the fsdp and harbor extras (so the GSM8K venv
# is untouched) plus Modal's proxy extra (python-socks, aiohttp-socks), stages the long-context Prelude-base
# (prelude-base@4eafcf9, 262k positions) with the reason tokenizer and the model's own chat template, extracts
# open-thoughts/CodeContests into Harbor task directories, and, if Modal credentials are present, starts one Modal
# sandbox from the task image (this also builds and caches the image on Modal). Idempotent; rerun after a failure.
#   sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_skyrl_harbor_setup.sbatch
# Needs uv in ~/.local/bin and the vLLM image (scripts/leonardo_vllm_container_pull.sbatch); see
# scripts/leonardo_skyrl_setup.sbatch for why the venv is built inside that image.
#SBATCH --job-name=skyrl-harbor-setup
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
: "${HARBOR_ROOT:=$ROOT/sources/harbor-3de07a0}"
: "${VENV:=$ROOT/venvs/skyrl-v030-harbor}"
: "${MODEL:=$ROOT/models/prelude-base-long-reason}"
: "${DATA:=$ROOT/data/harbor/CodeContests}"
: "${CONTAINER:=$ROOT/containers/vllm-openai-v0.32.0-x86_64-cu129.sif}"
: "${MODAL_PROBE:=1}"
# Modal 1.5.1 (SkyRL's pin) knows image builders up to 2025.06; a workspace set to a newer one (2026.10 seen on
# 2026-11-26) makes every image build fail with VersionError. This asks for 2025.06 for this job's images only; the
# workspace setting is unchanged.
: "${MODAL_IMAGE_BUILDER_VERSION:=2025.06}"
export MODAL_IMAGE_BUILDER_VERSION
SKYRL_COMMIT=f5bc3b78dfddfb352870d5d7430cd226e5785838   # skyrl-v0.3.0; its lock pins harbor 3de07a0 and modal 1.5.1
HARBOR_COMMIT=3de07a0e01f3368921766437fc7afece3ddec23d  # must equal the git rev in SkyRL's uv.lock (checked below)
# "Replace with prelude-long" (2026-11-25): max_position_embeddings 262144, rope_theta 2e7, vocab 262145.
MODEL_REPO=openeurollm/prelude-base MODEL_REVISION=4eafcf909c431ef4fff6c84e849a58bd4cb99dc1
# The model repo's own tokenizer has <start_of_turn>/<end_of_turn> in ids 3/4, so its chat template's <|im_start|> and
# <|im_end|> would be plain text. The reason tokenizer (same merges, max id 262144) has them as ids 3/4.
TOKENIZER_REPO=openeurollm/tokenizer-256k-reason TOKENIZER_REVISION=baaafeb0375b5fa55fedeef8f5eedce7d4d242dd
TEMPLATE_URL=https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REVISION/chat_template.jinja   # the model's own
TEMPLATE_SHA256=a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8
DATASET=open-thoughts/CodeContests DATASET_REVISION=11f66f5e81d8035f44c3a576ed6772994d1ed90b   # 9,644 tasks

export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || { echo "uv not found; install it on a login node first" >&2; exit 1; }
[[ -r $CONTAINER ]] || { echo "no container at $CONTAINER (sbatch scripts/leonardo_vllm_container_pull.sbatch)" >&2; exit 1; }
export UV_CACHE_DIR=$ROOT/cache/uv UV_PYTHON_INSTALL_DIR=$ROOT/cache/uv-python HF_HOME=$ROOT/hf-cache
mkdir -p logs "$(dirname "$SKYRL_ROOT")" "$(dirname "$VENV")" "$(dirname "$DATA")" "$UV_CACHE_DIR"
IN=(singularity exec -B /leonardo_work,/leonardo_scratch "$CONTAINER" env -u PYTHONPATH -u VIRTUAL_ENV)
"${IN[@]}" ldd --version | head -1

echo "=== 1. SkyRL $SKYRL_COMMIT -> $SKYRL_ROOT"
[[ -d $SKYRL_ROOT/.git ]] || git clone --filter=blob:none https://github.com/NovaSky-AI/SkyRL.git "$SKYRL_ROOT"
git -C "$SKYRL_ROOT" fetch --depth 1 origin "$SKYRL_COMMIT"
git -C "$SKYRL_ROOT" -c advice.detachedHead=false checkout --detach "$SKYRL_COMMIT"
[[ $(git -C "$SKYRL_ROOT" rev-parse HEAD) == "$SKYRL_COMMIT" ]]

# Harbor is a git source in SkyRL's lockfile, and the container has no git for uv to fetch it. So the host's git
# clones it at the locked commit, uv sync skips it (--no-install-package harbor; that path needs no git), and it is
# installed from this clone with --no-deps below (its dependencies come from the lockfile through the harbor extra).
grep -q "harbor?rev=$HARBOR_COMMIT#" "$SKYRL_ROOT/uv.lock" || { echo "SkyRL's uv.lock does not pin harbor $HARBOR_COMMIT" >&2; exit 1; }
echo "=== 1b. Harbor $HARBOR_COMMIT -> $HARBOR_ROOT"
[[ -d $HARBOR_ROOT/.git ]] || git clone --filter=blob:none https://github.com/laude-institute/harbor.git "$HARBOR_ROOT"
git -C "$HARBOR_ROOT" fetch --depth 1 origin "$HARBOR_COMMIT"
git -C "$HARBOR_ROOT" -c advice.detachedHead=false checkout --detach "$HARBOR_COMMIT"
[[ $(git -C "$HARBOR_ROOT" rev-parse HEAD) == "$HARBOR_COMMIT" ]]

echo "=== 2. venv (fsdp + harbor extras) -> $VENV"
UV_ENV=(sh -c 'for v in $(env | sed -n "s/^\(UV_[A-Z_]*\)=.*/\1/p"); do unset "$v"; done
  unset VIRTUAL_ENV PYTHONPATH; exec env "$@"' sh UV_CACHE_DIR="$UV_CACHE_DIR" UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR")
(cd "$SKYRL_ROOT" && "${IN[@]}" "${UV_ENV[@]}" UV_PROJECT_ENVIRONMENT="$VENV" "$(command -v uv)" \
  sync --frozen --extra fsdp --extra harbor --no-dev --no-install-package harbor)
# `uv sync` removes packages it did not install, so both of these run after every sync:
# Harbor from the host-git clone, and Modal's api-proxy-support extra (python-socks, aiohttp-socks; not in SkyRL's
# lock), without which Modal cannot use the proxy.
"${IN[@]}" "${UV_ENV[@]}" "$(command -v uv)" pip install --python "$VENV/bin/python" --no-deps "$HARBOR_ROOT"
"${IN[@]}" "${UV_ENV[@]}" "$(command -v uv)" pip install --python "$VENV/bin/python" \
  "python-socks[asyncio]>=2.5" "aiohttp-socks>=0.8"
"${IN[@]}" "$VENV/bin/python" - <<'PY'
import importlib, sys
failed = False
for name in ("torch", "vllm", "transformers", "ray", "flash_attn", "skyrl", "harbor", "modal", "litellm",
             "python_socks", "aiohttp_socks"):
    try:
        module = importlib.import_module(name)
        print(f"{name:14s}", getattr(module, "__version__", "ok"))
    except Exception as error:
        failed = True
        print(f"{name:14s} IMPORT FAILED: {type(error).__name__}: {error}")
from harbor.environments.modal import ModalEnvironment  # noqa: F401  (fails if the modal extra is missing)
print("harbor modal environment ok")
sys.exit(1 if failed else 0)
PY

echo "=== 3. model -> $MODEL"
if [[ -r $MODEL/config.json ]] && grep -q "\"$MODEL_REVISION\"" "$MODEL/SOURCE.json" 2>/dev/null \
   && grep -q "\"$TEMPLATE_SHA256\"" "$MODEL/SOURCE.json"; then
  echo "already staged:"; cat "$MODEL/SOURCE.json"
else
  TEMPLATE=$ROOT/cache/prelude-base-${MODEL_REVISION:0:7}-chat_template.jinja
  [[ -s $TEMPLATE ]] || curl -fsSL -o "$TEMPLATE" "$TEMPLATE_URL"
  echo "$TEMPLATE_SHA256  $TEMPLATE" | sha256sum -c -
  "${IN[@]}" "$VENV/bin/python" "$CONTROL_ROOT/scripts/stage_chat_model.py" \
    --model "$MODEL_REPO" --model-revision "$MODEL_REVISION" \
    --tokenizer "$TOKENIZER_REPO" --tokenizer-revision "$TOKENIZER_REVISION" \
    --chat-template "$TEMPLATE" --output "$MODEL"
fi

echo "=== 4. $DATASET@$DATASET_REVISION -> $DATA"
(cd "$SKYRL_ROOT" && "${IN[@]}" HF_HOME="$HF_HOME" "$VENV/bin/python" - "$DATASET" "$DATASET_REVISION" "$DATA" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from examples.train_integrations.harbor.prepare_harbor_dataset import extract_parquet
repo, revision, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
snapshot = Path(snapshot_download(repo_id=repo, repo_type="dataset", revision=revision))
total = sum(extract_parquet(p, out) for p in sorted(snapshot.glob("**/*.parquet")))
tasks = sorted(d for d in out.iterdir() if (d / "instruction.md").is_file())
print(f"{total} extracted; {len(tasks)} task directories in {out}")
if len(tasks) != 9644:
    sys.exit(f"expected 9644 tasks, found {len(tasks)}")
(out.parent / "CodeContests.SOURCE").write_text(f"{repo}@{revision}\n")
PY
)

echo "=== 5. Modal sandbox probe (direct internet)"
if [[ $MODAL_PROBE != 1 ]]; then
  echo "skipped (MODAL_PROBE=$MODAL_PROBE)"
elif [[ -r $HOME/.modal.toml || -n ${MODAL_TOKEN_ID:-} ]]; then
  "${IN[@]}" "$VENV/bin/python" "$CONTROL_ROOT/scripts/probe_modal_sandbox.py" --task-dir "$DATA/code_contests-0000"
else
  echo "no Modal credentials (~/.modal.toml or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET); create them on a login node with"
  echo "  \"$VENV/bin/modal\" token set --token-id <id> --token-secret <secret>   (inside the container) and rerun"
fi

echo "skyrl_root=$SKYRL_ROOT"
echo "venv=$VENV"
echo "model=$MODEL"
echo "data=$DATA"
