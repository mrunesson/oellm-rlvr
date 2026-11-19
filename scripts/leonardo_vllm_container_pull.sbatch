#!/bin/bash
# Pull the public vLLM OpenAI-server image into a SIF on an internet node (lrd_all_serial), for accounts without access to a
# project vLLM container. The cu129 build (CUDA 12.9) runs on A100 drivers >= 525; newer vLLM tags are CUDA 13 only.
#   sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_vllm_container_pull.sbatch
# Use it with: VLLM_CONTAINER=<sif> VLLM_PY=python3 (leonardo_laion_agent_rollout.sbatch, leonardo_vllm_smoke.sbatch).
#SBATCH --job-name=vllm-pull
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.%x.out
set -euo pipefail
: "${OELLM_SCRATCH:?set the OELLM_* variables first}"
: "${VLLM_TAG:=v0.32.0-x86_64-cu129}"
: "${VLLM_IMAGE:=docker://vllm/vllm-openai:$VLLM_TAG}"
DIR=$OELLM_SCRATCH/oellm-rl/containers
DEST=$DIR/vllm-openai-$VLLM_TAG.sif
# Cache and temp on scratch (home is small); one per job, removed at the end.
export SINGULARITY_CACHEDIR=$DIR/cache-${SLURM_JOB_ID:-manual} SINGULARITY_TMPDIR=$DIR/tmp-${SLURM_JOB_ID:-manual}
mkdir -p "$DIR" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" logs
trap 'rm -rf "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"' EXIT
if [[ -s $DEST ]]; then echo "already there: $DEST"; else
  echo "pulling $VLLM_IMAGE -> $DEST"
  singularity pull "$DEST.part" "$VLLM_IMAGE"
  mv "$DEST.part" "$DEST"
fi
ls -la "$DEST"
singularity exec "$DEST" python3 -c 'import vllm, transformers, torch; print("vllm", vllm.__version__, "transformers", transformers.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
echo "vllm_container=$DEST"
