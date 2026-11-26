#!/bin/bash
# Token-protected HTTP CONNECT proxy on an internet node (lrd_all_serial), so the offline GPU nodes can reach Modal
# (scripts/leonardo_skyrl_harbor_codecontests.sbatch). lrd_all_serial allows 4 h, so each proxy job submits its
# successor to start PROXY_OVERLAP_MIN minutes before it exits; the successor rewrites $PROXY_DIR/proxy.host and the
# training job's loopback relay picks the new address up on its next connection.
#   sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_modal_proxy.sbatch
# The chain stops when $PROXY_DIR/STOP exists, or (after the first generation) when no job named $WATCH_JOB_NAME of
# yours is pending or running. Stop it by hand with: touch $PROXY_DIR/STOP; scancel -n modal-proxy
#SBATCH --job-name=modal-proxy
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.%x.out
set -euo pipefail
: "${OELLM_SCRATCH:?set the OELLM_* variables first}"
: "${CONTROL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${ROOT:=$OELLM_SCRATCH/oellm-rl}"
: "${PROXY_DIR:=$ROOT/proxy-modal}"
: "${VENV:=$ROOT/venvs/skyrl-v030-harbor}"   # any Python >= 3.8 works; the system one is 3.6
: "${CONTAINER:=$ROOT/containers/vllm-openai-v0.32.0-x86_64-cu129.sif}"
: "${PROXY_SECONDS:=14000}" "${PROXY_OVERLAP_MIN:=30}" "${CHAIN:=1}" "${PROXY_GEN:=1}"
: "${WATCH_JOB_NAME:=skyrl-cc-modal}"
mkdir -p "$PROXY_DIR" logs; chmod 700 "$PROXY_DIR"

if [[ -e $PROXY_DIR/STOP ]]; then echo "$PROXY_DIR/STOP exists; not starting"; exit 0; fi
if (( PROXY_GEN > 1 )) && [[ -n $WATCH_JOB_NAME ]] && [[ -z $(squeue -h -u "$USER" -n "$WATCH_JOB_NAME" -t PD,R,CF -o %i) ]]; then
  echo "no $WATCH_JOB_NAME job pending or running; chain ends at generation $PROXY_GEN"; exit 0
fi
if [[ $CHAIN == 1 ]]; then
  BEGIN_MIN=$(( PROXY_SECONDS / 60 - PROXY_OVERLAP_MIN ))
  # A refused submission (e.g. a per-user job limit on lrd_all_serial) must not stop this proxy from serving.
  if NEXT=$(cd "$CONTROL_ROOT" && sbatch --parsable -A "$SLURM_JOB_ACCOUNT" --begin="now+${BEGIN_MIN}minutes" \
      --export=ALL,PROXY_GEN=$((PROXY_GEN + 1)),CONTROL_ROOT="$CONTROL_ROOT" scripts/leonardo_modal_proxy.sbatch 2>&1); then
    echo "generation $PROXY_GEN; successor $NEXT starts in $BEGIN_MIN min"
  else
    echo "WARNING: generation $PROXY_GEN could not submit its successor ($NEXT); the chain ends with this proxy" >&2
  fi
fi
singularity exec -B /leonardo_work,/leonardo_scratch "$CONTAINER" env -u PYTHONPATH "$VENV/bin/python" -I \
  "$CONTROL_ROOT/scripts/laion_proxy.py" 0 "$PROXY_DIR/proxy.host" "$PROXY_DIR/proxy.token" "$PROXY_SECONDS"
