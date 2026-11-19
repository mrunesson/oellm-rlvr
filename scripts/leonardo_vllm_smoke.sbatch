#!/bin/bash
# Smoke test before a rollout: GPU driver, vLLM in the container sees CUDA, the model loads with the same server flags as
# leonardo_laion_agent_rollout.sbatch, and answers one chat request. One GPU, debug queue.
#   MODEL=$OELLM_SCRATCH/oellm-rl/models/<model> VLLM_CONTAINER=<sif> \
#     sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_vllm_smoke.sbatch
#SBATCH --job-name=vllm-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.%x.out
set -euo pipefail
: "${OELLM_SCRATCH:?set the OELLM_* variables first}"
: "${MODEL:?set MODEL=<local model dir>}"
: "${VLLM_CONTAINER:?set VLLM_CONTAINER=<sif>}"
: "${VLLM_PY:=python3}"
: "${MAX_MODEL_LEN:=32768}"
: "${PORT:=18190}"
[[ -r $MODEL/config.json ]] || { echo "no model at $MODEL" >&2; exit 1; }
: "${CHAT_TEMPLATE:=}"
[[ -z $CHAT_TEMPLATE && -f $MODEL/chat_template.jinja ]] && CHAT_TEMPLATE=$MODEL/chat_template.jinja
EXTRA=(); [[ -n $CHAT_TEMPLATE ]] && EXTRA+=(--chat-template "$CHAT_TEMPLATE")
LOG=logs/${SLURM_JOB_ID:-manual}.vllm-smoke.server.log
# vllm-openai:v0.32.0-x86_64-cu129 ships a torchcodec built for CUDA 13 (libnvrtc.so.13); importing it raises OSError, which
# vLLM's optional-audio import does not catch. A stand-in package that raises ImportError disables audio input instead.
: "${VLLM_STUB_TORCHCODEC:=1}"
VLLM_PP=""
if [[ $VLLM_STUB_TORCHCODEC == 1 ]]; then
  VLLM_PP=$OELLM_SCRATCH/oellm-rl/containers/stubs; mkdir -p "$VLLM_PP/torchcodec"
  echo 'raise ImportError("torchcodec disabled: the container build needs CUDA 13 libraries (see leonardo_vllm_smoke.sbatch)")' \
    > "$VLLM_PP/torchcodec/__init__.py"
fi
# Leonardo's driver (R535, CUDA 12.1/12.2) is older than the container's CUDA 12.9. Precompiled kernels run on it, but code
# compiled at run time (FlashInfer, Triton PTX) may not. CUDA forward compatibility: the container's /usr/local/cuda/compat
# holds a newer user-mode driver that works with the R535 kernel module on data-centre GPUs (A100). --nv puts the host
# libcuda first on LD_LIBRARY_PATH, so the wrapper puts compat before it. VLLM_CUDA_COMPAT=0 turns it off.
# The FlashInfer sampler is the kernel that failed warm-up under the plain R535 driver; vLLM's PyTorch sampler replaces it.
: "${VLLM_CUDA_COMPAT:=1}" "${VLLM_USE_FLASHINFER_SAMPLER:=0}"
export VLLM_CUDA_COMPAT VLLM_USE_FLASHINFER_SAMPLER
COMPAT_WRAP=(sh -c 'if [ "$VLLM_CUDA_COMPAT" = 1 ] && [ -d /usr/local/cuda/compat ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH; fi; exec "$@"' sh)
RUN=(singularity exec --nv -B /leonardo_work,/leonardo_scratch --env "PYTHONPATH=$VLLM_PP" "$VLLM_CONTAINER" "${COMPAT_WRAP[@]}")

echo "=== 1. GPU and driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "=== 1b. CUDA driver API version seen inside the container (12020 = host R535 only, 12090 = forward compatibility active)"
"${RUN[@]}" sh -c 'ls -d /usr/local/cuda/compat 2>/dev/null || echo "no /usr/local/cuda/compat in the container"'
"${RUN[@]}" "$VLLM_PY" -c 'import ctypes; l = ctypes.CDLL("libcuda.so.1"); v = ctypes.c_int(); l.cuDriverGetVersion(ctypes.byref(v)); print("cuDriverGetVersion", v.value)'
echo "VLLM_CUDA_COMPAT=$VLLM_CUDA_COMPAT VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
echo "=== 2. vLLM and CUDA inside the container"
"${RUN[@]}" "$VLLM_PY" -c 'import vllm, transformers, torch; print("vllm", vllm.__version__, "transformers", transformers.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "gpu ok", torch.cuda.is_available())'
# The server imports much more than `import vllm` (multimodal, audio); fail here, quickly, if any of it does not load.
"${RUN[@]}" "$VLLM_PY" -c 'import vllm.entrypoints.openai.api_server; print("api server imports ok")'

echo "=== 3. start the server (log: $LOG)"
"${RUN[@]}" env VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=fork "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name policy --host 127.0.0.1 --port "$PORT" --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 --dtype bfloat16 --enable-prefix-caching --reasoning-parser qwen3 "${EXTRA[@]}" \
  > "$LOG" 2>&1 & pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $pid 2>/dev/null || { echo "vLLM exited early. Engine error (end of traceback):"
    grep "ERROR" "$LOG" | tail -15 | cut -c1-300; exit 1; }
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "vLLM not ready after 15 min:"; tail -40 "$LOG"; exit 1; }
echo "ready after ~$((i * 5)) s"

echo "=== 4. one chat request"
curl -sS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "policy", "max_tokens": 1024, "temperature": 0.6,
  "messages": [{"role": "user", "content": "List the files in the current directory with a shell command. Reply with the command only."}]
}' > reply.$$.json
"${RUN[@]}" "$VLLM_PY" - reply.$$.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if "choices" not in r:
    print("ERROR:", json.dumps(r)[:1000]); sys.exit(1)
m = r["choices"][0]["message"]
print("reasoning:", (m.get("reasoning_content") or m.get("reasoning") or "")[:600])
print("content:  ", (m.get("content") or "")[:600])
print("finish:   ", r["choices"][0].get("finish_reason"), "| tokens:", r.get("usage"))
PY
rm -f reply.$$.json
echo "SMOKE OK"
