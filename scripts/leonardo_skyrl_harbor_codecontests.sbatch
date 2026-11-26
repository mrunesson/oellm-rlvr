#!/bin/bash
# SkyRL + Harbor agentic GRPO on open-thoughts/CodeContests from Prelude-SFT-Propella-Excellent-v0.2 (9B, 262k
# positions, 5daebfa) on one Leonardo node (4 x A100 64 GB),
# with every Harbor sandbox (agent commands and the verifier) running on Modal. The agent is Harbor's terminus-2; it
# talks to SkyRL's vLLM engines on this node, and its terminal runs in a Modal sandbox built from the task's
# Dockerfile. Reward: 1 if /app/solution.py passes the task's tests, else 0. Recipe as upstream run_codecontest.sh
# (step-wise trajectories, token-mean loss, no std normalisation, token TIS, overlong filtering), sized for one node.
#
# The GPU node has no internet. Modal (only Modal) goes through a loopback relay on this node to the proxy chain on
# lrd_all_serial (scripts/leonardo_modal_proxy.sbatch), which must be running before this job starts:
#   sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_modal_proxy.sbatch
# Setup first: scripts/leonardo_skyrl_harbor_setup.sbatch. Then a 2-step smoke on the debug queue:
#   MAX_STEPS=2 TRAIN_BATCH=4 N_SAMPLES=4 RUN_NAME=cc-smoke \
#     sbatch -A $OELLM_ACCOUNT --qos=boost_qos_dbg --time=00:30:00 --export=ALL scripts/leonardo_skyrl_harbor_codecontests.sbatch
# and the run (resubmit the same command to resume from the last checkpoint; paths depend only on RUN_NAME):
#   sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_skyrl_harbor_codecontests.sbatch
#SBATCH --job-name=skyrl-cc-modal
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.%x.out
set -euo pipefail
ulimit -c 0
: "${OELLM_SCRATCH:?set the OELLM_* variables first}"
: "${CONTROL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${ROOT:=$OELLM_SCRATCH/oellm-rl}"
: "${SKYRL_ROOT:=$ROOT/sources/SkyRL-v0.3.0}"
: "${VENV:=$ROOT/venvs/skyrl-v030-harbor}"
: "${MODEL:=$ROOT/models/prelude-sft-propella-v0.2}"
: "${DATA:=$ROOT/data/harbor/CodeContests}"
: "${EVAL_DATA:=}"                       # optional Harbor task directory for evaluation; empty = no evaluation
: "${CONTAINER:=$ROOT/containers/vllm-openai-v0.32.0-x86_64-cu129.sif}"   # OS only (glibc 2.35)
: "${PROXY_DIR:=$ROOT/proxy-modal}"
: "${RUN_NAME:=prelude-sft-codecontests-modal}"
: "${RUN_ROOT:=$ROOT/outputs/skyrl/$RUN_NAME}"
# Batch: TRAIN_BATCH tasks x N_SAMPLES trials per step, each its own Modal sandbox.
: "${TRAIN_BATCH:=32}" "${N_SAMPLES:=8}" "${MINI_BATCH:=$TRAIN_BATCH}" "${MAX_STEPS:=}" "${EPOCHS:=1}"
: "${LR:=1.0e-6}" "${TEMPERATURE:=1.0}"
# The model takes 262k positions; the limit is learner memory on 4 x 64 GB (upstream uses 32768 on 8 GPUs). The
# full-vocabulary logits alone are 262145 x MAX_MODEL_LEN values per sequence.
: "${MAX_MODEL_LEN:=16384}" "${MAX_TOKENS_PER_TURN:=4096}" "${MAX_TURNS:=32}" "${AGENT_TIMEOUT:=1200}"
: "${MAX_CONCURRENCY:=128}" "${TRAJECTORIES_PER_SECOND:=5}"
# Reasoning mode of the SFT model's chat template: think (the template appends "/think" to the system prompt, and the
# model reasons in <think>...</think> before each answer), no_think, or auto (no instruction; the model chooses).
# Applied to every request through vLLM's default_chat_template_kwargs. With think, vLLM's qwen3 reasoning parser (<think>
# and </think> are single tokens, ids 18/17) moves the reasoning into reasoning_content, so terminus-2 parses its JSON
# from the answer alone: its parser takes the first top-level "{" of the reply, which could be in the reasoning. The
# token ids SkyRL trains on are vLLM's raw output, reasoning included. A turn cut off before </think> has no answer.
: "${THINKING:=think}"
: "${SANDBOX_CPUS:=1}" "${SANDBOX_MEMORY_MB:=2048}" "${SANDBOX_STORAGE_MB:=4096}"
: "${SANDBOX_TIMEOUT:=3600}" "${SANDBOX_IDLE_TIMEOUT:=1200}"   # so sandboxes die if this job does
: "${GPU_MEMORY_UTILIZATION:=0.6}" "${ENFORCE_EAGER:=false}" "${FLASH_ATTN:=true}" "${POLICY_CPU_OFFLOAD:=false}"
: "${CKPT_INTERVAL:=10}" "${HF_SAVE_INTERVAL:=20}" "${EVAL_INTERVAL:=20}" "${LOGGER:=console}"
: "${MODAL_PREFLIGHT:=1}"
# Modal 1.5.1 (SkyRL's pin) knows image builders up to 2025.06; a workspace set to a newer one (2026.10 seen on
# 2026-11-26) makes every image build fail with VersionError. This asks for 2025.06 for this job's images only; the
# workspace setting is unchanged.
: "${MODAL_IMAGE_BUILDER_VERSION:=2025.06}"
export MODAL_IMAGE_BUILDER_VERSION
case $THINKING in
  think) THINKING_ARGS=("generator.inference_engine.engine_init_kwargs.default_chat_template_kwargs.thinking_instruction=think"
                        generator.inference_engine.engine_init_kwargs.reasoning_parser=qwen3) ;;
  no_think) THINKING_ARGS=("generator.inference_engine.engine_init_kwargs.default_chat_template_kwargs.thinking_instruction=no_think") ;;
  auto) THINKING_ARGS=() ;;
  *) echo "THINKING must be think, no_think or auto, not '$THINKING'" >&2; exit 1 ;;
esac
[[ -r $MODEL/config.json && -r $MODEL/SOURCE.json ]] || { echo "no staged model at $MODEL (run the setup job)" >&2; exit 1; }
[[ -r $DATA/code_contests-0000/instruction.md ]] || { echo "no CodeContests tasks in $DATA (run the setup job)" >&2; exit 1; }
[[ -x $VENV/bin/python ]] || { echo "no venv at $VENV (run the setup job)" >&2; exit 1; }
[[ -r $CONTAINER ]] || { echo "no container at $CONTAINER" >&2; exit 1; }
[[ -r $HOME/.modal.toml || -n ${MODAL_TOKEN_ID:-} ]] || { echo "no Modal credentials (~/.modal.toml or MODAL_TOKEN_ID)" >&2; exit 1; }
# With --dependency=after:<proxy job> this job can start a few seconds before the proxy has written its address.
for _ in $(seq "${PROXY_WAIT:=600}"); do [[ -s $PROXY_DIR/proxy.host && -r $PROXY_DIR/proxy.token ]] && break; sleep 1; done
[[ -s $PROXY_DIR/proxy.host && -r $PROXY_DIR/proxy.token ]] || {
  echo "no proxy in $PROXY_DIR after ${PROXY_WAIT}s: sbatch scripts/leonardo_modal_proxy.sbatch first" >&2; exit 1; }
mkdir -p logs "$RUN_ROOT"/{checkpoints,exports,logs,ray,trials,manifests}
cat "$MODEL/SOURCE.json"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

NODE_TMP=/tmp/oellm-skyrl-$SLURM_JOB_ID RAY_NODE_TMP=/tmp/oellm-ray-$SLURM_JOB_ID
mkdir -p "$NODE_TMP" "$RAY_NODE_TMP"
RELAY_PID=
cleanup() {
  [[ -n $RELAY_PID ]] && kill "$RELAY_PID" 2>/dev/null || true
  cp -a "$RAY_NODE_TMP"/. "$RUN_ROOT/ray/$SLURM_JOB_ID"/ 2>/dev/null || true
  rm -rf "$NODE_TMP" "$RAY_NODE_TMP"
}
trap cleanup EXIT
CACHE=$ROOT/cache/skyrl-run
export TMPDIR=$NODE_TMP RAY_TMPDIR=$RAY_NODE_TMP
export HF_HOME=$ROOT/hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 WANDB_MODE=offline WANDB_DIR=$RUN_ROOT/logs TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor VLLM_CACHE_ROOT=$CACHE/vllm
export XDG_CACHE_HOME=$CACHE/xdg TENSORBOARD_DIR=$RUN_ROOT/tensorboard
export LITELLM_LOCAL_MODEL_COST_MAP=True   # LiteLLM would otherwise try to download its cost map
export VLLM_USE_FLASHINFER_SAMPLER=0       # failed under R535 (see leonardo_skyrl_gsm8k_grpo.sbatch)
: "${VLLM_CUDA_COMPAT:=1}"
export VLLM_CUDA_COMPAT
COMPAT_WRAP=(sh -c 'if [ "$VLLM_CUDA_COMPAT" = 1 ] && [ -d /usr/local/cuda/compat ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH; fi; exec "$@"' sh)
# The image's own Python environment must not leak into the venv; SkyRL's examples and oellm_rlvr are importable.
IN=(singularity exec --nv -B /leonardo_work,/leonardo_scratch "$CONTAINER" env -u VIRTUAL_ENV
  PYTHONPATH="$SKYRL_ROOT:$CONTROL_ROOT/src" "${COMPAT_WRAP[@]}")

# Loopback relay to the current proxy (re-read on every connection, so the proxy chain can rotate underneath).
RELAY_PORT_FILE=$NODE_TMP/relay.port
"${IN[@]}" "$VENV/bin/python" -I "$CONTROL_ROOT/scripts/laion_rootless/local_relay.py" \
  "$PROXY_DIR/proxy.host" "$PROXY_DIR/proxy.token" "$RELAY_PORT_FILE" 172800 > "$RUN_ROOT/logs/relay-$SLURM_JOB_ID.log" 2>&1 &
RELAY_PID=$!
for _ in $(seq 60); do [[ -s $RELAY_PORT_FILE ]] && break; sleep 1; done
[[ -s $RELAY_PORT_FILE ]] || { echo "relay did not start:" >&2; cat "$RUN_ROOT/logs/relay-$SLURM_JOB_ID.log" >&2; exit 1; }
export OELLM_MODAL_PROXY=http://127.0.0.1:$(cat "$RELAY_PORT_FILE")
echo "modal proxy: $OELLM_MODAL_PROXY -> $(cat "$PROXY_DIR/proxy.host")"

git -C "$CONTROL_ROOT" rev-parse HEAD > "$RUN_ROOT/manifests/oellm-rlvr.commit" 2>/dev/null || true
git -C "$SKYRL_ROOT" rev-parse HEAD > "$RUN_ROOT/manifests/skyrl.commit"
cp "$MODEL/SOURCE.json" "$RUN_ROOT/manifests/model-SOURCE.json"
echo "$THINKING" > "$RUN_ROOT/manifests/thinking-$SLURM_JOB_ID.txt"
cat "$(dirname "$DATA")/CodeContests.SOURCE" > "$RUN_ROOT/manifests/data-SOURCE.txt" 2>/dev/null || true
"${IN[@]}" "$VENV/bin/python" -c 'import ctypes, torch, harbor, modal; l = ctypes.CDLL("libcuda.so.1"); v = ctypes.c_int()
l.cuDriverGetVersion(ctypes.byref(v)); print("cuDriverGetVersion", v.value, "torch", torch.__version__, "gpus",
torch.cuda.device_count(), "modal", modal.__version__)'

if [[ $MODAL_PREFLIGHT == 1 ]]; then
  echo "=== Modal preflight through the relay"
  "${IN[@]}" "$VENV/bin/python" "$CONTROL_ROOT/scripts/probe_modal_sandbox.py" \
    --task-dir "$DATA/code_contests-0000" --output "$RUN_ROOT/manifests/modal-preflight-$SLURM_JOB_ID.json"
fi

STEPS=(); [[ -n $MAX_STEPS ]] && STEPS=(trainer.max_training_steps="$MAX_STEPS")
if [[ -n $EVAL_DATA ]]; then
  EVAL=(data.val_data="['$EVAL_DATA']" trainer.eval_interval="$EVAL_INTERVAL" trainer.eval_batch_size=128
        generator.eval_n_samples_per_prompt=2)
else
  EVAL=(data.val_data="[]" trainer.eval_interval=-1)
fi

cd "$SKYRL_ROOT"
srun --ntasks=1 "${IN[@]}" "$VENV/bin/python" "$CONTROL_ROOT/scripts/run_harbor_train.py" \
  data.train_data="['$DATA']" "${EVAL[@]}" \
  harbor_trial_config.trials_dir="$RUN_ROOT/trials" \
  harbor_trial_config.agent.name=terminus-2 \
  harbor_trial_config.agent.override_timeout_sec="$AGENT_TIMEOUT" \
  harbor_trial_config.agent.kwargs.max_turns="$MAX_TURNS" \
  harbor_trial_config.agent.kwargs.temperature="$TEMPERATURE" \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_TOKENS_PER_TURN" \
  harbor_trial_config.agent.kwargs.llm_kwargs.max_tokens="$MAX_TOKENS_PER_TURN" \
  harbor_trial_config.environment.type=modal \
  harbor_trial_config.environment.override_cpus="$SANDBOX_CPUS" \
  harbor_trial_config.environment.override_memory_mb="$SANDBOX_MEMORY_MB" \
  harbor_trial_config.environment.override_storage_mb="$SANDBOX_STORAGE_MB" \
  harbor_trial_config.environment.kwargs.sandbox_timeout_secs="$SANDBOX_TIMEOUT" \
  harbor_trial_config.environment.kwargs.sandbox_idle_timeout_secs="$SANDBOX_IDLE_TIMEOUT" \
  trainer.policy.model.path="$MODEL" trainer.strategy=fsdp \
  trainer.policy.fsdp_config.cpu_offload="$POLICY_CPU_OFFLOAD" \
  trainer.placement.colocate_all=true trainer.placement.policy_num_nodes=1 trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=4 trainer.placement.ref_num_gpus_per_node=4 \
  generator.inference_engine.served_model_name=prelude-sft \
  generator.inference_engine.backend=vllm generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=4 generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl generator.inference_engine.enforce_eager="$ENFORCE_EAGER" \
  generator.inference_engine.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false "${THINKING_ARGS[@]}" \
  generator.batched=false generator.n_samples_per_prompt="$N_SAMPLES" \
  generator.step_wise_trajectories=true generator.merge_stepwise_output=true \
  generator.apply_overlong_filtering=true \
  generator.sampling_params.max_generate_length="$MAX_TOKENS_PER_TURN" \
  generator.rate_limit.enabled=true generator.rate_limit.trajectories_per_second="$TRAJECTORIES_PER_SECOND" \
  generator.rate_limit.max_concurrency="$MAX_CONCURRENCY" \
  trainer.algorithm.advantage_estimator=grpo trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.grpo_norm_by_std=false trainer.algorithm.use_kl_loss=false trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.off_policy_correction.tis_ratio_type=token \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0 \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  trainer.epochs="$EPOCHS" "${STEPS[@]}" trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="$TRAIN_BATCH" trainer.policy_mini_batch_size="$MINI_BATCH" \
  trainer.micro_forward_batch_size_per_gpu=1 trainer.micro_train_batch_size_per_gpu=1 \
  trainer.flash_attn="$FLASH_ATTN" \
  trainer.gradient_checkpointing=true trainer.policy.optimizer_config.lr="$LR" \
  trainer.eval_before_train=false \
  trainer.ckpt_interval="$CKPT_INTERVAL" trainer.hf_save_interval="$HF_SAVE_INTERVAL" trainer.max_ckpts_to_keep=2 \
  trainer.resume_mode=latest trainer.ckpt_path="$RUN_ROOT/checkpoints" trainer.export_path="$RUN_ROOT/exports" \
  trainer.log_path="$RUN_ROOT/logs" trainer.logger="$LOGGER" \
  trainer.project_name=oellm-harbor-codecontests trainer.run_name="$RUN_NAME" \
  2>&1 | tee -a "$RUN_ROOT/logs/training-$SLURM_JOB_ID.log"

echo "skyrl_run=$RUN_ROOT"
