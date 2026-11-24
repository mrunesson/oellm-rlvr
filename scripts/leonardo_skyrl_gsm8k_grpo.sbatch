#!/bin/bash
# SkyRL GRPO on GSM8K for Prelude-base (9B) on one Leonardo node (4 x A100 64 GB). Training and vLLM generation share
# the four GPUs (colocate_all): FSDP2 shards the policy, which moves to CPU while the four vLLM engines sample.
# Rewards: SkyRL's gsm8k env (1 if the first "#### <number>" equals the answer, else 0). No KL term, no reference model.
# Setup first: scripts/leonardo_skyrl_setup.sbatch. Then a 2-step smoke on the debug queue:
#   MAX_STEPS=2 TRAIN_BATCH=16 EVAL_BEFORE_TRAIN=false RUN_NAME=smoke \
#     sbatch -A $OELLM_ACCOUNT --qos=boost_qos_dbg --time=00:30:00 --export=ALL scripts/leonardo_skyrl_gsm8k_grpo.sbatch
# and the run (one epoch = 116 steps of 64 prompts x 8 samples; resubmit the same command to resume from the last
# checkpoint, since paths depend only on RUN_NAME):
#   sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_skyrl_gsm8k_grpo.sbatch
#SBATCH --job-name=skyrl-gsm8k
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
: "${ROOT:=$OELLM_SCRATCH/oellm-rl}"
: "${SKYRL_ROOT:=$ROOT/sources/SkyRL-v0.3.0}"
: "${VENV:=$ROOT/venvs/skyrl-v030}"
: "${MODEL:=$ROOT/models/prelude-base-reason}"
: "${DATA:=$ROOT/data/gsm8k-skyrl}"
: "${CONTAINER:=$ROOT/containers/vllm-openai-v0.32.0-x86_64-cu129.sif}"   # OS only (glibc 2.35), see the setup job
: "${RUN_NAME:=prelude-base-gsm8k-grpo}"
: "${RUN_ROOT:=$ROOT/outputs/skyrl/$RUN_NAME}"
: "${TRAIN_BATCH:=64}" "${N_SAMPLES:=8}" "${MINI_BATCH:=$TRAIN_BATCH}" "${MICRO_BATCH:=2}"
: "${MAX_STEPS:=}" "${EPOCHS:=1}" "${LR:=1.0e-6}" "${TEMPERATURE:=1.0}"
: "${MAX_PROMPT_LENGTH:=512}" "${MAX_GENERATE_LENGTH:=1024}"
: "${GPU_MEMORY_UTILIZATION:=0.6}" "${ENFORCE_EAGER:=false}" "${FLASH_ATTN:=true}" "${POLICY_CPU_OFFLOAD:=false}"
: "${EVAL_BEFORE_TRAIN:=true}" "${EVAL_INTERVAL:=10}" "${CKPT_INTERVAL:=20}" "${HF_SAVE_INTERVAL:=20}"
: "${LOGGER:=console}"
[[ -r $MODEL/config.json && -r $MODEL/SOURCE.json ]] || { echo "no staged model at $MODEL (run the setup job)" >&2; exit 1; }
[[ -r $DATA/train.parquet && -r $DATA/validation.parquet ]] || { echo "no GSM8K data at $DATA" >&2; exit 1; }
[[ -x $VENV/bin/python ]] || { echo "no venv at $VENV" >&2; exit 1; }
[[ -r $CONTAINER ]] || { echo "no container at $CONTAINER" >&2; exit 1; }
mkdir -p logs "$RUN_ROOT"/{checkpoints,exports,logs,ray}
cat "$MODEL/SOURCE.json"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Ray's Unix sockets need a short path: node-local, copied into the run directory at exit.
NODE_TMP=/tmp/oellm-skyrl-$SLURM_JOB_ID RAY_NODE_TMP=/tmp/oellm-ray-$SLURM_JOB_ID
mkdir -p "$NODE_TMP" "$RAY_NODE_TMP"
trap 'cp -a "$RAY_NODE_TMP"/. "$RUN_ROOT/ray/$SLURM_JOB_ID"/ 2>/dev/null || true; rm -rf "$NODE_TMP" "$RAY_NODE_TMP"' EXIT
CACHE=$ROOT/cache/skyrl-run
export TMPDIR=$NODE_TMP RAY_TMPDIR=$RAY_NODE_TMP
export HF_HOME=$ROOT/hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 WANDB_MODE=offline WANDB_DIR=$RUN_ROOT/logs TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor VLLM_CACHE_ROOT=$CACHE/vllm
export XDG_CACHE_HOME=$CACHE/xdg TENSORBOARD_DIR=$RUN_ROOT/tensorboard
# The FlashInfer sampler failed warm-up under Leonardo's R535 driver (leonardo_vllm_smoke.sbatch); use vLLM's own.
export VLLM_USE_FLASHINFER_SAMPLER=0
# CUDA forward compatibility (as in leonardo_vllm_smoke.sbatch): the container's /usr/local/cuda/compat user-mode
# driver in front of the host R535 libcuda, for code compiled at run time. VLLM_CUDA_COMPAT=0 turns it off.
: "${VLLM_CUDA_COMPAT:=1}"
export VLLM_CUDA_COMPAT
COMPAT_WRAP=(sh -c 'if [ "$VLLM_CUDA_COMPAT" = 1 ] && [ -d /usr/local/cuda/compat ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH; fi; exec "$@"' sh)
# The image's own Python environment must not leak into the venv.
IN=(singularity exec --nv -B /leonardo_work,/leonardo_scratch "$CONTAINER" env -u PYTHONPATH -u VIRTUAL_ENV
  "${COMPAT_WRAP[@]}")
STEPS=(); [[ -n $MAX_STEPS ]] && STEPS=(trainer.max_training_steps="$MAX_STEPS")
"${IN[@]}" "$VENV/bin/python" -c 'import ctypes, torch; l = ctypes.CDLL("libcuda.so.1"); v = ctypes.c_int()
l.cuDriverGetVersion(ctypes.byref(v)); print("cuDriverGetVersion", v.value, "torch", torch.__version__, "gpus", torch.cuda.device_count())'

cd "$SKYRL_ROOT"
srun --ntasks=1 "${IN[@]}" "$VENV/bin/python" -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA/train.parquet']" data.val_data="['$DATA/validation.parquet']" \
  trainer.policy.model.path="$MODEL" trainer.strategy=fsdp \
  trainer.policy.fsdp_config.cpu_offload="$POLICY_CPU_OFFLOAD" \
  trainer.placement.colocate_all=true trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=4 trainer.placement.ref_num_gpus_per_node=4 \
  generator.inference_engine.backend=vllm generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=4 generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl generator.inference_engine.enforce_eager="$ENFORCE_EAGER" \
  generator.inference_engine.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  generator.batched=true generator.n_samples_per_prompt="$N_SAMPLES" \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.sampling_params.temperature="$TEMPERATURE" \
  environment.env_class=gsm8k \
  trainer.algorithm.advantage_estimator=grpo trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.epochs="$EPOCHS" "${STEPS[@]}" trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="$TRAIN_BATCH" trainer.policy_mini_batch_size="$MINI_BATCH" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_BATCH" trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH" \
  trainer.max_prompt_length="$MAX_PROMPT_LENGTH" trainer.flash_attn="$FLASH_ATTN" \
  trainer.gradient_checkpointing=true trainer.policy.optimizer_config.lr="$LR" \
  trainer.eval_before_train="$EVAL_BEFORE_TRAIN" trainer.eval_interval="$EVAL_INTERVAL" trainer.eval_batch_size=1319 \
  trainer.ckpt_interval="$CKPT_INTERVAL" trainer.hf_save_interval="$HF_SAVE_INTERVAL" trainer.max_ckpts_to_keep=2 \
  trainer.resume_mode=latest trainer.ckpt_path="$RUN_ROOT/checkpoints" trainer.export_path="$RUN_ROOT/exports" \
  trainer.log_path="$RUN_ROOT/logs" trainer.logger="$LOGGER" \
  trainer.project_name=oellm-skyrl-gsm8k trainer.run_name="$RUN_NAME"

echo "skyrl_run=$RUN_ROOT"
