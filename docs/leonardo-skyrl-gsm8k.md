# SkyRL GRPO on GSM8K for Prelude-base on Leonardo

Status: **full run completed (2026-11-21, job 64837039): GSM8K test 0.343 -> 0.933 in one epoch (116 steps,
4 h 43 min).** This trains `openeurollm/prelude-base` (9B, Qwen3 architecture) with SkyRL's GRPO on GSM8K, on one
Leonardo booster node (4 x A100 64 GB).

Greedy `eval/all/avg_score` on the 1,319 test questions:

| Step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 116 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Score | 0.343 | 0.886 | 0.911 | 0.918 | 0.923 | 0.923 | 0.927 | 0.928 | 0.930 | 0.923 | 0.929 | 0.929 | 0.933 |

Most of the gain is the answer format. At step 0 only 552 of 1,319 responses contain `####`, so 767 of the 866
wrong answers give no answer at all, and 107 run to the length limit, often repeating the question. Of the 552
that do answer, 453 are right (82%). From step 10 on every response has `####`, and nothing runs to the length
limit by step 116. Accuracy rises from 82% (of the answered subset at step 0) to 93% (of everything at step 116),
so arithmetic also improves, but by far less than the headline jump. Every response ends with `<eos>` (id 2),
never `<|im_end|>`: both are stop tokens and the reward ignores the ending, so GRPO never taught the turn
terminator. Stacks that stop only on `<|im_end|>` will run on; a later run could add a reward for it or stop on
id 4 alone. Training reward ended at 0.92-0.96
(pass@8 0.98-1.0), so most prompts no longer give a GRPO signal by the end of the epoch. Whether the test set was
in the pretraining data has not been checked. HF exports are at steps 20, 40, 60, 80, 100 and 116 in
`outputs/skyrl/prelude-base-gsm8k-grpo/exports/`; restart checkpoints at 100 and 116.

The smoke run (job 64832231, 2 steps of 16 prompts x 8 samples) took ~45 s per step and scored 0.401 after step 2;
responses stop on their own (mean 368 tokens, min 70). `policy/response_length` in the training metrics is the
padded batch length, not the mean.

## What is pinned

| Part | Version | Why |
|---|---|---|
| SkyRL | `skyrl-v0.3.0` (`f5bc3b7`), installed from its `uv.lock` with `--extra fsdp` | torch 2.11 cu128 + vLLM 0.23 run on Leonardo's R535 driver; v0.4+ is CUDA 13 only. Same commit as the LUMI qualification. |
| Weights | `openeurollm/prelude-base@7ab2bc6` | the final annealed base checkpoint |
| Tokenizer | `openeurollm/tokenizer-256k-reason@baaafeb` | puts `<\|im_start\|>`=3, `<\|im_end\|>`=4, `</think>`=17, `<think>`=18 and the tool tokens in reserved slots. The vocabulary and merges are otherwise the base tokenizer's: all 7,473 GSM8K train rows tokenize identically. (The repo README lists the think tokens as 15/16; the file says 17/18.) |
| OS | the vLLM image `vllm-openai-v0.32.0-x86_64-cu129.sif` (Ubuntu, glibc 2.35), for its OS only | SkyRL v0.3.0 sends all rollouts through `vllm-router`, whose only wheel needs glibc 2.35; Leonardo (RHEL 8) has 2.28. The venv is built and run inside the image, which also brings the CUDA forward-compatibility driver for R535. |
| Chat template | `qwen3.jinja` from `OpenEuroLLM/post-training@9ac9328` (sha256 checked) | the template the Prelude SFT recipe uses |

`scripts/stage_chat_model.py` combines these into `$OELLM_SCRATCH/oellm-rl/models/prelude-base-reason` and, as
post-training's `build_tokenizer`/`align_generation_eos` do, sets the tokenizer's eos to `<|im_end|>` and
`generation_config.eos_token_id` to `[4, 2]`, so vLLM stops at the end of a turn. `SOURCE.json` there records the
revisions. The template adds no `<bos>`, matching post-training.

## Run it

```bash
source leonardo.env && cd $OELLM_WORK/oellm-rlvr && mkdir -p logs
# 0. once, if $OELLM_SCRATCH/oellm-rl/containers/vllm-openai-v0.32.0-x86_64-cu129.sif is missing (~1 h)
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_vllm_container_pull.sbatch
# 1. once: SkyRL venv, staged model, GSM8K parquet (lrd_all_serial, internet; needs uv in ~/.local/bin)
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_skyrl_setup.sbatch
# 2. two-step smoke on the debug queue
MAX_STEPS=2 TRAIN_BATCH=16 EVAL_BEFORE_TRAIN=false RUN_NAME=smoke \
  sbatch -A $OELLM_ACCOUNT --qos=boost_qos_dbg --time=00:30:00 --export=ALL scripts/leonardo_skyrl_gsm8k_grpo.sbatch
# 3. the run: one epoch, 116 steps of 64 prompts x 8 samples; resubmit the same command to resume
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_skyrl_gsm8k_grpo.sbatch
```

Outputs go to `$OELLM_SCRATCH/oellm-rl/outputs/skyrl/<RUN_NAME>` (`checkpoints/` for restart state, `exports/` for
HF checkpoints, `ray/` for worker logs). Every setting in the job is an environment variable (see the top of
`leonardo_skyrl_gsm8k_grpo.sbatch`).

## Recipe

Colocated FSDP2 policy and four vLLM engines (TP 1, 60% GPU memory), GRPO advantages, no KL loss and no reference
model, lr 1e-6, temperature 1.0, 512 prompt / 1024 response tokens. The reward is SkyRL's `gsm8k` env: 1 if the
first `#### <number>` in the response equals the answer, else 0. The prompt is SkyRL's: the question followed by
`Let's think step by step and output the final answer after "####".` Greedy evaluation runs on the 1,319 test
questions before training and every 10 steps; checkpoints and HF exports are written every 20 steps.

## Things to check in the first runs

- **Does the base model end its turn?** It has not been trained on `<|im_start|>`/`<|im_end|>`. If the step-0
  evaluation shows most responses running to 1,024 tokens, or a reward near 0 on every sample (no GRPO signal), the
  prompt needs work (for example few-shot examples) before a long run.
- **Driver compatibility.** The job puts the container's CUDA forward-compatibility driver in front of the host R535
  one and prints `cuDriverGetVersion` (12090 means it is active). The FlashInfer sampler is disabled because it
  failed under R535 before. If a kernel fails with an unsupported PTX or toolchain error, try `VLLM_CUDA_COMPAT=0`.
- **flash-attn.** The setup job prints an import check. If `flash_attn` fails to import, run with `FLASH_ATTN=false`.
- **Memory.** If the learner runs out of memory, set `POLICY_CPU_OFFLOAD=true` or `MICRO_BATCH=1`. If vLLM does,
  lower `GPU_MEMORY_UTILIZATION`.
