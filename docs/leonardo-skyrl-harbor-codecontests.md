# SkyRL + Harbor agentic GRPO on CodeContests for Prelude SFT, Leonardo GPUs + Modal sandboxes

Status: **no completed training step yet.** On 2026-11-26 the smoke runs with `prelude-base@4eafcf9` got the whole
path working (proxy chain, relay, Modal sandboxes through the proxy, vLLM and the trainer on the R535 driver, weight
sync, agent turns). None reached a verifier: of the 32 trials in `cc-smoke2`, 16 failed on the Modal workspace's
spend limit (`ResourceExhaustedError`), 10 on sandboxes Modal stopped when that limit was hit (`NotFoundError`), and 6
filled the 16k context (`ContextLengthExceededError`). Startup took 16 of the debug queue's 30 minutes, so run the next
smoke on the normal queue with `--time=01:00:00`. The model then changed to the SFT checkpoint below, which has not
run yet.

This trains `openeurollm/Prelude-SFT-Propella-Excellent-v0.2` at `5daebfa` (9B, Qwen3 architecture, 262,144
positions; `prelude-anneal300b-256k-rope20m-cont@47a42eb` fine-tuned on `Propella-Excellent-SFT` with `/think` and
`/no_think` conditioning), with its own tokenizer and chat template, with SkyRL's GRPO on
[`open-thoughts/CodeContests`](https://huggingface.co/datasets/open-thoughts/CodeContests) (9,644 Harbor tasks). The
learner and vLLM run on one Leonardo booster node (4 x A100 64 GB). Every Harbor sandbox, both the agent's terminal and
the verifier, runs on [Modal](https://modal.com). The agent is Harbor's `terminus-2`: each turn the model answers with
shell keystrokes, Harbor runs them in the sandbox and returns the terminal output. A trial scores 1 if
`/app/solution.py` passes the task's tests, otherwise 0.

## How the pieces connect

```
 boost_usr_prod GPU node (no internet)                         lrd_all_serial (internet, 4 h jobs)
 ┌──────────────────────────────────────────────┐             ┌──────────────────────────────┐
 │ SkyRL FSDP2 learner + 4 vLLM engines         │             │ laion_proxy.py (CONNECT,     │
 │   ^ LiteLLM (localhost, no proxy)            │             │ token) - modal-proxy chain   │──> api.modal.com
 │ Harbor trials (terminus-2) -> Modal client ──┼─> relay ───>│ each job starts its          │    sandboxes
 │   OELLM_MODAL_PROXY=http://127.0.0.1:<port>  │  127.0.0.1  │ successor 30 min early       │
 └──────────────────────────────────────────────┘             └──────────────────────────────┘
```

- **Only Modal uses the proxy.** Modal 1.5.1 can tunnel its gRPC and blob traffic through an HTTP CONNECT proxy, but
  it reads `HTTPS_PROXY`, which Ray, vLLM, the vLLM router and LiteLLM would also follow for their node-local traffic.
  `oellm_rlvr.modal_proxy` hands the URL in `OELLM_MODAL_PROXY` to Modal alone; `scripts/run_harbor_train.py`
  installs it in the Ray worker that runs the Harbor trials. It needs Modal's `api-proxy-support` extra
  (`python-socks[asyncio]`, `aiohttp-socks`), which the setup job adds to the venv; it is not in SkyRL's lockfile.
- **The proxy renews itself.** `lrd_all_serial` jobs last at most 4 h, so `leonardo_modal_proxy.sbatch` submits its
  successor with `--begin` set 30 minutes before it exits. The successor writes its address to the same
  `proxy.host`, and the relay on the GPU node (`local_relay.py`) re-reads that file on every new connection. Modal
  calls that were open through the old proxy when it exits fail. SkyRL's Harbor generator runs a trial that ends
  without a verifier result once more (`MAX_NUM_RETRIES_PER_TRIAL = 2`), and a trial that fails both times is
  loss-masked, not scored 0. Expect a few masked trials
  every 3.5 h (`generate/num_masked_instances`).
- The chain stops itself when no `skyrl-cc-modal` job is pending or running, or when `$PROXY_DIR/STOP` exists.
- `lrd_all_serial` limits per user (QOS `lrd_all_serial`, checked 2026-11-26): 2 running and 10 submitted jobs,
  8 CPUs and 30.8 GB of memory in total, 4 h. Two overlapping proxies (2 CPUs, 4 GB each) fit. A setup job (30 GB)
  does not fit beside a running proxy and waits until it ends. If the successor cannot be submitted, the proxy keeps
  serving until its own end and logs a warning.

## What is pinned

| Part | Version | Note |
|---|---|---|
| SkyRL | `skyrl-v0.3.0` (`f5bc3b7`), `--extra fsdp --extra harbor` from its `uv.lock` | same commit as the GSM8K run; separate venv `venvs/skyrl-v030-harbor` |
| Harbor | `3de07a0` (0.13.1) | the commit SkyRL's lockfile pins. The vLLM image has no `git`, so the setup job clones it with the host's git into `sources/harbor-3de07a0`, runs `uv sync --no-install-package harbor` (still installs Harbor's dependencies, needs no git), and installs Harbor from the clone with `--no-deps` |
| Modal client | 1.5.1 + `python-socks`, `aiohttp-socks` | the last two are unpinned (`>=2.5`, `>=0.8`, Modal's own extra) |
| Modal image builder | `2025.06` (`MODAL_IMAGE_BUILDER_VERSION`, set by the jobs) | the newest 1.5.1 supports. The workspace default (`2026.10` on 2026-11-26) fails with `VersionError` (setup job 65280871). The variable applies to this job's image builds only; the workspace setting is unchanged |
| Weights | `Prelude-SFT-Propella-Excellent-v0.2@5daebfa` | one 18 GB `model.safetensors`; `max_position_embeddings` 262144, `rope_theta` 2e7, vocabulary 262,145 rows. Staged to `models/prelude-sft-propella-v0.2`; the earlier `models/prelude-base-long-reason` is left alone |
| Tokenizer | the model repo's own, same revision | 262,145 tokens; `<\|im_start\|>`/`<\|im_end\|>` at ids 3/4, eos `<\|im_end\|>`, pad 262144 |
| Chat template | the model's own `chat_template.jinja` at `5daebfa` (sha256 `826f27fc…`) | post-training's Qwen3 template with `thinking_instruction` (`think`/`no_think`, appended to the system prompt as `/think`/`/no_think`) and `{% generation %}` markers. Training sets `thinking_instruction=think` (see Recipe). It raises an error on a finished assistant turn without reasoning when rendered without a generation prompt; vLLM always renders with one, and staging's probe includes a `<think>` block |
| Generation eos | `[4]` (`<\|im_end\|>`), written by `stage_chat_model.py` | the repo's `generation_config.json` lists `[4, 4, 151645]`; 151645 is the text piece `▁Уточ` (a Qwen id), which would stop generation mid-text, so staging drops it |
| Dataset | `open-thoughts/CodeContests@11f66f5` | 9,644 tasks, all with the same Dockerfile (Ubuntu 24.04 + python3), so Modal builds one image |
| OS | `vllm-openai-v0.32.0-x86_64-cu129.sif` | for glibc 2.35, as in the GSM8K run |

## Run it

All commands on a Leonardo login node, from `$OELLM_WORK/oellm-rlvr` with the `OELLM_*` variables set.

```bash
cd $OELLM_WORK/oellm-rlvr && mkdir -p logs
# 0. Modal credentials, once: write ~/.modal.toml yourself (keep it private), e.g.
#      printf '[default]\ntoken_id = "%s"\ntoken_secret = "%s"\nactive = true\n' <id> <secret> > ~/.modal.toml
#      chmod 600 ~/.modal.toml
# 1. once: harbor venv, the SFT model with its own tokenizer and template, CodeContests tasks, and a Modal probe that
#    also builds the task image on Modal (lrd_all_serial; needs uv in ~/.local/bin and the vLLM image)
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_skyrl_harbor_setup.sbatch
# 2. start the proxy chain (keeps itself alive while a skyrl-cc-modal job is queued or running)
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_modal_proxy.sbatch
# 3. two-step smoke on the debug queue (4 tasks x 4 trials)
MAX_STEPS=2 TRAIN_BATCH=4 N_SAMPLES=4 RUN_NAME=cc-smoke \
  sbatch -A $OELLM_ACCOUNT --qos=boost_qos_dbg --time=00:30:00 --export=ALL scripts/leonardo_skyrl_harbor_codecontests.sbatch
# 4. the run: 32 tasks x 8 trials per step, one epoch = 301 steps; resubmit the same command to resume
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_skyrl_harbor_codecontests.sbatch
# stop the proxy chain when done
touch $OELLM_SCRATCH/oellm-rl/proxy-modal/STOP; scancel -n modal-proxy
```

The training job first starts the relay, then runs `probe_modal_sandbox.py` through it (one sandbox, `python3 --version`,
DNS from inside the sandbox) and stops if that fails, before any GPU work. Outputs go to
`$OELLM_SCRATCH/oellm-rl/outputs/skyrl/<RUN_NAME>`: `logs/training-<job>.log`, `trials/` (one Harbor directory per
trial with the agent transcript, verifier output and `result.json`), `checkpoints/`, `exports/` (HF checkpoints),
`manifests/` (commits, model and data sources, the Modal preflight). Every setting is an environment variable at the top
of `leonardo_skyrl_harbor_codecontests.sbatch`.

## Recipe

As upstream `examples/train_integrations/harbor/run_codecontest.sh`, sized for one node: colocated FSDP2 policy and
four vLLM engines (TP 1, 60% GPU memory), GRPO without std normalisation, token-mean loss, no KL, token-level TIS
(cap 2), overlong filtering, step-wise trajectories merged where the prefix matches, lr 1e-6, temperature 1.0.
Reasoning is on: `THINKING=think` (default) passes `thinking_instruction=think` to every request through vLLM's
`default_chat_template_kwargs`, so the system prompt ends in `/think` and the model reasons in `<think>…</think>`
before each answer (`THINKING=no_think` or `auto` to change it; recorded in `manifests/thinking-<job>.txt`). With
`think`, vLLM's `qwen3` reasoning parser moves the reasoning into `reasoning_content`, because terminus-2's JSON
parser takes the first top-level `{` of the reply, which could be inside the reasoning. SkyRL still trains on vLLM's
raw output tokens, reasoning included.
Terminus-2 gets up to 32 turns of at most 4,096 tokens each, within a 16,384-token context, and 1,200 s per trial.
Each sandbox gets 1 CPU, 2 GB memory and 4 GB disk, lives at most 1 h, and stops after 20 idle minutes. Up to 128
trials run at once, at most 5 new trials per second. Checkpoints every 10 steps, HF exports every 20. No evaluation by
default; set `EVAL_DATA` to a Harbor task directory to add one.

Differences from upstream: Qwen3-8B on 8 GPUs with a 32k context there; Prelude SFT on 4 GPUs with 16k here. The
chat template is the model's own, not SkyRL's `qwen3_acc_thinking`. It drops `<think>` blocks from earlier assistant
turns, so when the model thinks, a turn's prompt no longer starts with the previous turn's tokens. Those turns are then
not merged and each becomes its own training sequence: correct, but more learner work per step than upstream.

## Things to check in the smoke before a long run

- **Does the SFT model produce any reward?** This is the main risk. Terminus-2 expects a JSON reply with keystrokes
  on every turn, and on LUMI an earlier 9B SFT checkpoint failed this interface (all rewards 0, many parser errors;
  see `qualification-harbor-training-2026-09-08.md`). If every group is all-0, GRPO has no signal and the run burns
  GPU hours and Modal credit for nothing. Count rewards in `trials/*/result.json` and parser errors in the agent logs.
  If they are all 0, consider a run on a few hundred tasks with `N_SAMPLES=16` to find any mixed groups.
- **Thinking length.** With `/think` every turn starts with reasoning, which uses up the 4,096-token turn cap and
  the 16k context faster; the `prelude-base` smoke already had 6 of 32 trials fill the context. A turn cut off inside
  `<think>` has no keystrokes. Check how often turns hit the cap and trials end in `ContextLengthExceededError`; if
  often, raise `MAX_TOKENS_PER_TURN` and `MAX_MODEL_LEN` (learner memory permitting) before the long run.
- **Context and learner memory.** The model takes 262k positions, so `MAX_MODEL_LEN` is limited by memory, not
  the model. The default is 16,384; upstream's 32,768 is worth trying once a 16k step fits. The largest cost is the
  full-vocabulary logits: 262,145 values per token, about 8.6 GB in bf16 per 16k sequence, before activations.
  A trial that hits the limit ends in `ContextLengthExceededError` and is trained with reward 0; check how often that
  happens.
- **Modal limits and cost.** A full step starts 256 sandboxes (128 at a time), and each one runs the verifier's
  `apt-get` + `uv` install. Check the Modal dashboard after the smoke for concurrency limits, failed sandbox
  creations and cost per step, and lower `MAX_CONCURRENCY` if creations are throttled.
- **Proxy rotation.** In a run longer than 3.5 h, check `logs/*.modal-proxy.out` for a new generation every
  ~3 h 20 min, and `generate/num_masked_instances` around the switch.
- **Memory.** If the learner runs out of memory, lower `MAX_MODEL_LEN` or set `POLICY_CPU_OFFLOAD=true`. If vLLM does, lower
  `GPU_MEMORY_UTILIZATION`. Driver and flash-attn notes are the same as in [leonardo-skyrl-gsm8k.md](leonardo-skyrl-gsm8k.md).
