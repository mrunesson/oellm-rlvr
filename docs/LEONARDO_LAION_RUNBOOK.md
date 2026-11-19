# Runbook: LAION OT-validated RL environments on Leonardo

For humans and for AI agents picking up this work. It says what exists, what state it is in, how to run each step, what has gone
wrong before and how it was fixed, and what to do next. Written 16 November 2026 from a working session. For the narrative and the
measured results see [leonardo-laion-rl-writeup.md](leonardo-laion-rl-writeup.md). **If something here disagrees with the live
cluster, trust the cluster and fix this file.**

## 0. Read this first (rules for agents)

- **The account and the cluster are shared.** Other sessions run their own experiments on the same Slurm account. Never cancel, edit
  or reuse a job, directory, port, token or file you did not create. Check `squeue -u $USER` before assuming a job is yours.
- **Do not publish or push without the user's explicit instruction.** Commits are local until told otherwise. This repo and the
  benchmark repo are public. Never put tokens, passwords, personal e-mail addresses, usernames or cluster paths into commits,
  issues, pull requests or outside requests.
- **Do not run code you did not read**, and do not install new third-party packages on the cluster without asking.
- **Leonardo compute nodes have no internet.** Anything that downloads must run on a login node or through the build proxy.
- **Do not poll.** Use a background monitor or `squeue` checks spaced out; the harness blocks chained `sleep`.
- **Report honestly.** If a run failed or a number is uncertain, say so. Several early diagnoses here were wrong (see section 7).
- Terminal-Bench 2.1 and TBLite are **evaluation only**. Never train on them.

## 1. Goal and status

Goal: run RLVR on [`laion/ot-validated-rl-datasets`](https://huggingface.co/collections/laion/ot-validated-rl-datasets) on Leonardo,
using Harbor agentic tasks that run in apptainer-style sandboxes.

| Step | Status (16 Nov 2026) |
|---|---|
| Download `laion/rts-1k-rootless-apptainer` (984 tasks) | done |
| Build x86_64 sandbox images (801 unique) | done, 696 built; about 105 fail (task-specific npm `@jsr` 404s) |
| Validate tasks: oracle must pass, nop must fail | done, **741 of 984 usable** |
| Roll out a model against the tasks and score it | working |
| Find tasks with mixed pass/fail outcomes | first pass: 12 tasks, a lower bound; rerun pending |
| Port the SkyRL training step | **not started** |
| The other 11 datasets in the collection | not started |

No training has happened. The reason to hold off is data, not code: with Prelude-9B-SFT-v2 only about 8 % of scored trials pass.

## 2. Facts about Leonardo that shaped everything

- Scheduler: Slurm. GPU partition `boost_usr_prod` (4 × A100 64 GB per node, 32 cores). Debug QoS `boost_qos_dbg` gives nodes quickly
  but only 30 minutes. `lrd_all_serial` is a login-type partition with internet. No `dcgp_usr_prod` budget on our account.
- Container runtime is **SingularityPRO 4.3.1** (`singularity`, no `apptainer`). No podman, buildah, enroot or proot. `ptrace_scope=2`.
- **Login nodes**: internet, but user namespaces disabled. **Compute nodes**: user namespaces enabled, **no internet**.
- `/etc/subuid` is empty, so `singularity --fakeroot` fails everywhere.
- System `python3` is 3.6. Use the Harbor venv's Python 3.12.
- GPU nodes run NVIDIA driver **535** (CUDA 12.2 and older natively). Newer CUDA containers need CUDA forward compatibility
  (section 5.3). The rollout job's default vLLM container lives in another project's work area and is not readable from other
  accounts; pull your own (5.3).
- Node-local `/tmp` (the job's `TMPDIR`) is **10 GB** by default; Slurm rejects larger `tmpfs` requests (tried 40 to 200 GB).
- The SSH certificate lasts about 12 hours. Renew from your own machine (needs your login and second factor, an agent cannot do it):
  `step ssh login '<CINECA e-mail>' --provisioner cineca-hpc`, then check `ssh leonardo hostname`.
- `sbatch` copies the script at submit time. Editing a script does not change jobs already queued.
- A `sbatch` job writes logs to `logs/%j...` relative to the submit directory. Submit from a directory that has a `logs/` folder.

Configure site settings once: copy [../leonardo.env.example](../leonardo.env.example) to `leonardo.env` (git-ignored), fill in
`OELLM_ACCOUNT`, `OELLM_WORK`, `OELLM_SCRATCH`, then `source leonardo.env` before `sbatch`. Everything below uses `$OELLM_SCRATCH`.
Alternatively export the same variables near the top of `~/.bashrc` (above any interactive-only `return`), which bash reads for
remote `ssh leonardo '...'` commands too, so no `source` is needed; see the
[image build guide](LEONARDO_LAION_IMAGE_BUILD_GUIDE.md), step 1. Avoid `~/.bash_profile` for this: only login shells read it, so
remote commands see empty variables and every path collapses to `/oellm-rl/...`.

## 3. Layout

Repo files (this repository):

| Path | Purpose |
|---|---|
| `scripts/laion_rootless/singularity` | Wrapper that makes `singularity build` work without root (section 5.1) |
| `scripts/laion_rootless/fakeown.c` | `LD_PRELOAD` shim: `chown`/`setuid`-style calls return success |
| `scripts/laion_rootless/local_relay.py` | Loopback relay that adds proxy credentials for Java (which ignores URL credentials) |
| `scripts/laion_proxy.py`, `leonardo_laion_proxy.sbatch` | Token-protected HTTP/CONNECT proxy on an internet node (4 h) |
| `scripts/leonardo_harbor_apptainer_bootstrap.sh` | One-time: Harbor fork at `82efa984` and its Python 3.12 venv (login node, needs `uv`) |
| `scripts/leonardo_laion_download.sh` | One-time per dataset: download `tasks.tar.zst`, extract, verify `SHA256SUMS` (login node) |
| `scripts/laion_context_hash_shards.py` | Lists the images still to build (Harbor's own discovery and cache check) and splits them into shards |
| `scripts/leonardo_laion_sif_build_rootless.sbatch` | Builds sandbox images for one dataset or one shard |
| `scripts/leonardo_laion_validate.sbatch` | Oracle and nop validation, writes `leonardo-compatible.jsonl` |
| `scripts/leonardo_laion_validate_all.sh` | Validates every task that has an image: chunks, submits, resubmits, merges into one list |
| `docs/LEONARDO_LAION_IMAGE_BUILD_GUIDE.md`, `docs/LEONARDO_LAION_VERIFICATION_GUIDE.md` | Step-by-step guides for humans: new account to images, images to the usable-task list |
| `scripts/summarize_laion_validation.py` | Summarises validation results (keyed on the task directory name) |
| `scripts/leonardo_laion_agent_rollout.sbatch` | vLLM + Terminus-2 + sandboxes + verifier, writes `rollout-summary.json` |
| `scripts/leonardo_vllm_container_pull.sbatch` | Pulls the public `vllm/vllm-openai` image into a SIF on an internet node |
| `scripts/leonardo_vllm_smoke.sbatch` | One-GPU check: driver, CUDA compat, vLLM imports, model loads, one chat reply |
| `scripts/analyze_rollout.py` | Classifies rollout outcomes (passed, used all turns, stopped early, each error type) |
| `scripts/leonardo_halu_bench.sbatch` | Hallucination benchmark job (section 5.6) |
| `src/oellm_rlvr/web_gateway.py`, `scripts/leonardo_web_gateway.sbatch` | Web-search gateway sidecar (section 6). Tested locally, never run on Leonardo |
| `docs/leonardo-laion-rl-writeup.md` | Narrative status and measured results |

On Leonardo (under `$OELLM_SCRATCH/oellm-rl/`): `venvs/harbor-laion` (Python 3.12 with Harbor), `src/harbor-laion` (the Harbor fork
`marianna13/harbor` at commit `82efa984`), `data/laion-ot/<dataset>/tasks`, `sif-cache-x86_64/` (built images), `proxy/`
(`proxy.host`, `proxy.token`, `fakeown.so`), `laion-validation/` (validation outputs and
`<dataset>-leonardo-compatible-all.jsonl`, the rollout job's default task list; older runs named it
`rts-1k-leonardo-compatible-all.jsonl`, which the rollout job still finds), `laion-rollouts/<dataset>-<jobid>/` (rollout outputs),
`models/` (downloaded checkpoints), `containers/` (a pulled vLLM image, if the project container is not readable).
Copy or clone the repo's `scripts/` to the cluster before submitting (login nodes have internet, so `git clone` works).

## 4. Daily commands

```bash
ssh leonardo 'squeue -u $USER -o "%i %j %T %M %r"'              # what is running or queued
ssh leonardo 'sacct -u $USER -S now-1day -X -o JobID,JobName%20,State,Elapsed'
ls $OELLM_SCRATCH/oellm-rl/sif-cache-x86_64/*.sif | wc -l        # images built (the cache also holds .build.json files)
```

Count `*.sif` only. Counting every file in the cache doubles the number (each image has a `.sif.build.json`). This mistake was made once.

## 5. Procedures

### 5.0 Set up a new account (once)

Nothing under `$OELLM_SCRATCH/oellm-rl` is shared between users; a new account starts empty. On a login node:

```bash
git clone https://github.com/BirgerMoell/oellm-rlvr.git $OELLM_WORK/oellm-rlvr && cd $OELLM_WORK/oellm-rlvr && mkdir -p logs
scripts/leonardo_harbor_apptainer_bootstrap.sh                      # needs uv in ~/.local/bin; ~2 minutes
scripts/leonardo_laion_download.sh rts-1k-rootless-apptainer        # 338 MB extracted, 984 tasks
```

Submit every job below from this checkout (the sbatch files find `scripts/` and `logs/` relative to the submit directory).

### 5.1 Build images (once per dataset)

Rootless build recipe, in short: `unshare -rm singularity build` (root inside a single-id namespace), the `fakeown.so` shim so file
ownership changes succeed, apt told to run as root, internet through a token-protected proxy plus a per-job loopback relay.

1. Start the proxy on an internet node (it exits after 14,000 s, so plan for that):
   `sbatch -A $OELLM_ACCOUNT scripts/leonardo_laion_proxy.sbatch`. It writes `$PROXY_DIR/proxy.host` and `proxy.token`.
2. Write the shard lists. The script uses Harbor's `discover_builds` (one entry per unique build context: 801 for the 984 rts-1k
   tasks) and leaves out images already current in the cache, so rerunning it after a proxy expiry shards only the remaining work.
   It takes about 2.5 minutes on Lustre and prints nothing until done (two `SyntaxWarning`s from task test files are harmless):

```bash
R=$OELLM_SCRATCH/oellm-rl
PYTHONPATH=$R/src/harbor-laion/src $R/venvs/harbor-laion/bin/python -u scripts/laion_context_hash_shards.py \
  $R/data/laion-ot/rts-1k-rootless-apptainer/tasks $R/sif-cache-x86_64 $R/sif-shards/rts-1k-rootless-apptainer --shards 4
```

3. Submit the shards (about 100 images per hour overall with four shards of 3 to 4 workers; the first 696 took about 5 hours):

```bash
for f in $OELLM_SCRATCH/oellm-rl/sif-shards/rts-1k-rootless-apptainer/shard-*; do
  DATASET=rts-1k-rootless-apptainer CONTEXT_HASH_FILE=$f SIF_BUILD_WORKERS=4 \
    sbatch -A $OELLM_ACCOUNT --time=04:00:00 --export=ALL --job-name=laion-sif-$(basename $f) \
    scripts/leonardo_laion_sif_build_rootless.sbatch
done
```

4. Build jobs read the proxy address once at start (the loopback relay in `local_relay.py` reads it when the job starts).
   **If the proxy expires first, restart a proxy, rerun step 2, then resubmit the shards.** Finished images are skipped, so no work
   is lost. Each job uses its own singularity cache under `apptainer-cache/<jobid>`, deleted when the job exits.
5. Image failures about 13 % are expected (npm `@jsr/...` 404s and similar). `glibc` images only; alpine/musl images do not work with the shim.

### 5.2 Validate tasks

Whole dataset (every task whose image built), from the repo checkout. Step by step for humans:
[LEONARDO_LAION_VERIFICATION_GUIDE.md](LEONARDO_LAION_VERIFICATION_GUIDE.md).

```bash
scripts/leonardo_laion_validate_all.sh submit   # chunks of 150 tasks, one 30-minute debug job each, ATTEMPTS=3
scripts/leonardo_laion_validate_all.sh retry    # later: submits chunks the queue refused or that ended without results
scripts/leonardo_laion_validate_all.sh merge    # when all jobs ended: writes laion-validation/<dataset>-leonardo-compatible-all.jsonl
```

`submit` takes the tasks named in the cache's `.sif.build.json` files (written only after a successful build) and writes the
run to `laion-validation/<dataset>-full-<date>/` (`chunk-NN.txt`, `jobs.txt`, `no-image.txt`). The debug QoS accepts **4 jobs per
user** (`QOSMaxSubmitJobPerUserLimit`) and **runs 2 at a time** (the others show reason `QOSMaxJobsPerUserLimit` while they
wait; that is not a failure). `submit` stops at the limit and `retry` submits the rest once a job has ended. A 200-task chunk used
the full 30 minutes (one ran 31.5), hence the default of 150. 874 tasks therefore take about 6 chunks and 1.5 hours. `merge`
refuses to write the list while any chunk lacks results, and writes `failures.json` with the reason for every rejected task.
Overrides: `CHUNK`, `ATTEMPTS`, `QOS` (empty for the normal queue, then also set `TIME`), `WORKERS_PER_NODE`.

A single list or a shipped subset:

```bash
DATASET=rts-1k-rootless-apptainer SUBSET=list TASK_LIST=$PWD/val-chunk-00 ATTEMPTS=3 \
  sbatch -A $OELLM_ACCOUNT --qos=boost_qos_dbg --time=00:30:00 --export=ALL scripts/leonardo_laion_validate.sbatch
```

`SUBSET` is `overfit`, `diagnostic`, `all` or `list` (with `TASK_LIST`, one task id per line). About 250 tasks per 30-minute debug job
is the limit. The output `leonardo-compatible.jsonl` lists tasks where the oracle scored 1 and the nop agent 0 on every attempt.
One task with `ATTEMPTS=1` takes about 2.5 minutes in total.

### 5.3 Roll out a model

**vLLM.** The default `VLLM_CONTAINER` (under `/leonardo_work/OELLM_prod2026/`) and `VLLM_PY` (a venv under
`tmax-reproduction/`) only exist for the account that made them. Otherwise pull the public image and use its own Python:

```bash
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_vllm_container_pull.sbatch     # ~1 h on lrd_all_serial, 10.9 GB SIF
export VLLM_CONTAINER=$OELLM_SCRATCH/oellm-rl/containers/vllm-openai-v0.32.0-x86_64-cu129.sif VLLM_PY=python3
```

`v0.32.0-x86_64-cu129` is the newest tag built for CUDA 12 (later tags are CUDA 13 only). Two fixes make it run on driver 535;
both are on by default in the smoke and rollout jobs:

- `VLLM_CUDA_COMPAT=1`: puts the container's `/usr/local/cuda/compat` (CUDA forward-compatibility driver) ahead of the host libcuda
  that `--nv` adds. Without it the engine dies during kernel warm-up. The smoke job prints `cuDriverGetVersion 12090` when active.
  `VLLM_USE_FLASHINFER_SAMPLER=0` uses vLLM's PyTorch sampler instead of FlashInfer's run-time compiled one.
- `VLLM_STUB_TORCHCODEC=1`: the image's `torchcodec` is a CUDA 13 build; its `OSError` escapes vLLM's optional-audio import. A
  stand-in package raising `ImportError` disables audio input (unused by text models).

**Model.** Download on a login node (`hf download <repo> --local-dir $OELLM_SCRATCH/oellm-rl/models/<name>`, about 18 GB for 9B).
A `chat_template.jinja` in the model folder is passed to vLLM automatically (`CHAT_TEMPLATE` overrides). Check a new model or
container on one GPU first (debug queue, about 5 minutes; ends with `SMOKE OK`):

```bash
MODEL=$OELLM_SCRATCH/oellm-rl/models/<model> sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_vllm_smoke.sbatch
```

Then a short rollout (`N_TASKS=8 ATTEMPTS=2 MAX_TURNS=30`, `--time=01:00:00`, about 30 minutes) before the full one:

```bash
MODEL=$OELLM_SCRATCH/oellm-rl/models/<model> DP_SIZE=4 N_TASKS=100 ATTEMPTS=4 CONCURRENT=48 \
  WORKERS_PER_NODE=28 MAX_TURNS=100 SEED=2 \
  sbatch -A $OELLM_ACCOUNT --time=04:00:00 --export=ALL scripts/leonardo_laion_agent_rollout.sbatch
```

What it does: starts vLLM (`DP_SIZE` replicas, one per GPU), starts the Harbor bridge server and workers, waits for vLLM (loading 17 GB
on four GPUs takes about 3 to 10 minutes), then runs Harbor's **Terminus-2** agent on `N_TASKS` shuffled tasks from the validated list,
and writes `rollout-summary.json` (per-task rewards, exceptions). Run `scripts/analyze_rollout.py <run>/jobs/rollout --max-turns <N>` for a
breakdown.

Useful variables: `TASK_LIST` (default `laion-validation/<dataset>-leonardo-compatible-all.jsonl` from 5.2), `MAX_TURNS`, `TEMPERATURE` (0.6), `MAX_MODEL_LEN` (32768),
`MAX_TOKENS_PER_TURN` (4096), `AGENT_TIMEOUT_MULT` (3.0), `EXCLUDE_BASE` (see 7), `STAGING_ROOT` (default `/dev/shm`),
`BRIDGE_OVERLAY_SIZE_MB` (512), `VLLM_WAIT_STEPS` (5 s steps, default 420).

The job also writes `service-logs/staging-usage.log` (sandbox staging size by component, every minute) and `vllm.log`, `worker.log`,
`server.log`.

### 5.4 Read results

Rewards are in `jobs/rollout/<task>__<id>/verifier/reward.txt` and `result.json`. A **mixed task** (some attempts pass, some fail) is a
training candidate. A task with 0 of 4 passes gives no RL signal, and a task that passes 4 of 4 is too easy. Trials that errored
(`exception_info` set) have no reward; count them separately, never as zero.

### 5.5 Models used so far

- `marianna13/prelude-9b-sft-v2`: Qwen3 architecture, ChatML template with `<think>` and Hermes tool calls. Works in the harness.
  On a second account (vLLM 0.32 container, 19 Nov 2026), 8 tasks x 2 attempts at 30 turns: 16 of 16 scored, 1 passed, 13 used
  all 30 turns; so give it 100 turns.
- `openeurollm/Prelude-Marianna-midtrain-SFT-Excellent-v0.2`: same template family (`chat_template.jinja`), loads and answers
  with separate reasoning. In the harness it declares the task done after about 3 turns (11 of 16 trials), 0 of 12 scored passed,
  and 4 of 16 ended in `AddTestsDirError` (bridge HTTP 409 when adding the tests); same 8 tasks as above.
- `openeurollm/prelude-base`: base model without a chat template; not usable with Terminus-2 as is.
- `birgermoell/oellm-9b-256k-stage5-e8-search-rl-step160`: fails in the harness (vLLM rejects think tags in message content). Not a
  terminal-agent model.

### 5.6 Hallucination benchmark

Repo: `BirgerMoell/oellm-hallucination` (500 questions in 10 languages; judge `gemma-4-31B-it` on 2 GPUs).

```bash
git clone https://github.com/BirgerMoell/oellm-hallucination $SCRATCH/halu-bench-<tag>
BENCH=$SCRATCH/halu-bench-<tag> SCRATCH=$SCRATCH MODELS="<label>=<model dir>" \
  sbatch -A $OELLM_ACCOUNT --time=02:00:00 --export=ALL scripts/leonardo_halu_bench.sbatch
```

It uses the project container and a Python with vLLM 0.24.0 (needed for Gemma 4), and the judge model already on disk at
`/leonardo_work/OELLM_prod2026/models/google/gemma-4-31B-it`. That setup matches the benchmark repo's baseline, so numbers compare.
About 25 minutes to generate and 25 minutes to judge.

## 6. Web-search gateway (not in use yet)

`src/oellm_rlvr/web_gateway.py` is a sidecar so agents can search and fetch the live web through a restricted interface: only
`web_search` and `web_fetch`, URLs only from search results, fetched pages or an operator seed, HTTPS on port 443, no IP literals,
allow and deny lists, redirects re-validated hop by hop, connections pinned to validated IPs, size and time caps, HTML reduced to text
with hidden text and chat-template markup neutralised, per-session budgets, an audit log, and a record/replay cache. 38 unit tests pass
(`python -m pytest tests/test_web_gateway.py`). Not done: deploying it on a node with internet (`leonardo_web_gateway.sbatch` needs a venv with
`fastapi`, `uvicorn`, `pyyaml`), registering the tools in a rollout, choosing a backend (Wikipedia keyless, Brave with a key, or
SearXNG), and deciding whether to relay through a small VPS. Compute nodes can reach a service on a serial node over the cluster
network, as another session demonstrated with its own relay.

## 7. Known failures and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `singularity build --fakeroot` fails | empty `/etc/subuid` | use the rootless wrapper (5.1) |
| Build: `$APPTAINER_ROOTFS` empty, `mkdir /app` fails | SingularityPRO only defines `SINGULARITY_ROOTFS` | wrapper rewrites the def file |
| Build: Maven "Network is unreachable" | Java ignores proxy credentials in URLs | loopback relay + `JAVA_TOOL_OPTIONS` (done). Clojure tooling still ignores it, so 1 known image fails |
| Build: `TOOMANYREQUESTS` from Docker Hub | one shared proxy IP | base images pulled from `mirror.gcr.io` (done) |
| Build: `invalid character 'e'/'}' after top-level value` | corrupted shared singularity cache (written during rate limiting and by parallel jobs) | one cache dir per job (done; until 17 Nov the sbatch file still shared one cache despite this row); delete a poisoned cache |
| `ls: ... OELLM_prod2026/container_images: Permission denied` | the default vLLM container belongs to another project | pull your own (5.3) |
| vLLM: `OSError: Could not load this library: .../torchcodec/libtorchcodec_image.so` (`libnvrtc.so.13`) | CUDA 13 torchcodec in a CUDA 12.9 image | `VLLM_STUB_TORCHCODEC=1` (default) |
| vLLM: `EngineCore failed to start` in `warmup_kernels` | CUDA 12.9 container on the 535 driver | `VLLM_CUDA_COMPAT=1`, `VLLM_USE_FLASHINFER_SAMPLER=0` (defaults) |
| Rollout fails in seconds: `FileNotFoundError ... compatible-all.jsonl` | task list name differs | fixed: the default follows `validate_all.sh merge`; or set `TASK_LIST` |
| `Failed to install asciinema using pip` in the rollout `.err` | the agent tries to install a recorder; sandboxes are offline | harmless |
| `sbatch: error: QOSMaxSubmitJobPerUserLimit` | the debug QoS accepts 4 jobs per user | wait for one to end; `leonardo_laion_validate_all.sh retry` |
| Pending jobs with reason `QOSMaxJobsPerUserLimit` | the debug QoS runs 2 jobs per user at a time | nothing; they start when one ends |
| `LiteLLM ... ConnectError ... model_prices_and_context_window.json` in `.err` | litellm fetches a price list; compute nodes are offline | harmless, it falls back to its built-in copy |
| Download: thousands of `sha256sum: tasks/...: No such file or directory` | checksums were verified before `tasks.tar.zst` was extracted | extract first, then verify (fixed 17 Nov) |
| Remote command finds nothing under `/oellm-rl/...` | `OELLM_*` set in `~/.bash_profile` (not read by `ssh host 'cmd'`) or below an interactive-only `return` in `~/.bashrc` | move the exports to the top of `~/.bashrc` (section 2) |
| Build jobs die when the proxy ends | proxy limit is 4 h and build jobs read the address once | restart proxy, resubmit shards |
| Summary shows task "failed" though it passed | Harbor reports the `task.toml` name, not the directory name | summarizer keys on the task directory (done) |
| vLLM: `libxcb.so.1` missing during startup | spawned workers import OpenCV | `VLLM_WORKER_MULTIPROC_METHOD=fork` (done) |
| Every trial fails at once: `hosted_vllm models require model_info` | Terminus-2 needs token limits and costs | `--agent-kwarg model_info=...` (done) |
| Trial error: `tmux: unknown option -- e` | `tmux` too old in `ubuntu:20.04`, `ubuntu:18.04`, `debian:bullseye`, `debian:buster` images | those bases are excluded by default (`EXCLUDE_BASE`) |
| Job exits at 10 min: vLLM "did not become ready" | 4 replicas load 17 GB from Lustre at once | wait up to 35 min (done) |
| Most trials use every turn and fail | turn limit (30) too short, tasks allow 600 to 1,800 s | use 100 turns and `AGENT_TIMEOUT_MULT=3` |
| `AgentTimeoutError` after about 60 to 90 turns | task agent timeout (600 s × multiplier) | raise the multiplier |
| vLLM `BadRequestError: Think tags in content` | model writes `<think>` in content, the Qwen3 reasoning parser leaves it, Terminus-2 sends it back | open: serve without `--reasoning-parser` or strip think tags |
| `[Errno 28] No space left on device` at sandbox start, late in a run | per-sandbox staging fills the 10 GB node-local disk. **Exact component not identified** | stopgap: staging in `/dev/shm` (RAM, counts against `--mem`). Check `staging-usage.log` |
| `only root user can use sandbox as overlay in setuid mode` | a sandbox (directory) overlay started in setuid mode | 25 of 400 trials; not investigated |
| `Permission denied (publickey…)` | SSH certificate expired | renew (section 2) |
| Debug job fails right at launch: `OELLM_SCRATCH: source leonardo.env first` | environment not passed | `source leonardo.env`, `--export=ALL` |

Diagnostic mistakes to avoid: an error counter grouped by message text is meaningless if the text contains job IDs (normalise
first); "COMPLETING" is a normal state, not an ending; a job that is `PENDING` for hours on a busy node is waiting for priority, not
broken. Prefer the debug QoS for short jobs.

## 8. Measured results to date (Prelude-9B-SFT-v2)

- 3-task smoke test: oracle 1.0, nop 0.0 on all three.
- Validation: 79 of 100 (overfit set), 741 of 984 (full set).
- Second account from scratch (17 to 18 Nov 2026): 801 unique images, 708 built (88 %); 874 tasks with an image; **743 of 984
  usable** (131 rejected in validation, 110 without an image). Four 200-task chunks and one 74-task chunk, about 30 minutes each.
- Agent rollouts: 0 of 15 (20 turns); 2 of about 110 scored (30 turns; 87 % used every turn); **17 of 203 scored (100 turns)** with 12
  mixed tasks. 186 of 400 trials had no score because of sandbox errors (154 disk, 25 overlay, 7 tmux).
- Hallucination benchmark: 97.7 % fabrication on made-up subjects (default mode), 93.6 % (thinking), 2.3 % and 6.4 % honest "I don't
  know". Worse than every baseline in the benchmark repo.

## 9. What to do next (in order)

1. Check the profile rerun (the same 100 tasks, `/dev/shm` staging, usage log). Read `staging-usage.log` to see which component grows,
   and fix it: limit or drop the per-instance `/tmp` bind, shrink or skip the pre-copy of writable prefixes (`BRIDGE_WRITABLE_PREFIXES`),
   or clean up leaked staging. Rerun if needed.
2. Build the training pool from the mixed tasks. If there are too few (under a few dozen), widen: more tasks and attempts, easier
   datasets such as `laion/nemotron-terminal-easy-clean1k` or `laion/tasktrove-curriculum-easy`, or a stronger starting checkpoint.
3. Port the SkyRL training step with the apptainer environment: bridge and workers inside the training job, CUDA and A100 memory
   limits, the reward and verifier plumbing. The LUMI version is `scripts/lumi_harbor_agentic_train.sbatch`.
4. Decide on the gateway (section 6). Fix the E8 think-tag error if that model matters.

## 10. Glossary

- **SIF**: a Singularity/Apptainer image file; one per unique task build context.
- **Harbor**: the task and sandbox framework. A **task pack** is a directory of tasks (`task.toml`, `environment/Dockerfile`, `tests/`).
- **Bridge**: Harbor's apptainer environment service. A server plus workers start sandboxes and execute commands in them.
- **Terminus-2**: Harbor's terminal agent; it drives a `tmux` session in the sandbox and talks to the model through litellm.
- **Oracle / nop agent**: the reference solution (must score 1) and the do-nothing agent (must score 0). Both must hold for a task to be
  usable for RL.
- **Mixed task**: some attempts pass and some fail. The only kind that gives a policy-gradient signal.
