# Guide: run a language model on the LAION tasks on Leonardo

This guide takes you from the list of usable tasks to a measured result: how often a model solves each task, and which tasks are
worth training on. It continues where [LEONARDO_LAION_VERIFICATION_GUIDE.md](LEONARDO_LAION_VERIFICATION_GUIDE.md) ends, for the
dataset `laion/rts-1k-rootless-apptainer`. Every step says what it does, what you should see, and what to do if it goes wrong. It
assumes you are logged in to a Leonardo **login node** and have done the two earlier guides. The runbook calls this step a
"rollout"; see sections 5.3 to 5.5 of [LEONARDO_LAION_RUNBOOK.md](LEONARDO_LAION_RUNBOOK.md).

## What a rollout is, and why

In a rollout, a language model tries to solve tasks on its own, and the task's checker scores each try with 1 (solved) or 0. No
training happens; the model is only measured. Three programs work together on one GPU node:

| Part | What it does |
|---|---|
| **vLLM** | serves the model, one copy on each of the node's 4 GPUs |
| **Terminus-2** (Harbor's agent) | shows the model the task, runs the commands the model writes in the task's sandbox, and shows it the output; one such exchange is a **turn** |
| **Harbor's sandbox service and checker** | opens a fresh sandbox per try from the task's image, and scores the result when the agent stops |

Each task is tried several times (**attempts**). The result sorts tasks into three groups:

| Group | Meaning | Useful for reinforcement learning? |
|---|---|---|
| **always fail** | no attempt passed | no: there is no success to learn from |
| **mixed** | some attempts passed, some failed | **yes**: these are the training tasks |
| **always pass** | every attempt passed | no: there is nothing left to learn |

Time needed: about 1 hour of setup the first time (mostly downloads), 5 minutes for a one-GPU check, 30 minutes for a short test
rollout, and about 3 hours for 100 tasks.

---

## Part A: before you start (2 minutes)

### Step 1. Check the task list and the scripts

```bash
cd $OELLM_WORK/oellm-rlvr
wc -l < $OELLM_SCRATCH/oellm-rl/laion-validation/rts-1k-rootless-apptainer-leonardo-compatible-all.jsonl   # usable tasks
ls scripts/leonardo_vllm_container_pull.sbatch scripts/leonardo_vllm_smoke.sbatch \
   scripts/leonardo_laion_agent_rollout.sbatch scripts/analyze_rollout.py
df -h $OELLM_SCRATCH | tail -1                                         # free space
```

What to look for:

- **Task list:** about 740 lines. If the file is missing, finish the verification guide first.
- **Scripts:** if any is missing, your checkout is older than this guide; update it with `git pull`.
- **Space:** you need about 30 GB: 11 GB for the vLLM container and about 18 GB for a 9-billion-parameter model.

All commands below are run from `$OELLM_WORK/oellm-rlvr`, because the jobs write their logs to `logs/` there.

---

## Part B: get vLLM and the model (about 1 hour, once)

### Step 2. Pull the vLLM container

```bash
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_vllm_container_pull.sbatch
```

What it does: starts a job on `lrd_all_serial` (the partition with internet) that downloads the public `vllm/vllm-openai` image
and turns it into one file, `$OELLM_SCRATCH/oellm-rl/containers/vllm-openai-v0.32.0-x86_64-cu129.sif` (10.9 GB). It takes about
an hour. When done, `logs/<jobid>.vllm-pull.out` ends with the vLLM, PyTorch and transformers versions and a line
`vllm_container=...`.

Why this version: Leonardo's GPU nodes have NVIDIA driver 535, which speaks CUDA 12.2. `v0.32.0-x86_64-cu129` is the newest vLLM
built for CUDA 12; later versions need CUDA 13. The jobs below make it run on the older driver (see "Two fixes" in step 5).

Why not the project's own vLLM container: it lives in a project folder that other accounts cannot read.

If the job is still `PENDING` with reason `QOSMaxMemoryPerUser`, you already have another job on that partition; it starts when
that one ends. Submitting a second pull is harmless: it sees the file is already there and stops.

### Step 3. Tell the jobs to use it

Add these lines to the end of your `~/.bashrc`, so every new login and every job (`--export=ALL`) sees them:

```bash
export VLLM_CONTAINER=$OELLM_SCRATCH/oellm-rl/containers/vllm-openai-v0.32.0-x86_64-cu129.sif
export VLLM_PY=python3
```

Then run `source ~/.bashrc` once (or log in again). `VLLM_PY=python3` means "use the Python inside the container".

### Step 4. Download the model

Models come from Hugging Face. Login nodes have internet, so download there. Choose a model with a **chat template**; the one we
used is `marianna13/prelude-9b-sft-v2`:

```bash
REPO=marianna13/prelude-9b-sft-v2
DEST=$OELLM_SCRATCH/oellm-rl/models/${REPO#*/}
mkdir -p $DEST
nohup uvx --from huggingface_hub hf download $REPO --local-dir $DEST < /dev/null > $DEST.download.log 2>&1 &
```

What it does: downloads the model in the background (about 18 GB, 10 to 30 minutes), so it keeps going if your connection drops,
as long as you stay on the same login node. `${REPO#*/}` is the part after the slash, so the model ends up in
`models/prelude-9b-sft-v2`. Running it again resumes and skips files that are complete.

Follow it with `tail -f $DEST.download.log` (Ctrl-C stops watching, not the download). It is done when the log ends with the
folder path and the folder holds `config.json`, several `*.safetensors` files and, usually, `chat_template.jinja`:

```bash
ls $DEST
```

`chat_template.jinja` tells vLLM how to turn a conversation into the model's input. The jobs pass it to vLLM automatically. A
model without one (a base model, such as `openeurollm/prelude-base`) cannot be used with this agent as it is.

---

## Part C: check the model on one GPU (5 minutes)

### Step 5. Run the smoke test

```bash
MODEL=$OELLM_SCRATCH/oellm-rl/models/prelude-9b-sft-v2 sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_vllm_smoke.sbatch
```

What it does: a short job in the debug queue on one GPU that checks each link of the chain in turn, so a problem shows up here and
not an hour into a rollout:

1. the GPU and its driver (`nvidia-smi`);
2. the CUDA version vLLM sees inside the container;
3. that vLLM can be imported;
4. that the model loads with the same settings as the rollout;
5. that the model answers one chat message.

Read the result (replace `<jobid>` with the number `sbatch` printed, without `< >`):

```bash
cat logs/<jobid>.vllm-smoke.out
```

A good run shows `cuDriverGetVersion 12090` in step 1b, `ready after ~...s` in step 3, a short reply in step 4, and ends with
**`SMOKE OK`**.

**Two fixes for the older driver.** Both are switched on by default in the smoke and rollout jobs; you only need to know they
exist when reading logs:

- **CUDA forward compatibility** (`VLLM_CUDA_COMPAT=1`): the container carries a newer CUDA driver library
  (`/usr/local/cuda/compat`) that works on top of driver 535. The jobs put it first, so vLLM sees CUDA 12.9 (`12090`) instead of
  12.2 (`12020`). Without it, vLLM crashes while warming up. It also uses vLLM's own sampler instead of FlashInfer's
  (`VLLM_USE_FLASHINFER_SAMPLER=0`).
- **torchcodec stand-in** (`VLLM_STUB_TORCHCODEC=1`): one library in the container (`torchcodec`, for audio) was built for CUDA 13
  and crashes on import. The jobs put an empty stand-in in front of it, which turns off audio input. Text models do not use it.

---

## Part D: a short test rollout (30 minutes)

### Step 6. Run 8 tasks

```bash
MODEL=$OELLM_SCRATCH/oellm-rl/models/prelude-9b-sft-v2 DP_SIZE=4 N_TASKS=8 ATTEMPTS=2 MAX_TURNS=30 \
  CONCURRENT=16 WORKERS_PER_NODE=16 SEED=2 \
  sbatch -A $OELLM_ACCOUNT --time=01:00:00 --export=ALL scripts/leonardo_laion_agent_rollout.sbatch
```

What the settings mean:

| Setting | Meaning |
|---|---|
| `DP_SIZE=4` | one model copy on each of the 4 GPUs |
| `N_TASKS=8` | take 8 tasks from the usable list (shuffled; `SEED` fixes the order, so the same seed gives the same tasks) |
| `ATTEMPTS=2` | try each task twice |
| `MAX_TURNS=30` | stop an attempt after 30 turns |
| `CONCURRENT=16`, `WORKERS_PER_NODE=16` | run up to 16 attempts (and sandboxes) at the same time |

What it does, in order (all inside one job on one node):

1. starts vLLM; loading the model on 4 GPUs takes 3 to 10 minutes;
2. starts Harbor's sandbox service;
3. runs the agent on the 16 attempts;
4. scores each attempt and writes a summary.

The results go to `$OELLM_SCRATCH/oellm-rl/laion-rollouts/rts-1k-rootless-apptainer-<jobid>/`.

This run is not about the score. It proves that everything works together before you spend hours on a big one. Read it as in
step 9. It is good if all or nearly all of the 16 attempts got a score (the `ERROR` lines are few or absent). Ours had 16 of 16
scored and 1 passed.

---

## Part E: the real rollout (about 3 hours)

### Step 7. Submit it

```bash
MODEL=$OELLM_SCRATCH/oellm-rl/models/prelude-9b-sft-v2 DP_SIZE=4 N_TASKS=100 ATTEMPTS=4 MAX_TURNS=100 \
  CONCURRENT=48 WORKERS_PER_NODE=28 SEED=2 \
  sbatch -A $OELLM_ACCOUNT --time=04:00:00 --export=ALL scripts/leonardo_laion_agent_rollout.sbatch
```

The same job, bigger: 100 tasks, 4 attempts each (400 attempts), up to 100 turns each, 48 at a time. Four attempts per task are
what make "mixed" visible; 100 turns give the model room for longer tasks. Ours took 2 hours 56 minutes of the 4 allowed.

### Step 8. Follow the progress

```bash
squeue -u $USER -n laion-agent-rollout -o "%i %T %M %L"          # state, time used, time left
O=$OELLM_SCRATCH/oellm-rl/laion-rollouts/rts-1k-rootless-apptainer-<jobid>
ls $O/jobs/rollout/*/result.json 2>/dev/null | wc -l              # attempts finished (done at 400)
cat $O/jobs/rollout/*/verifier/reward.txt 2>/dev/null | sort | uniq -c   # scores so far: count of 0 and 1
tail -5 logs/<jobid>.laion-agent-rollout.out
```

For the first 5 to 10 minutes nothing is finished: vLLM is loading the model. If the count stays at 0 much longer, look at the end
of `$O/service-logs/vllm.log`.

When the job has ended, the `.out` log ends with a summary (tasks, attempts, how many were scored, the errors by type) and the
line `rollout_artifact=...`.

---

## Part F: understand the result

### Step 9. Sort the attempts and the tasks

```bash
O=$OELLM_SCRATCH/oellm-rl/laion-rollouts/rts-1k-rootless-apptainer-<jobid>
$OELLM_SCRATCH/oellm-rl/venvs/harbor-laion/bin/python scripts/analyze_rollout.py $O/jobs/rollout \
  --max-turns 100 --examples 1 --groups $O/task-groups.json
```

Use the same `--max-turns` as the rollout (30 for the test). It prints two things.

**First, every attempt in one class**, with its share, the average number of turns and one example (the last lines of the
checker's output and the model's last message):

| Class | Meaning |
|---|---|
| `PASS` | the checker gave 1 |
| `FAIL: used all 100 turns` | the agent was still working when the turn limit stopped it |
| `FAIL: agent stopped early` | the model said it was done, but the checker gave 0 |
| `ERROR <type>` | the attempt crashed and got **no score**. These are left out of all pass rates, never counted as 0 |

**Second, the tasks in their groups**: `always_pass`, `mixed`, `always_fail`, and `no_score` (every attempt crashed), with the
list of mixed tasks. The groups are saved in `task-groups.json` in the run folder; its `mixed` list is the input for training.

### What we got

`marianna13/prelude-9b-sft-v2`, 100 tasks x 4 attempts, 100 turns (November 2026):

| Attempts | Count | Share |
|---|---|---|
| used all 100 turns | 236 | 59% |
| stopped early, failed | 101 | 25% |
| **passed** | **22** | **6%** |
| errors (no score) | 41 | 10% |

| Tasks | Count |
|---|---|
| always pass | 2 |
| **mixed** | **8** |
| always fail | 88 |
| no score | 2 |

What this means:

- **Pass rate.** 22 of 359 scored attempts passed (6%), the same as the 30-turn test. More turns did not help this model.
- **Training tasks.** Only about 1 task in 10 is mixed. To find more, run the rest of the list (step 10) or start from a stronger
  model.
- **Using all turns.** Many of these attempts end with the message `Technical difficulties. Please continue with the task.` That
  is the agent filling in for a model reply it could not use. One possible reason is that the conversation outgrew the model's
  window (`MAX_MODEL_LEN`, 32768 tokens): these attempts wrote about 23,000 tokens on average. This is not yet confirmed; check it
  before reading that 59% as the model's own failure.
- **Stopped early.** The model often declared success when the checker disagreed. In the example, it trusted a check script inside
  the task instead of the task's requirements.
- **Errors** (not the model's fault):
  - `AgentTimeoutError` (16): an attempt ran longer than its time limit (`AGENT_TIMEOUT_MULT` times the task's own limit).
  - `HTTPError 404` (12) and `AddTestsDirError` (5): the sandbox was gone when the agent or the checker reached for it.
  - `RuntimeError ... tmux: unknown option -- e` (8): the task's image has a tmux too old for the agent. These tasks will fail the
    same way every time.

### Step 10. Run more tasks

The tasks are taken in a fixed shuffled order, so a run with the same `SEED` and a larger `N_TASKS` repeats the first 100. To try
other tasks, make a list without the ones already done and pass it as `TASK_LIST`:

```bash
O=$OELLM_SCRATCH/oellm-rl/laion-rollouts/rts-1k-rootless-apptainer-<jobid>
python3 -c 'import json,sys
done = {t for v in json.load(open(sys.argv[1])).values() for t in v}
for l in open(sys.argv[2]):
    if json.loads(l)["task_id"] not in done: print(l, end="")' \
  $O/task-groups.json $OELLM_SCRATCH/oellm-rl/laion-validation/rts-1k-rootless-apptainer-leonardo-compatible-all.jsonl \
  > $OELLM_SCRATCH/oellm-rl/rollout-remaining.jsonl
wc -l < $OELLM_SCRATCH/oellm-rl/rollout-remaining.jsonl
```

Then repeat step 7 with `TASK_LIST=$OELLM_SCRATCH/oellm-rl/rollout-remaining.jsonl` added in front of `sbatch`. At about 3 hours per
100 tasks, the remaining ~640 tasks take about 7 such jobs. Each job takes the first `N_TASKS` of its shuffled list, so give jobs
that run at the same time different, non-overlapping lists (for example, split the file with `split -n l/7`).

### Trying another model

Download it (step 4), run the smoke test (step 5) and the short rollout (step 6) before a full one. Models we tried:

| Model | Result |
|---|---|
| `marianna13/prelude-9b-sft-v2` | works; results above |
| `openeurollm/Prelude-Marianna-midtrain-SFT-Excellent-v0.2` | loads and answers, but declares the task done after about 3 turns; 0 of 12 scored attempts passed in the short test |
| `openeurollm/prelude-base` | no chat template; not usable as is |

---

## Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| `FATAL: ... permission denied` for a `.sif` under `/leonardo_work/OELLM_prod2026/` | `VLLM_CONTAINER` is not set, so the job uses the project container, which you cannot read | Steps 2 and 3 |
| `no model at ...` | `MODEL` points to a folder without `config.json` | Check the path; wait for the download to finish (step 4) |
| Smoke step 1b prints `12020`, or the vLLM engine dies while warming up | CUDA forward compatibility is not active | Make sure `VLLM_CUDA_COMPAT` is not set to 0 and the container has `/usr/local/cuda/compat` (the `cu129` image does) |
| `OSError: libnvrtc.so.13` mentioning `torchcodec` | The torchcodec stand-in is off | Do not set `VLLM_STUB_TORCHCODEC=0` with this container |
| `no task list at ...` | The usable-task list is missing or has another name | Run `scripts/leonardo_laion_validate_all.sh merge`, or set `TASK_LIST` |
| No attempts finished after 15 minutes | vLLM did not come up | Look at the end of `$O/service-logs/vllm.log` |
| Job `PENDING` with reason `Priority` or `Resources` | The GPU partition is busy | Wait |
| Many `ERROR` attempts of one type | A problem outside the model | Look at one example's `result.json` (`exception_info`) and `$O/service-logs/server.log` and `worker.log` |
| `asciinema: command not found` in the logs | The agent tries to record the terminal; the sandboxes do not have the recorder | Harmless |
| `perl: warning: Setting locale failed` | Your Mac sends a locale Leonardo does not know | Harmless; `export LC_ALL=C` on the Mac before `ssh` silences it |
| `bash: syntax error near unexpected token` with `<jobid>` | The placeholder was typed literally | Replace `<jobid>` with the number, without `< >` |
