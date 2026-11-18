# Guide: build the sandbox images for a LAION dataset on Leonardo

This guide takes you from an empty Leonardo account to a folder of ready-to-use sandbox images for the dataset
`laion/rts-1k-rootless-apptainer`, and then (Part D) to a checked list of the tasks that work. Every step says what it does and why, what you should see, and what to do if it goes wrong.
It assumes you are logged in to a Leonardo **login node** in a normal shell. For background and other procedures (validation,
rollouts) see [LEONARDO_LAION_RUNBOOK.md](LEONARDO_LAION_RUNBOOK.md).

## What you are building, and why it takes several steps

The dataset holds 984 small programming tasks. Each task describes its own Linux environment with a Dockerfile. To run a task
on Leonardo, that environment has to exist as a **Singularity image** (a `.sif` file). Many tasks share an environment, so the
984 tasks need 801 images.

Building those images on Leonardo is awkward for three reasons, and the steps below exist to work around them:

| Problem | Workaround used in this guide |
|---|---|
| Building normally needs root, and Leonardo's `--fakeroot` does not work (no `/etc/subuid`) | A wrapper runs each build in a private user namespace, where you count as root. This only works on **compute nodes**. |
| Compute nodes have **no internet**, but builds download packages | A small **proxy** job runs on a node that has internet. Build jobs send their downloads through it. |
| The proxy job can run for at most about 4 hours | Builds happen in **rounds**: start a proxy, build for as long as it lives, then start a new proxy and continue where you left off. |

Time needed: about 15 minutes of setup, then two or three rounds of about 4 hours each. You only need to act between rounds.

Expect about 1 in 8 images to fail permanently (mostly because a JavaScript package they need no longer exists online). That
is normal; those tasks are simply left out later.

---

## Part A: one-time setup (about 15 minutes)

### Step 1. Check your site settings

All scripts find their folders through four environment variables. Check that they are set:

```bash
echo "account=$OELLM_ACCOUNT  work=$OELLM_WORK  scratch=$OELLM_SCRATCH"
```

You should see three non-empty values, such as an account like `AIFAC_...`, a work folder under `/leonardo_work/...`,
and a scratch folder under `/leonardo_scratch/...`.

If any are empty, add them to `~/.bashrc` (ask your project lead for the account name), log out and back in:

```bash
export OELLM_ACCOUNT=$PROJECT                                   # Slurm account that pays for the jobs
export OELLM_WORK=/leonardo_work/${PROJECT}/$USER               # small, permanent: the code lives here
export OELLM_SCRATCH=/leonardo_scratch/large/userexternal/$USER # large, temporary: data and images live here
```

Why `~/.bashrc`: bash reads it both for your interactive shells and for the shell that a remote `ssh leonardo '...'` command
runs in, so the variables are there without a `source`. `~/.bash_profile` is read only by login shells; the standard one loads
`~/.bashrc` (check that yours has a line like `. ~/.bashrc`). Slurm jobs submitted with `--export=ALL` inherit the variables.

> Put the lines **near the top** of `~/.bashrc`, before any line such as `[[ $- != *i* ]] && return` or
> `case $- in *i*) ;; *) return;; esac`. Those lines stop the file early for non-interactive shells, and anything after them would
> be missing in remote commands. If you previously set the same variables in `~/.bash_profile`, remove them there.
>
> Check from your own computer that it works: `ssh leonardo 'echo $OELLM_SCRATCH'` should print your scratch folder.

### Step 2. Get the code

This repository holds the build scripts. Clone it into your work folder and create the `logs/` folder that Slurm jobs write to:

```bash
git clone https://github.com/BirgerMoell/oellm-rlvr.git $OELLM_WORK/oellm-rlvr
cd $OELLM_WORK/oellm-rlvr
mkdir -p logs
```

**Always submit jobs from this folder.** The job scripts look for `scripts/` and `logs/` relative to where you submit.

### Step 3. Install Harbor (the task framework)

Harbor is the software that reads the tasks and builds the images. It needs Python 3.12, which Leonardo does not provide, so
the script uses the tool `uv` to fetch one. Check that you have `uv`:

```bash
command -v uv || echo "uv is missing"
```

If it is missing, install it into your home folder (this downloads a program from the internet; check with your team that this
is fine): `curl -LsSf https://astral.sh/uv/install.sh | sh`, then log out and back in.

Now run the setup script:

```bash
scripts/leonardo_harbor_apptainer_bootstrap.sh
```

What it does: downloads Python 3.12, clones a specific version of the Harbor fork (`marianna13/harbor`, commit `82efa984`) to
`$OELLM_SCRATCH/oellm-rl/src/harbor-laion`, and installs it with about 90 dependencies into a private environment at
`$OELLM_SCRATCH/oellm-rl/venvs/harbor-laion`. Nothing is installed system-wide.

It takes 1 to 2 minutes. Success looks like:

```
apptainer environment importable: harbor.environments.apptainer.prebuild_sifs ...
SingularityPRO version 4.3.1-1.el8
```

The line `no apptainer on PATH` is expected; Leonardo has `singularity` instead.

### Step 4. Download the dataset

```bash
scripts/leonardo_laion_download.sh rts-1k-rootless-apptainer
```

What it does: downloads the dataset from Hugging Face as one compressed archive, unpacks it to
`$OELLM_SCRATCH/oellm-rl/data/laion-ot/rts-1k-rootless-apptainer/tasks/`, and checks every file against the published
checksums. It takes about a minute.

Success ends with `done.` and a size of about 338 MB. Check the task count:

```bash
ls $OELLM_SCRATCH/oellm-rl/data/laion-ot/rts-1k-rootless-apptainer/tasks | wc -l    # expect 984
```

If you see thousands of lines like `sha256sum: tasks/...: No such file or directory`, your copy of the script is older than the
fix that unpacks before checking. Update the repository (`git pull`) and run the step again; nothing is downloaded twice.

---

## Part B: one build round (repeat until done)

Each round has four steps: start the proxy, make the work lists, submit the build jobs, and watch them.

### Step 5. Start the internet proxy

```bash
cd $OELLM_WORK/oellm-rlvr
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_laion_proxy.sbatch
```

What it does: starts a job on a Leonardo node that has internet access (partition `lrd_all_serial`). The job runs a small
password-protected proxy and writes its address and password to `$OELLM_SCRATCH/oellm-rl/proxy/`. It stops by itself after
about 3 hours 50 minutes.

The message `no gres/tmpfs specified, using default` is harmless. Wait until the job is `RUNNING`, usually within a minute:

```bash
squeue -u $USER -o "%i %j %T %M %L"
ls -la $OELLM_SCRATCH/oellm-rl/proxy/      # proxy.host and proxy.token should exist and be fresh
```

**Do not continue until the proxy is running.** A build job reads the proxy address once, when it starts, and cannot switch to
a new proxy later.

### Step 6. Make the work lists ("shards")

The 801 images are split into four lists so that four build jobs can work in parallel:

```bash
R=$OELLM_SCRATCH/oellm-rl
PYTHONPATH=$R/src/harbor-laion/src $R/venvs/harbor-laion/bin/python -u scripts/laion_context_hash_shards.py \
  $R/data/laion-ot/rts-1k-rootless-apptainer/tasks $R/sif-cache-x86_64 $R/sif-shards/rts-1k-rootless-apptainer --shards 4
```

What it does: groups the tasks by environment (each group gets an identifier, its "context hash"), **leaves out every image
that has already been built**, and writes the remaining identifiers into four files `shard-00` to `shard-03`.

Be patient: it reads every task file on the shared file system and takes about 2 to 3 minutes, printing nothing until the end.
Two `SyntaxWarning` lines about task test files are harmless. On the first round you should see:

```
984 tasks, 801 unique builds, 0 current in cache, 801 to build
.../shard-00: 201
.../shard-01: 200
...
```

On later rounds the "to build" number is smaller. If it prints no shard files, everything that can be built is done.

### Step 7. Submit the build jobs

First make sure no build jobs from an earlier round are still running (two jobs building the same image waste time):

```bash
squeue -u $USER -o "%i %j %T"      # there should be no laion-sif-shard-* jobs
```

Then submit one job per shard:

```bash
for f in $OELLM_SCRATCH/oellm-rl/sif-shards/rts-1k-rootless-apptainer/shard-*; do
  DATASET=rts-1k-rootless-apptainer CONTEXT_HASH_FILE=$f SIF_BUILD_WORKERS=4 \
    sbatch -A $OELLM_ACCOUNT --time=04:00:00 --export=ALL --job-name=laion-sif-$(basename $f) \
    scripts/leonardo_laion_sif_build_rootless.sbatch
done
```

What it does: each job gets one GPU-partition compute node (only its CPUs are used), starts a small local relay to the proxy,
and builds up to four images at a time from its shard. Finished images go to `$OELLM_SCRATCH/oellm-rl/sif-cache-x86_64/`.

The jobs may wait in the queue (`PENDING` with reason `Priority`) before they start. That is normal, but every minute they wait
is a minute less of proxy time.

### Step 8. Watch progress

A few minutes after the jobs start, check that one of them is working (replace `<jobid>` with one of your build job numbers):

```bash
head -6 $OELLM_WORK/oellm-rlvr/logs/<jobid>.laion-sif-shard-00.out
```

Good output names the node and the proxy, then lists builds:

```
node=lrdnNNNN... which singularity=.../scripts/laion_rootless/singularity proxy=loginNN...:<port>
...: 232 tasks, 201 unique SIF builds
[1/201] BUILD .../sif-cache-x86_64/build_002cfea2c18f12f6.sif
```

After that, check every 20 to 30 minutes:

```bash
squeue -u $USER -o "%i %j %T %M %L"                                       # job states and time left
ls $OELLM_SCRATCH/oellm-rl/sif-cache-x86_64/*.sif | wc -l                  # images built so far
grep -c "^FAILED" $OELLM_WORK/oellm-rlvr/logs/*.laion-sif-shard-*.out     # failures per job
```

Count only `*.sif` files: each image also has a `.sif.build.json` file next to it, so counting everything doubles the number.

Expect about 100 images per hour across the four jobs. To see why images failed:

```bash
grep -h "^FAILED" $OELLM_WORK/oellm-rlvr/logs/*.laion-sif-shard-*.out | cut -c1-200 | head -20
```

A few failures mentioning `npm` or `@jsr` and `404` are expected. Many failures mentioning connections or the network mean the
proxy has stopped: go to step 9.

### Step 9. Start the next round

When the proxy job has ended (it disappears from `squeue`), builds still in progress fail and the build jobs fail quickly through
the rest of their list. Nothing that was finished is lost.

1. Wait until all `laion-sif-shard-*` jobs have ended, or cancel **your own** build jobs with `scancel <jobid>`.
2. Go back to **step 5** and start a new proxy.
3. Run **step 6** again. It now lists only the images that are still missing, including the ones that failed.
4. Run **step 7** and **step 8**.

Stop when a round builds almost nothing new: the remaining images (about 100) fail for reasons a retry will not fix.

---

## Part C: using the result

### Where the images are

```
$OELLM_SCRATCH/oellm-rl/sif-cache-x86_64/
```

| File | Meaning |
|---|---|
| `build_<hash>.sif` | An image shared by all tasks with the same environment |
| `build_<task-name>-<hash>.sif` | An image for one task whose Dockerfile copies in task files |
| `<image>.sif.build.json` | Written only for a successful build: which tasks use the image, extra packages added, and a fingerprint used to skip it next time |

To find the image for a task:

```bash
grep -l rts_task_00a6948b8e05276bf20748da $OELLM_SCRATCH/oellm-rl/sif-cache-x86_64/*.build.json
```

The image is that file name without `.build.json`. The validation and rollout jobs find images by themselves; you only need this
for debugging.

To look inside an image, start a short interactive job on a compute node (images do not open on login nodes):

```bash
srun -A $OELLM_ACCOUNT -p boost_usr_prod --qos=boost_qos_dbg -t 00:30:00 --pty \
  singularity shell $OELLM_SCRATCH/oellm-rl/sif-cache-x86_64/<image>.sif
```

### Keep them safe

Scratch is temporary storage, and CINECA can delete scratch files that have not been used for a while (check the current
Leonardo user guide for the exact policy). If you need the images for weeks, or want to share them with the project, copy the
finished cache to the project work area.

### What comes next

The images are only the first step. Part D checks that each task actually works.

---

## Part D: check that the tasks work (about 1 to 2 hours)

An image existing does not mean its task is usable. Each task is run in its sandbox by two automatic "agents":

- the **oracle**, which runs the task's reference solution. It must score **1**, showing the task can be solved and the checker
  recognises a correct answer;
- the **nop** agent, which does nothing. It must score **0**, showing the checker does not pass an empty attempt.

Only tasks that get 1 and 0 on every attempt are kept for training. No model or GPU is involved; the sandboxes only use CPU.

### Step 10. Try one task first (optional, 5 minutes)

To see the whole chain work before starting the big run, run one task once. Pick any task whose image built:

```bash
cd $OELLM_WORK/oellm-rlvr
TASK=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_names"][0])' \
  $(ls $OELLM_SCRATCH/oellm-rl/sif-cache-x86_64/*.sif.build.json | head -1))
head -5 $OELLM_SCRATCH/oellm-rl/data/laion-ot/rts-1k-rootless-apptainer/tasks/$TASK/instruction.md   # what the task asks
echo $TASK > $OELLM_SCRATCH/oellm-rl/one-task.txt
DATASET=rts-1k-rootless-apptainer SUBSET=list TASK_LIST=$OELLM_SCRATCH/oellm-rl/one-task.txt ATTEMPTS=1 WORKERS_PER_NODE=2 \
  sbatch -A $OELLM_ACCOUNT --qos=boost_qos_dbg --time=00:30:00 --cpus-per-task=8 --mem=32G --export=ALL \
  scripts/leonardo_laion_validate.sbatch
```

It takes about 2.5 minutes once it starts. Then, with the job number `sbatch` printed:

```bash
tail -5 logs/<jobid>.laion-validate.out                       # ends with "compatible": 1 for a working task
find $OELLM_SCRATCH/oellm-rl/laion-validation/*-<jobid>/jobs -name reward.txt | xargs grep .   # oracle 1, nop 0
```

### Step 11. Check every task

```bash
cd $OELLM_WORK/oellm-rlvr
scripts/leonardo_laion_validate_all.sh submit
```

What it does: lists every task whose image built (874 of 984 in our run; the others are written to `no-image.txt`), splits them
into chunks of 200, and submits one 30-minute debug job per chunk. Each task is run 3 times with each agent.

The debug queue accepts only **4 jobs per user** at a time, so with 5 chunks you will see:

```
STOP: the queue refused chunk-04.txt (the debug queue allows 4 jobs per user).
```

That is expected. When one of the jobs has finished (check with `squeue -u $USER`), submit the rest:

```bash
scripts/leonardo_laion_validate_all.sh retry
```

`retry` is always safe to run: it skips chunks that are queued, running or done, and also resubmits a chunk whose job ended
without results (for example because it ran out of time).

### Step 12. Collect the results

When `squeue -u $USER` shows no more `laion-validate` jobs:

```bash
scripts/leonardo_laion_validate_all.sh merge
```

It prints how many tasks passed in each chunk and in total, and writes:

| File | What it holds |
|---|---|
| `$OELLM_SCRATCH/oellm-rl/laion-validation/rts-1k-rootless-apptainer-leonardo-compatible-all.jsonl` | The usable tasks, one per line. The rollout job reads this file by default |
| `.../laion-validation/rts-1k-rootless-apptainer-full-<date>/failures.json` | Every rejected task and why (oracle failed, nop passed, or an error) |

If it says `NOT merged`, some chunks have no results yet; it lists them. Run `retry`, wait, and run `merge` again.

In the earlier run on another account, 741 of the 984 tasks were usable. A similar number is a good result.

After this, the tasks are ready for a model: see section 5.3 of the [runbook](LEONARDO_LAION_RUNBOOK.md) for running a model on them.

---

## Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| Paths printed as `/oellm-rl/...` | The `OELLM_*` variables are not set in this shell | Step 1. For remote `ssh` commands, check that the lines are above any interactive-only `return` in `~/.bashrc` |
| `perl: warning: Setting locale failed` | Your laptop's language setting is passed over SSH | Harmless. Prefix the `ssh` command with `LC_ALL=C` to silence it |
| `set PYTHON=<python>=3.12> or install uv` | `uv` is not installed | Step 3 |
| Thousands of `sha256sum: ... No such file or directory` | An old copy of the download script | `git pull`, rerun step 4 |
| Step 6 seems to hang | It is reading about 10,000 files on the shared file system | Wait 3 minutes |
| A build job ends at once with `no proxy` | No proxy has been started under your account | Step 5 first |
| Build job log: `relay did not start` | The proxy files exist but the proxy is not reachable | Check that the proxy job is `RUNNING`; restart it (step 5) and resubmit |
| Many `FAILED` lines with network errors | The proxy has stopped | Step 9 |
| `invalid character 'e' after top-level value` | A corrupted download cache | Each job now has its own cache; resubmit. If it persists, report it |
| `sbatch: error: QOSMaxSubmitJobPerUserLimit` | The debug queue accepts 4 jobs per user | Wait for one to finish, then `scripts/leonardo_laion_validate_all.sh retry` |
| `merge` says `NOT merged` | Some chunks are still running, failed, or were never submitted | `retry`, wait, `merge` again |
| `Permission denied (publickey)` when connecting | Your Leonardo SSH certificate (about 12 hours) has expired | Renew it from your own computer with `step ssh login` |
