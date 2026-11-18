# Guide: check which LAION tasks work on Leonardo

This guide takes you from a folder of built sandbox images to a checked list of tasks that can be used for training. It continues
where [LEONARDO_LAION_IMAGE_BUILD_GUIDE.md](LEONARDO_LAION_IMAGE_BUILD_GUIDE.md) ends, for the dataset
`laion/rts-1k-rootless-apptainer`. Every step says what it does, what you should see, and what to do if it goes wrong. It
assumes you are logged in to a Leonardo **login node** and have done Part A of the build guide (site variables, code, Harbor,
dataset). The runbook calls this step "validation"; see section 5.2 of [LEONARDO_LAION_RUNBOOK.md](LEONARDO_LAION_RUNBOOK.md).

## What this checks, and why

A task is a small programming problem plus an automatic **checker** (the "verifier") that gives a reward of 1 (solved) or 0
(not solved). For training, both halves must be trustworthy. That is tested by running every task with two automatic agents
that need no language model:

| Agent | What it does | Must score | If it does not, the task is broken because... |
|---|---|---|---|
| **oracle** | runs the task's own reference solution | **1** | the task cannot be solved in this sandbox, or the checker rejects a correct answer |
| **nop** | does nothing at all | **0** | the checker hands out reward for free |

Each agent runs every task **3 times**, and a task is kept only if all 6 runs give the expected result. No GPU is used; the
sandboxes only need CPU.

Time needed: about 5 minutes for a first single-task test, then about 1.5 hours for all tasks (mostly waiting).

Expect about 3 in 4 of all tasks to end up usable: the first full run gave 741 of 984, the second 743 of 984.

---

## Part A: before you start (2 minutes)

### Step 1. Go to the code and check your images

```bash
cd $OELLM_WORK/oellm-rlvr
ls scripts/leonardo_laion_validate_all.sh                            # the script used in part C
ls $OELLM_SCRATCH/oellm-rl/sif-cache-x86_64/*.sif | wc -l            # images built
ls $OELLM_SCRATCH/oellm-rl/sif-cache-x86_64/*.sif.build.json | wc -l # images built successfully
squeue -u $USER -o "%i %j %T"                                        # your running jobs
```

What to look for:

- **Script:** if `leonardo_laion_validate_all.sh` is missing, your checkout is older than this guide; update it with `git pull`.
- **Images:** about 700 of the 801 possible. The two counts should be equal; a `.sif.build.json` file is written only after a
  successful build, and only those images are used.
- **Jobs:** no `laion-sif-*` build jobs should be running. Checking while images are still being built gives a smaller list.
  Unrelated jobs of yours are fine.

All commands below are run from `$OELLM_WORK/oellm-rlvr`, because the jobs write their logs to `logs/` there.

---

## Part B: try one task (optional, 5 minutes)

Doing one task first shows that the whole chain works (image, sandbox service, checker) before you spend an hour on all of them.

### Step 2. Pick a task and look at it

```bash
TASK=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_names"][0])' \
  $(ls $OELLM_SCRATCH/oellm-rl/sif-cache-x86_64/*.sif.build.json | head -1))
echo $TASK
head -5 $OELLM_SCRATCH/oellm-rl/data/laion-ot/rts-1k-rootless-apptainer/tasks/$TASK/instruction.md
```

This takes the first task whose image built and shows what it asks an agent to do. For example, one task asks for a Node.js
script that validates phone numbers with a single regular expression and writes its output to `/app/result.txt`.

Each task folder also holds `solution/` (the reference solution the oracle runs), `tests/` (the checker), and `environment/`
(the Dockerfile the image was built from).

### Step 3. Run it

```bash
echo $TASK > $OELLM_SCRATCH/oellm-rl/one-task.txt
DATASET=rts-1k-rootless-apptainer SUBSET=list TASK_LIST=$OELLM_SCRATCH/oellm-rl/one-task.txt ATTEMPTS=1 WORKERS_PER_NODE=2 \
  sbatch -A $OELLM_ACCOUNT --qos=boost_qos_dbg --time=00:30:00 --cpus-per-task=8 --mem=32G --export=ALL \
  scripts/leonardo_laion_validate.sbatch
```

What it does: starts a short job in the debug queue (`boost_qos_dbg`, which usually starts within minutes but allows at most
30 minutes). The job starts Harbor's sandbox service on the node, opens the task's image as a sandbox, runs the oracle once and
the nop agent once, runs the checker after each, and writes a summary.

Note the job number that `sbatch` prints. The job takes about 2.5 minutes once it is running.

### Step 4. Read the result

Replace `<jobid>` with the job number (type the number itself, without the `< >`):

```bash
squeue -j <jobid>                                     # no job listed (or "Invalid job id") once it has ended
tail -8 logs/<jobid>.laion-validate.out
```

A working task ends with:

```
{
  "expected_tasks": 1,
  "compatible": 1,
  "failed": 0,
  "attempts_per_arm": 1
}
validation_artifact=.../laion-validation/rts-1k-rootless-apptainer-list-<jobid>
```

`"compatible": 1` means the oracle scored 1 and the nop agent 0. To see the rewards themselves:

```bash
find $OELLM_SCRATCH/oellm-rl/laion-validation/*-<jobid>/jobs -name reward.txt | xargs grep .
```

You should see one path under `jobs/oracle/` ending in `:1` and one under `jobs/nop/` ending in `:0`.

Each run (a "trial") has its own folder, `jobs/<agent>/<task>__<random id>/`. It holds `result.json` (the summary of that run,
including any error), `verifier/` (the checker's output and `reward.txt`) and the agent's own logs. List one to look around:

```bash
ls -R $OELLM_SCRATCH/oellm-rl/laion-validation/*-<jobid>/jobs/oracle | head -30
```

The `.err` log will contain several `LiteLLM ... ConnectError` warnings. They are harmless: a library tries to download a price
list, the compute nodes have no internet, and it uses its built-in copy instead.

---

## Part C: check every task (about 1.5 hours)

### Step 5. Submit the checks

```bash
scripts/leonardo_laion_validate_all.sh submit
```

What it does:

1. Collects every task whose image built successfully (874 of the 984 in our run). Tasks without an image are written to
   `no-image.txt` and skipped.
2. Splits them into **chunks of 150 tasks**. A chunk of 200 needed the full 30 minutes, so 150 leaves a safety margin.
3. Submits one 30-minute debug job per chunk, each running every task 3 times with each agent, 16 sandboxes at a time.
4. Records everything in a **run folder**, `$OELLM_SCRATCH/oellm-rl/laion-validation/rts-1k-rootless-apptainer-full-<date>/`:
   the chunk lists (`chunk-00.txt`, ...), `jobs.txt` (which job checks which chunk) and `no-image.txt`.

Expected output, for 874 tasks:

```
984 tasks: 874 with an image, 110 without (see no-image.txt); 6 chunks of up to 150
run folder: .../laion-validation/rts-1k-rootless-apptainer-full-20261118-102455
64733260 chunk-00.txt
64733262 chunk-01.txt
64733263 chunk-02.txt
64733264 chunk-03.txt
sbatch: error: QOSMaxSubmitJobPerUserLimit
STOP: the queue refused chunk-04.txt (the debug queue accepts 4 jobs per user; this is expected with more chunks).
When a job has finished, run: scripts/leonardo_laion_validate_all.sh retry
```

**The `STOP` is expected.** The debug queue accepts at most **4 jobs per user**, so the last chunks have to wait. Step 7 submits
them.

### Step 6. Follow the progress

```bash
squeue -u $USER -o "%i %j %T %M %r"
```

You will see something like:

```
JOBID     NAME            STATE    TIME  REASON
64733263  laion-validate  PENDING  0:00  QOSMaxJobsPerUserLimit
64733264  laion-validate  PENDING  0:00  QOSMaxJobsPerUserLimit
64733260  laion-validate  RUNNING  12:40 None
64733262  laion-validate  RUNNING  12:40 None
```

The debug queue **runs only 2 jobs per user at a time**. `QOSMaxJobsPerUserLimit` means "waiting for one of your own jobs to
end"; it is not an error, and those jobs start by themselves.

To see how far a running job is, count its finished runs. Each task needs 6 (3 oracle and 3 nop), so a chunk of 150 is done at
900:

```bash
find $OELLM_SCRATCH/oellm-rl/laion-validation/*-<jobid>/jobs -name result.json | wc -l
```

(The count includes a few summary files, so it ends slightly above 900.)

### Step 7. Submit the rest

When at least one job has ended (it no longer appears in `squeue`):

```bash
scripts/leonardo_laion_validate_all.sh retry
```

`retry` looks at every chunk and submits only those that need it:

| Chunk state | What `retry` does |
|---|---|
| has results | skips it |
| job still queued or running | skips it |
| never submitted (refused by the queue) | submits it |
| job ended **without** results (for example it ran out of time) | submits it again |

It is always safe to run, so run it whenever you check on progress until it prints `nothing left to submit`. It may stop with
the same `STOP` message as in step 5; that only means the queue is full again.

### Step 8. Collect the results

When `squeue -u $USER` shows no more `laion-validate` jobs:

```bash
scripts/leonardo_laion_validate_all.sh merge
```

Expected output (from a run with 200-task chunks):

```
  job 64733260 (chunk-00.txt): 174 of 200 usable
  job 64733262 (chunk-01.txt): 171 of 200 usable
  job 64733263 (chunk-02.txt): 174 of 200 usable
  job 64733264 (chunk-03.txt): 171 of 200 usable
  job 64750889 (chunk-04.txt): 53 of 74 usable
usable: 743 of 874 tested tasks (110 more had no image)
wrote .../laion-validation/rts-1k-rootless-apptainer-leonardo-compatible-all.jsonl
why the others failed: .../rts-1k-rootless-apptainer-full-<date>/failures.json
```

If it says **`NOT merged`**, some chunks have no results yet, and it lists which. Nothing is written in that case, so you never
get an incomplete list by accident. Run `retry`, wait for the jobs, and run `merge` again.

---

## Part D: understand the result

### The usable list

```
$OELLM_SCRATCH/oellm-rl/laion-validation/rts-1k-rootless-apptainer-leonardo-compatible-all.jsonl
```

One line per usable task, such as `{"task_id": "rts_task_3b5b346aff717426ef550d27"}`. This is the file the rollout job (running
a real model on the tasks) reads by default. If you run `merge` again later, it replaces this file.

### Why tasks were rejected

`failures.json` in the run folder lists every rejected task with a reason for each agent (`null` means that agent behaved
correctly). Count the reasons:

```bash
RUN=$(ls -dt $OELLM_SCRATCH/oellm-rl/laion-validation/rts-1k-rootless-apptainer-full-* | head -1)
python3 -c 'import json,sys,collections
c = collections.Counter("oracle: %s | nop: %s" % (f["oracle"], f["nop"]) for f in json.load(open(sys.argv[1])))
for k, n in c.most_common(): print(n, k)' $RUN/failures.json
```

What the reasons mean:

| Reason | Meaning | Worth a second try? |
|---|---|---|
| `oracle: reward=0.0` (or `reward=None`) | the reference solution ran but the checker said it failed, or gave no reward | rarely; usually the task or its solution is broken |
| `nop: reward=1.0` | the checker passed a run that did nothing | no; the checker is too lenient |
| `exception` | a run crashed (sandbox, timeout, harness) instead of being scored | **yes**; often a temporary problem on the node |
| `only 2/3 trials` | the job ended before all runs of this task finished | **yes**; it ran out of time |

To see what went wrong for one task, open its trial folders in the chunk's results. Find the chunk's job number in
`$RUN/jobs.txt` and the task in `$RUN/chunk-*.txt`:

```bash
grep -l <task_id> $RUN/chunk-*.txt                        # which chunk
cat $RUN/jobs.txt                                         # which job checked that chunk
ls $OELLM_SCRATCH/oellm-rl/laion-validation/*-<jobid>/jobs/oracle/ | grep <task_id>
```

`result.json` in a trial folder shows the reward and, for an `exception`, the error message.

### Optional: give crashed tasks a second chance

If many rejections are `exception` or `only n/3 trials`, run those tasks again:

```bash
python3 -c 'import json,sys
for f in json.load(open(sys.argv[1])):
    if "exception" in (f["oracle"], f["nop"]) or "only" in "%s%s" % (f["oracle"], f["nop"]): print(f["task_id"])' \
  $RUN/failures.json > $OELLM_SCRATCH/oellm-rl/second-chance.txt
wc -l < $OELLM_SCRATCH/oellm-rl/second-chance.txt          # keep it at 150 or fewer for one debug job
DATASET=rts-1k-rootless-apptainer SUBSET=list TASK_LIST=$OELLM_SCRATCH/oellm-rl/second-chance.txt ATTEMPTS=3 \
  sbatch -A $OELLM_ACCOUNT --qos=boost_qos_dbg --time=00:30:00 --export=ALL scripts/leonardo_laion_validate.sbatch
```

When it has ended, add the tasks that now pass to the usable list:

```bash
cat $OELLM_SCRATCH/oellm-rl/laion-validation/*-<jobid>/leonardo-compatible.jsonl \
  >> $OELLM_SCRATCH/oellm-rl/laion-validation/rts-1k-rootless-apptainer-leonardo-compatible-all.jsonl
```

Do not re-run tasks rejected for `reward=...`: those failures are in the task itself and repeat every time.

### What comes next

The tasks are ready for a model. Running a real model on them (and finding the tasks it sometimes solves, which are the useful
ones for reinforcement learning) is section 5.3 of the [runbook](LEONARDO_LAION_RUNBOOK.md). It needs a downloaded model
checkpoint and four GPUs.

---

## Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| `sbatch: error: QOSMaxSubmitJobPerUserLimit` | The debug queue accepts 4 jobs per user | Expected. Wait for a job to end, then `retry` |
| Jobs `PENDING` with reason `QOSMaxJobsPerUserLimit` | The debug queue runs 2 jobs per user at a time | Nothing; they start on their own |
| Jobs `PENDING` with reason `Priority` | The cluster is busy | Wait |
| `merge` says `NOT merged` | Some chunks are still running, ended without results, or were never submitted | `retry`, wait, `merge` again |
| `retry` resubmits a chunk "ended without results" | Its job failed or ran out of time | Look at `logs/<jobid>.laion-validate.out` and `.err`; if it ran out of time, use smaller chunks (`CHUNK=100 scripts/leonardo_laion_validate_all.sh submit` starts a new run) |
| `LiteLLM ... ConnectError ... model_prices_and_context_window.json` in `.err` | A library cannot reach the internet from a compute node | Harmless |
| `bridge did not start` in a job log | Harbor's sandbox service failed to start on the node | Read `service-logs/server.log` in the job's results folder; resubmit with `retry` |
| `bash: syntax error near unexpected token` with `<jobid>` | The placeholder was typed literally | Replace `<jobid>` with the number, without `< >` |
| Paths printed as `/oellm-rl/...` | The `OELLM_*` variables are not set | Step 1 of the build guide |
| `no tasks at ...` | The dataset has not been downloaded | Step 4 of the build guide |
| `0 with an image` | No image has finished building | Finish the build guide first |
