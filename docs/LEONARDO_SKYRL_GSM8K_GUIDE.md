# Guide: train Prelude-base on GSM8K with GRPO on Leonardo, and check the result

This guide takes you from a Leonardo account to a trained model and an honest reading of its scores. It trains
`openeurollm/prelude-base` (9B) with reinforcement learning (GRPO, using the SkyRL framework) on the GSM8K grade-school
math problems. Every step says what it does, what you should see, and what to do if it goes wrong. Background,
pinned versions and the results of the first run are in [leonardo-skyrl-gsm8k.md](leonardo-skyrl-gsm8k.md).

Run all commands on a Leonardo **login node**, from `$OELLM_WORK/oellm-rlvr`, unless a step says otherwise.

## What happens, in one paragraph

The model gets a math question and writes an answer, 8 times per question. Each answer gets a reward of 1 if the number
after `####` is right, else 0. GRPO then nudges the model toward the answers that scored better than the other 7. One
pass over the 7,473 training questions takes 116 steps of 64 questions. Every 10 steps the model answers all 1,319 test
questions, and those answers are saved so you can read them afterwards.

| Part | What | Time | Where it runs |
|---|---|---|---|
| A | One-time setup: container, software, model, data | about 1-2 hours, mostly waiting | internet-connected serial nodes |
| B | A 2-step test run | about 30 minutes | 1 GPU node, debug queue |
| C | The training run | about 5 hours | 1 GPU node (4 x A100) |
| D | Reading the results | 15 minutes | login node |

---

## Part A: one-time setup

### Step 1. Check your site settings and tools

```bash
echo "account=$OELLM_ACCOUNT  work=$OELLM_WORK  scratch=$OELLM_SCRATCH"
command -v uv || echo "uv is missing"
```

You should see three non-empty values and a path to `uv`.

- **Empty values:** set them in `~/.bashrc` as described in Step 1 of
  [LEONARDO_LAION_IMAGE_BUILD_GUIDE.md](LEONARDO_LAION_IMAGE_BUILD_GUIDE.md), then log out and back in.
- **uv missing:** install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`, then log out and back in. `uv`
  installs the Python packages in Step 4.

### Step 2. Get the code

```bash
cd $OELLM_WORK/oellm-rlvr 2>/dev/null || git clone https://github.com/BirgerMoell/oellm-rlvr.git $OELLM_WORK/oellm-rlvr
cd $OELLM_WORK/oellm-rlvr && git pull && mkdir -p logs
ls scripts/leonardo_skyrl_setup.sbatch scripts/leonardo_skyrl_gsm8k_grpo.sbatch scripts/skyrl_eval_summary.py
```

The `ls` should list all three files. If they are missing, the scripts are not on your branch yet; ask which branch holds
them and `git checkout` it.

**Always submit jobs from this folder.** The jobs write their logs to `logs/` here.

### Step 3. Get the container (skip if you already have it)

SkyRL needs a newer Linux than Leonardo's, so it runs inside a container. Check whether you have it:

```bash
ls -lh $OELLM_SCRATCH/oellm-rl/containers/vllm-openai-v0.32.0-x86_64-cu129.sif
```

If the file is missing, download it (about 1 hour):

```bash
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_vllm_container_pull.sbatch
```

Wait until `squeue -u $USER` no longer lists the `vllm-pull` job, then run the `ls` above again. You should see a file of
several GB.

### Step 4. Install SkyRL and prepare the model and data

One job does all of it: installs SkyRL v0.3.0 into a Python environment, downloads the model, the tokenizer and the chat
template, and converts GSM8K into the format SkyRL reads.

```bash
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_skyrl_setup.sbatch
```

It prints `Submitted batch job <id>`. Note the number. When `squeue -u $USER` no longer shows `skyrl-setup`, check the log:

```bash
tail -n 40 logs/<id>.skyrl-setup.out
```

What you should see:

- an import table with versions and no `IMPORT FAILED`: `torch 2.11.0+cu128`, `vllm 0.23.0`, `transformers 5.8.0`,
  `flash_attn 2.8.3`, `torch cuda 12.8`;
- the model's `SOURCE.json` with `"eos_token": "<|im_end|>"` and `"generation_eos_token_ids": [4, 2]`;
- `train 7473 rows` and `validation 1319 rows`;
- the last lines `skyrl_root=...`, `venv=...`, `model=...`, `data=...`.

If it goes wrong:

| Message in the log | What to do |
|---|---|
| `uv not found` | Step 1: install `uv`. |
| `no container at ...` | Step 3. |
| `Read-only file system ... /opt/uv/cache` | Your copy of the setup script is old; `git pull` and resubmit. |
| a download fails with `401` or `403` | Log in to Hugging Face on the login node (`hf auth login`) and resubmit. |
| any other failure | Resubmit once; the job skips work that is already done. |

---

## Part B: a 2-step test run (about 30 minutes)

This runs the whole training loop twice on 16 questions, then saves a checkpoint and evaluates on the test set. It
proves that generation, training, saving and evaluation all work before you spend 5 GPU hours.

### Step 5. Submit the test

```bash
MAX_STEPS=2 TRAIN_BATCH=16 EVAL_BEFORE_TRAIN=false RUN_NAME=smoke \
  sbatch -A $OELLM_ACCOUNT --qos=boost_qos_dbg --time=00:30:00 --export=ALL scripts/leonardo_skyrl_gsm8k_grpo.sbatch
```

### Step 6. Check the test

When the job has left `squeue`, look at the end of the log and the scores:

```bash
tail -n 3 logs/<id>.skyrl-gsm8k.out
grep -E "cuDriverGetVersion|eval/all/avg_score|reward/avg_raw_reward" logs/<id>.skyrl-gsm8k.out | tail -n 5
```

What you should see:

- the last line `skyrl_run=.../outputs/skyrl/smoke`;
- `cuDriverGetVersion 12090` (the container's newer CUDA driver is active);
- a training reward around 0.3 and an `eval/all/avg_score` around 0.4.

If it goes wrong:

| Message in the log | What to do |
|---|---|
| `CUDA out of memory` | Add `MICRO_BATCH=1` in front of the `sbatch` command, or `POLICY_CPU_OFFLOAD=true`. |
| an error about `PTX` or `unsupported toolchain` | Add `VLLM_CUDA_COMPAT=0`. |
| an error importing `flash_attn` | Add `FLASH_ATTN=false`. |
| `no staged model`, `no GSM8K data` or `no venv` | Part A, Step 4 did not finish; check its log. |
| `DUE TO TIME LIMIT` | The debug queue allows 30 minutes; resubmit. The test resumes from where it stopped. |

The test run's files are in `$OELLM_SCRATCH/oellm-rl/outputs/skyrl/smoke`. You can delete that folder afterwards.

---

## Part C: the training run (about 5 hours)

### Step 7. Submit

```bash
sbatch -A $OELLM_ACCOUNT --export=ALL scripts/leonardo_skyrl_gsm8k_grpo.sbatch
```

This trains for one pass over the data (116 steps). The run is called `prelude-base-gsm8k-grpo`. To run a variant without
overwriting it, put `RUN_NAME=<new name>` in front of `sbatch`, together with any setting you change, for example
`RUN_NAME=lr2e-6 LR=2e-6`. All settings are listed at the top of `scripts/leonardo_skyrl_gsm8k_grpo.sbatch`.

### Step 8. Follow the run

```bash
JOB=<id>
squeue -j $JOB -o "%i %T %M"
grep -E "eval/all/avg_score|reward/avg_raw_reward|trainer/global_step" logs/$JOB.skyrl-gsm8k.out | tail -n 9
```

Run it now and then. What healthy looks like:

- `trainer/global_step` goes up by about one every 2-3 minutes;
- `reward/avg_raw_reward` starts around 0.3 and climbs (it reached about 0.95 in the first run);
- `eval/all/avg_score` appears every 10 steps and goes up: the first run went 0.34 at step 0, 0.89 at step 10, 0.93 at the end.

Warning signs: the reward drops suddenly and stays low, or the job stops early. Look at the end of the log
(`tail -n 50 logs/$JOB.skyrl-gsm8k.out`) for the error.

**If the job stops** (time limit, node failure), submit exactly the same command again. It continues from the last saved
checkpoint (saved every 20 steps), so you lose at most 20 steps.

### Step 9. Check that it finished

```bash
sacct -j $JOB -X -o JobID,State,ExitCode,Elapsed
ls $OELLM_SCRATCH/oellm-rl/outputs/skyrl/prelude-base-gsm8k-grpo/exports
```

You should see `COMPLETED 0:0`, and folders `global_step_20` ... `global_step_116` plus `dumped_evals`.

---

## Part D: read the results

The trained model and the saved test answers are in the run folder:

```bash
RUN=$OELLM_SCRATCH/oellm-rl/outputs/skyrl/prelude-base-gsm8k-grpo
```

| Folder | What it is |
|---|---|
| `$RUN/exports/global_step_116/policy` | **The trained model.** Load it with transformers or vLLM. Saved in 32-bit floats (34 GB); load it as bfloat16. |
| `$RUN/exports/global_step_<N>/policy` | Earlier versions, every 20 steps. |
| `$RUN/exports/dumped_evals/global_step_<N>_evals` | Every test answer at step N, one line per question, plus a one-line summary. |
| `$RUN/checkpoints` | Restart state for training. Not a model you can load. |

### Step 10. The score table

```bash
python3 scripts/skyrl_eval_summary.py $RUN
```

One row per evaluation. The columns:

| Column | Meaning |
|---|---|
| `accuracy` | Share of the 1,319 test questions answered correctly. **This is the headline score.** |
| `correct` | The same, as a count. |
| `has_marker` | Answers that contain `####` at all. |
| `wrong_has_marker` | Wrong answers that do contain `####`: real mistakes, or a wrongly read number. |
| `no_marker` | Answers without `####`. They always score 0, whatever the reasoning. |
| `cut_off` | Answers stopped at the 1,024-token limit, usually because the model repeats itself. |
| `cut_off_correct` | Cut-off answers that still scored 1, because the reward reads only the first `####`. |
| `avg_tokens` | Mean answer length in tokens. |
| `ends_eos`, `ends_im_end` | How answers end: on `<eos>` or on `<|im_end|>`. |

Add `--csv $RUN/eval_summary.csv` to save the table for a spreadsheet or a plot.

### Step 11. Interpret the numbers

Do not quote the jump in `accuracy` as "the model got better at math" before doing these three checks.

1. **How much is format?** Compare step 0 with the last step. If `no_marker` falls from hundreds to 0, much of the gain
   is the model learning to write `#### <number>`. A fairer measure of math skill at step 0 is
   `correct / has_marker`. In the first run that was 453 / 552 = 82%, against 93% at the end, a real but much smaller gain.
2. **Are good scores hiding bad answers?** If `cut_off_correct` is above 0, some answers were right but then rambled
   until the length limit. Such a checkpoint scores well but is not usable. The final checkpoint should have `cut_off` 0.
   In the first run this happened to more than a third of the answers at steps 20 and 30, and was gone by step 50.
3. **Is a difference real?** With 1,319 questions the score is accurate to about ±0.007. A difference of 0.01 between two
   steps or two runs is noise. Compare runs only by gaps of several points.

Things this evaluation cannot tell you:

- **Whether the test questions were in the model's pretraining data.** A high score could partly be memory. Testing on
  new GSM8K-style questions (for example GSM-Plus or GSM-Symbolic) would show this.
- **How it compares with published GSM8K numbers.** Those usually use lm-eval-harness with 8 worked examples in the
  prompt; this evaluation uses SkyRL's own prompt and none.

### Step 12. Read some answers

Numbers hide a lot. Read a few answers:

```bash
python3 scripts/skyrl_eval_summary.py $RUN --wrong 116          # 10 wrong answers at step 116
python3 scripts/skyrl_eval_summary.py $RUN --wrong 0 --limit 3  # what went wrong before training
python3 scripts/skyrl_eval_summary.py $RUN --question 3         # one question at every step
```

For each wrong answer it shows the question, the right answer (`truth`), and the end of the model's answer. Look for:

- **Arithmetic or reasoning slips:** real mistakes; these are what more training should fix.
- **The right number written differently**, such as `1,000` against `1000` or `$18`: a scoring problem, not a model
  problem. Count them; the true accuracy is that much higher.
- **Repetition until the cut-off:** the model has not learned to stop.

### Step 13. Keep the model

Scratch is cleared after some weeks without use. Copy the model somewhere permanent if you want to keep it:

```bash
mkdir -p $OELLM_WORK/models
cp -r $RUN/exports/global_step_116/policy $OELLM_WORK/models/prelude-base-gsm8k-grpo-step116
```

The model ends its answers with `<eos>`, never `<|im_end|>`. The saved `generation_config.json` stops on both, so
transformers and vLLM stop correctly. Software that stops only on `<|im_end|>` will keep generating.
