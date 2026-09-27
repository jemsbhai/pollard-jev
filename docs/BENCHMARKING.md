# Local OpenJev benchmark

On 2026-09-27 the pinned **4B v2 model ran successfully on the RTX 4090 Laptop
GPU**, reusing the Shellhacks project's existing Python environment and model
cache. No packages or model files were downloaded or installed. This completes
the first real-model run and a small synthetic benchmark; the 0.8B comparison,
recorded-sensor evaluation and device integration remain outstanding.

Release `0.11` includes the benchmark modules and documentation. The runner
scripts are included in the source distribution and repository; use a checkout
of tag `v0.11` or an extracted source archive to run the commands below.
Historical result snapshots retain the development package version used during
inference.

## Run again without downloads

From the pollard-jev repository in PowerShell, select an **existing** Python
environment containing Pollard and the optional OpenJev dependencies, and an
**existing** cache containing the pinned checkpoint. This machine's working paths:

```powershell
$modelPython = 'E:\data\code\hackathons\shellhacks2026\.venv\Scripts\python.exe'
$existingCache = 'E:\data\code\hackathons\shellhacks2026\.state\openjev-cache'
$resultDirectory = 'artifacts/benchmarks/local4b-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $modelPython scripts/benchmark_local.py --cache-dir $existingCache --output $resultDirectory --device cuda --cases-per-scenario 4
```

The script uses this checkout's source without installing it into or changing
the other project's environment. The output directory must be new or empty.
It sets Hugging Face/Transformers offline flags, disables implicit tokens and
telemetry, blocks Python socket connections, and uses `local_files_only=True`.
Missing dependencies or cached files cause failure; there is no download or
installation fallback. The loader uses the inspected local helper, whose file
hash is recorded with hashes for the config, tokenizer and full weights.

The run saves environment versions, source hashes, model hashes, the complete
corpus and its fingerprint, calibration grid, individual decisions, JSONL
exports, Pollard SQLite ledgers, context measurements and a summary. The
environment snapshot records package versions only, not credential-bearing
package source URLs. GPU load, warmup and steady-state decision timings are
separate. `ContextCheckedEncoder` measures complete unpadded input pairs and
rejects context overflow before the upstream helper can truncate them.

## Method

Default seed `20260927` produces 32 calibration and 32 held-out cases: four
independently sampled examples of each of clear, obstacle, low battery, stuck,
noisy coherent, conflicting, missing and stale evidence. The model sees opaque
identifiers, structured observations and the four existing action hypotheses;
it does not see scenario names, expected labels or split names. Expected labels
are assigned by the scenario specification before sampling, independently of
the baseline implementation.

The calibration grid searches support thresholds
`0, 0.25, 0.5, 0.65, 0.8, 0.9, 1` and margins `0, 0.05, 0.15, 0.3, 1`.
Selection minimizes incorrect accepted actions, then maximizes correct accepted
actions; ties prefer larger thresholds and margins. Only calibration outputs
are used. An all-abstaining policy has zero coverage and no accepted-action
accuracy, rather than being described as perfectly accurate.

Each case uses a fresh simulator and Pollard loop. Evidence clocks are frozen
at the case's evaluation time; inference deadlines and measured durations use
real wall time. This permits deterministic stale-evidence cases but does not
test evidence aging during real model computation. Existing loop tests cover
expiry during inference and at dispatch. The inference timeout is 120 seconds;
the benchmark stops on a timeout instead of accumulating abandoned workers.
The loop still uses a thread and cannot forcibly stop native computation.

## Measured results: 2026-09-27

Runtime: Python 3.12.2, PyTorch `2.11.0+cu128`, Transformers `5.15.0`, Hugging
Face Hub `1.33.0`, bfloat16, CUDA 12.8, NVIDIA driver 595.79. Model:
`AlexWortega/openjev/qwen3.5-4b-nli-v2`, revision
`552759daad712f1af6c4c13dabcb1e047886fc9c`, maximum input length 2,048 tokens.

The grid selected support **0.65** and margin **0.05** using calibration cases.
These are exploratory benchmark settings; the package defaults remain
support **0.80** and margin **0.15**.

| Held-out measure | Result |
| --- | --- |
| Model proposal agrees with expected action | 12 / 20 action-eligible cases (60%) |
| Calibrated policy accepts correct action | 11 / 20 action-eligible cases (55%) |
| Incorrect accepted actions | 0 / 32 cases |
| Required abstentions respected | 12 / 12 cases |
| Additional abstentions on action-eligible cases | 9 / 20 cases |
| Total abstention | 21 / 32 cases (65.6%) |
| Exact policy outcome agreement | 23 / 32 cases (71.9%) |
| State-machine outcome agreement | 32 / 32 cases (100%) |
| Median / p95 end-to-end decision time | 0.534 / 0.636 seconds |

The default policy, evaluated on the same saved held-out model outputs without
new inference or dispatch, would accept 9 correct actions and no incorrect
actions (45% action-eligible coverage). The calibrated policy executed 11
bounded simulated actions. Neither setting was chosen using held-out results.

All four clear and four noisy coherent cases executed `continue`. Three of four
stuck cases executed `recover`; the fourth deferred for insufficient support.
The model incorrectly proposed `continue` for all four low-battery and all four
obstacle cases; the support and feasibility checks blocked them. All conflicting,
missing and stale cases abstained as specified. Model proposal quality and
governed execution quality are therefore different measurements.

Model construction took 9.63 seconds in this run, after runtime imports and
file hashing. A separate warmup took 0.868 seconds end to end. Peak PyTorch GPU
allocation was 9.56 GB (8.90 GiB), with 9.73 GB reserved; this is allocator
measurement, not whole-system memory. All 65 inference attempts succeeded
(one warmup plus 64 measured cases). The largest complete NLI pair was 588
tokens; no input was truncated. Pollard verified 148 calibration-ledger nodes
and 150 held-out-ledger nodes without findings.

Baseline decision timing was roughly 0.015 milliseconds median. That measures
the state-machine decision only, while model end-to-end timing includes
inference, policy, simulated dispatch when accepted, and audit recording. The
timings should not be described as equivalent execution paths or kernel timing.

The [repository result snapshot](benchmarks/2026-09-27-4b.json) contains metrics,
all held-out decisions and scores, runtime/source provenance, model hashes and
local artifact hashes. Complete local artifacts are under
`artifacts/benchmarks/2026-09-27-4b/` (ignored by Git).

## Interpretation and next work

This run establishes that the existing pinned model/runtime work locally and
that the policy contains the observed model errors in this small experiment.
It does **not** establish an advantage over the state machine: the baseline
matched every expected outcome, and its rules encode this synthetic task well.
The two splits use independent values from the same eight scenario templates;
they are not an external dataset. No physical success, raw-sensor understanding,
energy savings, statistical safety guarantee or generalization claim follows.

Next, address the low-battery/obstacle failure modes using development examples
and a separately versioned input representation, then evaluate on a new held-out
set and recorded sensor cases. Preserve these results as the original baseline.
Compare 0.8B when an existing local checkpoint is located; none was found in the
inspected stores and none was downloaded. Add process-level inference termination
before device integration. Measured energy remains future work.

The subsequent [representation and task-rule experiment](REPRESENTATION_EXPERIMENT.md)
preserves this original result and compares opt-in candidates on fresh cases.
