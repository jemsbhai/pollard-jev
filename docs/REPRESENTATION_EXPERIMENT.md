# Input representation and task-rule experiment

This follow-up investigates the 4B model's low-battery and obstacle errors from
the [first benchmark](BENCHMARKING.md). It reuses an existing local Python runtime
and pinned checkpoint entirely offline. No weights or packages were downloaded,
installed or trained. These runs used a development checkout reporting package
version `0.1`; historical provenance retains that value. The benchmark tools and
input variants are included in release `0.11`. JSON remains the default.

The rules candidate produced a small held-out gain, but remains experimental:
obstacle and near-threshold errors persist, and one matched observation-order
pair changes its decision. The state machine still matches every expected
outcome. This experiment does not establish a model advantage.

## Development decisions

Two candidates were tested on the original seed `20260927` calibration cases,
now treated as exploratory development data. Neither candidate was selected
using the new evaluation sets.

| Development input | Correct proposals / 20 eligible | Correct accepted actions at default thresholds | Incorrect accepted actions |
| --- | --- | --- | --- |
| Original `json-v1` | 12 | 7 | 0 |
| Factual `text-v1` | 12 | 0 | 0 |
| Fixed task rules + facts, `robot-rules-v1` | 12 | 12 | 0 |

Factual text preserves each observation's exact value, unit, status, source,
identifier and timestamps, in the original order. It expands feature names but
does not compute freshness, feasibility, thresholds or an action. Unknown and
duplicate readings are retained, and strings are quoted to preserve embedded
line breaks. JSON rendering is byte-for-byte identical to the original adapter.

Plain text did not repair the wrong proposals and reduced confidence. The
second candidate prefixes the same factual text with fixed demonstration rules:
battery below 10% means request assistance; otherwise a stuck robot should
recover; otherwise clearance below 0.5 m calls for inspection; otherwise the
robot may continue. The prefix is identical for every request. It never calls
the baseline or policy or consults an expected label to construct the input.

**The rules candidate supplies additional task information.** It is not a
formatting-only improvement or evidence that the model discovered these rules.
It was selected for fresh evaluation because it accepted more correct development
actions at the unchanged default thresholds, despite unchanged proposal accuracy.
The original errors were still present in development.

## Frozen evaluation protocol

The candidate code and its hash were recorded before new evaluation inference.
Fresh seed `20260928` supplies 64 calibration and 64 test cases (eight per original
scenario in each split). JSON and the candidate receive identical requests,
hypotheses and choice order, using the same loaded encoder, bfloat16 weights,
CUDA device and 2,048-token limit. Execution order alternates per paired case;
each variant receives a separate warmup.

Each variant selects support and margin independently from its calibration
outputs, with the original grid and selection objective. Selected policies are
saved before test inference. Default-policy replays over those saved test outputs
separate input effects from threshold selection; replay never calls the model or
dispatches. Neither the test cases nor the challenge cases tune the renderer or
thresholds. Evidence clocks remain fixed per case; timings use real wall time.

A separate seed `20260929` generates 60 challenge cases across 15 strata:
clearance and battery just below/at/above thresholds; simultaneous low-battery
and stuck states; obstacles while stuck; matched original/permuted observation
order; unknown and duplicate observations; future observation/request times;
and excluded preferred actions. Matched order pairs share complete observation
objects, including IDs, with only their order changed in the premise.

Challenge labels are explicit synthetic task preferences, not physical outcomes.
In particular, excluded-preferred cases require abstention instead of substituting
another permitted skill. That requirement is stricter than the current policy,
which can permit an alternative such as inspection. Errors on that stratum
identify a design limitation, not necessarily an unsafe action. Exact numeric
boundaries likewise test the specified demo rules, not general language quality.

## Results: 2026-09-27

Both variants were evaluated once with frozen policies. Calibration selected
support/margin **0.65/0.05** for JSON and **0.80/0.05** for the rules candidate.
Only confidence settings changed; all physical feasibility and evidence checks
remained the original demo policy.

| Measure | JSON | Rules + facts |
| --- | --- | --- |
| Fresh test proposals correct, action-eligible cases | 24/40 (60%) | 27/40 (67.5%) |
| Fresh test correct accepted actions | 24/64 cases | 26/64 cases |
| Fresh test incorrect accepted actions | 0 | 0 |
| Fresh test required abstentions respected | 24/24 | 24/24 |
| Fresh test total abstentions | 40/64 | 38/64 |
| Fresh test median / p95 decision seconds | 0.616 / 0.718 | 0.636 / 0.716 |
| Challenge proposals correct, action-eligible cases | 24/40 (60%) | 28/40 (70%) |
| Challenge correct accepted actions | 16/60 cases | 27/60 cases |
| Challenge incorrect accepted actions | 0 | 0 |
| Challenge required abstentions respected | 20/20 | 20/20 |
| Challenge total abstentions | 44/60 | 33/60 |

The candidate corrected three fresh-test proposals without proposal regressions.
All three were low-battery cases; two passed its selected confidence gate and
one abstained. It still proposed `continue` for all eight original-template
obstacle cases and five of eight low-battery cases. The existing policy blocked
those proposals. Required-abstention success reflects the deterministic policy;
the model is not credited with recognizing missing or invalid evidence.

On the harder challenges, the candidate corrected four proposals without paired
proposal regressions: three low-battery/stuck priority cases and one reordered
obstacle case. It still missed all four just-below-battery and all four
just-below-clearance boundaries. The state machine matched all 64 fresh test
and all 60 challenge labels.

The order check is a material limitation: on one of four matched pairs, the
candidate proposed `continue` (blocked) in the original order and `inspect`
(accepted) after rotation of the **same complete readings**. JSON's proposals
were unchanged on all four pairs. Thus aggregate gains do not establish robust
interpretation of the observations.

Replaying unchanged default support/margin **0.80/0.15** on saved model outputs
would accept 15 correct fresh-test actions for JSON versus 25 for rules, and
8 versus 24 on challenges, with zero wrong acceptances in both sets. These
replays perform no new inference or dispatch; the table above reports actual
simulated actions using each calibrated policy.

All 378 inference attempts succeeded (376 paired calibration/test/challenge
calls plus two warmups). The largest complete NLI pair was 887 tokens; no
truncation occurred. Peak PyTorch GPU allocation was 9.59 GB, with 9.94 GB
reserved. The model, tokenizer, helper and configuration hashes match the
original benchmark. All six evaluation ledgers passed integrity verification,
covering 1,768 nodes without findings. The full offline suite passes 200 tests.

The [result snapshot](benchmarks/2026-09-27-rules-comparison.json) includes both
development attempts, per-scenario metrics, every evaluated proposal and score,
selected policies, provenance and artifact hashes. Complete local artifacts
are preserved in `artifacts/benchmarks/2026-09-27-rules-evaluation/`; the
development directories end in `representation-development` and
`rules-development`. These are additional results; the first benchmark was
not replaced or reinterpreted as an unseen test set.
Machine-specific executable paths are omitted from published result snapshots;
model hashes, runtime versions and measurements are retained.

## Reproduction without downloads

Run from this checkout using an existing environment and cache. Replace the
illustrative paths below with the paths to your existing environment and cache:

```powershell
$modelPython = 'C:\path\to\openjev-env\Scripts\python.exe'
$existingCache = 'C:\path\to\openjev-cache'
$resultDirectory = 'artifacts/benchmarks/rules-comparison-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $modelPython scripts/compare_representations.py --cache-dir $existingCache --output $resultDirectory --stage evaluation --candidate robot-rules-v1 --cases-per-scenario 8
```

`--stage development` restricts execution to the original calibration cases;
`--candidate text-v1` reproduces the formatting-only alternative. Existing
result directories are never overwritten. The command records a protocol,
corpus fingerprints, per-variant records/ledgers, calibration grids, model and
source hashes, environment versions, token lengths and summaries. It uses the
same offline flags and blocked Python socket connections as the first runner.

The general local runner also accepts `--representation json-v1`, `text-v1` or
`robot-rules-v1`; `json-v1` remains the default. Provider identities record
`settings.input_representation`, so outputs from different variants remain
distinguishable in the audit record.

## Scope of the experimental rules

`robot-rules-v1` describes the **fixed default demonstration physical rules**.
It does not track a custom `PolicyConfig`. It also does not encode the complete
evidence policy, including every duplicate, missing, future, conflict-tolerance
or excluded-choice check. Required abstentions enforced by the deterministic
policy must not be attributed to model recognition. Confidence calibration does
not establish physical success probabilities.

All observations are synthetic engineered features. Independent draws from the
same templates and the additional challenges do not replace recorded sensor
data, device testing or energy measurement. The 0.8B comparison and process-level
inference termination remain separate unfinished roadmap items.

## Numerical and priority development candidate: 2026-10-05

The opt-in `robot-rules-v2` supplies explicit strict/inclusive comparisons,
equality examples, and a first-matching-rule procedure. Low battery takes
priority over stuck and obstacles; stuck takes priority over obstacles. It
adds a fixed statement that observation order does not change rule priority.
The prefix is identical for every request, and the factual suffix remains
identical to `text-v1`, including reading order, duplicates and metadata.
It computes no comparisons, validity checks or preferred action. Like v1,
it supplies task information and describes only default demonstration rules.

A paired offline run reused the pinned cached 4B model on the original 32
calibration cases, treated exclusively as previously inspected development
data. No fresh test cases were used, no confidence thresholds were selected,
and package defaults remain unchanged.

| Development measure | JSON | Rules v2 |
| --- | --- | --- |
| Correct proposals / 20 eligible cases | 12 | 20 |
| Correct accepted simulated actions, default policy | 7 | 10 |
| Incorrect accepted actions | 0 | 0 |
| Required abstentions respected | 12/12 | 12/12 |

V2 corrected eight paired proposals with no proposal regressions. This is
exploratory development evidence, not generalization evidence. The earlier v1
development run accepted 12 correct actions, versus v2's 10, so improved raw
proposal accuracy does not establish improved governed coverage. The state
machine again matched every label. This run does not specifically evaluate
near-threshold, simultaneous-priority or matched-order challenges; robustness
on those strata remains unverified.

The [development snapshot](benchmarks/2026-10-05-rules-v2-development.json)
preserves metrics, runtime/source and model provenance, and local artifact
hashes without machine-specific executable paths. Complete records, ledgers
and input cases remain in the ignored directory
`artifacts/benchmarks/2026-10-05-rules-v2-development/`.

Reproduce with the existing environment/cache pattern above, a new output
directory, `--stage development --candidate robot-rules-v2`. The comparison
runner rejects v2 with `--stage evaluation`: its existing evaluation seeds
have already been inspected. A subsequent evaluation needs a newly reserved
corpus and a frozen protocol with independent calibration. The general local
runner also accepts v2; its default corpus is historical development data
and must not be presented as fresh evaluation.

The immediate modeling limitation is reliable numerical/priority interpretation,
not merely JSON formatting. Further input experiments must reserve new test
cases; these inspected sets are now diagnostic evidence. Process-level worker
termination can proceed independently before any device integration.
