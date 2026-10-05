# pollard-jev

A Python companion to Pollard that turns timestamped observations and permitted
choices into a governed decision, a bounded simulated action, and an outcome
record. The first milestone runs entirely offline after installation. It needs
no credentials, model weights, or GPU.

Current release: **0.13**, with reproducible local OpenJev benchmarks
and an opt-in numerical-boundary and action-priority input experiment. See the
[changelog](https://github.com/jemsbhai/pollard-jev/blob/main/CHANGELOG.md).

```powershell
python -m pip install --upgrade pollard-jev==0.13
pollard-jev demo --output artifacts
```

Published under the MIT license. The initial release is `0.1`; minor increments
are `+0.01`, major increments are `+0.10`, and releases at or above `1.0` require
the owner's explicit approval. See the
[release policy](https://github.com/jemsbhai/pollard-jev/blob/main/docs/RELEASING.md).

## Run on Windows PowerShell

Requires Python 3.11 or newer. Tested here on Python 3.12.2.

```powershell
git clone https://github.com/jemsbhai/pollard-jev.git
Set-Location pollard-jev
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pollard_jev demo --output artifacts
.\.venv\Scripts\python.exe -m pytest -q
```

Setup uses PyPI; the demo and tests make no network requests. The equivalent
installed entry point is `.\.venv\Scripts\pollard-jev.exe demo --output artifacts`.
Each demo writes uniquely named JSONL records and a Pollard SQLite ledger.

| Scenario | Synthetic proposal | Final policy | Simulated action |
| --- | --- | --- | --- |
| Fresh coherent observations | `continue` | accept | move 20 cm |
| Conflicting observations | `inspect` | need evidence | none |
| Missing evidence | `inspect` | need evidence | none |
| Evidence expires during inference | `continue` | defer | none |
| Request budget exhausted | none | defer | none |
| Provider failure | none | defer | none |

The stale scenario advances a virtual clock during inference so it is
reproducible. Separate tests exercise real waiting, timeout, and cancellation.
The state-machine baseline reads the same request at the final evaluation time.
It has no provider and therefore no provider failures or inference-request cost;
the table is a behavior comparison, not a performance benchmark.

## Typed inputs and outputs

```python
from datetime import datetime, timedelta, timezone
from pollard_jev.contracts import Observation
from pollard_jev.demo import example_request
from pollard_jev.loop import DecisionLoop
from pollard_jev.providers.fixture import FixtureProvider

now = datetime.now(timezone.utc)
range_reading = Observation(
    observation_id="range-42", source="front-range-sensor",
    feature="front_range_m", value=1.5, unit="m",
    observed_at=now, valid_until=now + timedelta(seconds=5),
)
# The demo request supplies all four required sensor features and four choices.
request = example_request(now, request_id="robot-42")
request = request.model_copy(update={
    "observations": (range_reading, *request.observations[1:]),
})
with DecisionLoop(FixtureProvider(), max_requests=2,
                  records_path="artifacts/decisions.jsonl",
                  pollard_path="artifacts/decisions.db") as loop:
    record = loop.decide(request)
    print(record.model_dump_json(indent=2))
```

Run the complete example with
`.\.venv\Scripts\python.exe examples\offline_loop.py`.
Unknown readings use `status="unknown", value=None`; absence of a required
feature is also explicit insufficient evidence. Datetimes must have timezones,
validity windows must be positive, and numeric values must be finite.

A successful record contains fields like these (excerpt):

```json
{
  "schema_version": "1",
  "provider_status": "ok",
  "provider_result": {
    "semantics": "synthetic_support",
    "scores": {"continue": 0.94, "inspect": 0.12, "recover": 0.12, "request_help": 0.12},
    "proposed_action": "continue",
    "parameters": {"distance_cm": 20},
    "evidence": "sufficient"
  },
  "policy": {"disposition": "accept", "reason": "supported_and_feasible"},
  "action": {"action": "continue", "status": "completed", "simulated": true},
  "budget_limit_requests": 2,
  "budget_spent_requests": 1
}
```

Full records include observation IDs, timestamps and units; the question and
allowed choices; provider identity, version and settings; policy configuration
and version; simulated skill version and observed state; timing and budget
fields; and Pollard root/model/action node IDs. Records round-trip through
validated Pydantic models. The JSONL is a convenient export; the same complete
record is also stored as a Pollard note.

## How the loop works

`DecisionProvider.infer(tuple[DecisionRequest, ...])` returns typed results.
The interface is general; this milestone's policy and skill set are specific to
the robot demonstration. Providers return data and receive no dispatcher.

Pollard **1.6.0** supplies the real model-call ledger, custom request budget,
registered action allowlist, integer argument bounds, refusal nodes, action
records, SQLite persistence, and integrity verification. The companion supplies
contracts, evidence/support policy, sensor validity checks, deadlines,
cancellation, simulation, and linked decision records. The older sibling
Pollard checkout was inspected but is not modified or used as a dependency.
See [the API inspection](https://github.com/jemsbhai/pollard-jev/blob/main/docs/POLLARD_API.md) for exact versions and boundaries.

The policy checks four engineered sensor features: range and camera clearance
in metres, battery percentage, and a `stuck` indicator of 0 or 1. Conflicting,
missing, duplicate, invalid, expired, and future-dated readings block action.
Support threshold 0.80, score margin 0.15, clearance 0.5 m, battery 10%, and
range disagreement 0.4 m are **demonstration settings**, configurable through
`PolicyConfig`. Required physical units cannot be relabeled to change their meaning.
The policy also checks feasibility separately from evidence support. Task
utility is represented by the caller's question and choice hypotheses; there
is no trained utility or physical success predictor here.

The bounded simulated skills are:

| Skill | Permitted integer parameter | Default |
| --- | --- | --- |
| `continue` | `distance_cm`: 1–50 | 20 |
| `inspect` | `samples`: 1–5 | 2 |
| `recover` | `distance_cm`: 1–20 | 10 |
| `request_help` | `retries`: 1 | 1 |

The request's selected parameter values must also match the provider proposal.
Acceptance never expands the caller's available actions. The registered
handler rechecks policy, freshness, cancellation, and the monotonic deadline
immediately before dispatch. A second action in the same batch requires new
observations, including when the first attempted action fails.

Inference runs in one daemon worker per loop. A timeout or cancellation discards
that worker's reply. Until it finishes, later calls return `busy`; this prevents
unbounded abandoned workers. Python cannot terminate arbitrary native model
compute. A timeout bounds response/dispatch eligibility, not GPU execution or
total process resource use. Cancellation does not undo an action already
dispatched. Actual controllers, watchdogs, immediate stopping, and motion limits
remain outside this supervisor.

The resource budget counts **logical provider batch attempts**: one
`infer(requests)` invocation costs one regardless of question count. Success,
malformed output, provider failure, timeout, and cancellation after admission
each consume one. Pre-cancelled, busy, and budget-refused calls consume zero.
Actions and audit notes consume zero model requests. This is not token, HTTP
request, measured energy, or estimated-joule accounting. An adapter may perform
multiple internal forward passes per logical batch. Budgets last for one loop
instance/run; creating a new loop creates a new budget.

Independent support/entailment scores remain independent; they are never
normalized into an action distribution. Categorical results are validated as
such only when explicitly declared. Neither score type proves physical success.
The fixture scores are synthetic and uncalibrated.

## History, policy simulation, and live reevaluation

These are separate operations:

```powershell
$recordFile = (Get-ChildItem artifacts\decisions-*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
.\.venv\Scripts\python.exe -m pollard_jev history $recordFile
.\.venv\Scripts\python.exe -m pollard_jev simulate-policy $recordFile
$ledgerFile = (Get-ChildItem artifacts\pollard-*.db | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
.\.venv\Scripts\pollard.exe verify $ledgerFile --json
```

`inspect_history(path)` only loads recorded events. `simulate_policy(record,
config, at=...)` evaluates recorded observations and provider output with a
specified policy/time. Neither has a dispatcher or a provider. Policy
simulation does not replay the resource ledger, cancellation state, or a new
physical trajectory. A changed action does not reveal what its consequences
would have been.

`live_reevaluate(record, loop)` is an explicit new provider invocation. It
consumes the loop's request budget, writes a new record linked to the original,
preserves the original timestamps, and always disables dispatch. It uses
whatever provider the caller explicitly put in that loop; with a fixture it
remains synthetic. Old evidence normally defers at today's time. Historical
records are never overwritten.

Credential fields, arbitrary malformed responses, and exception text are not
persisted. Provider authentication must remain inside the caller-owned client.
Do not put secrets in questions, observations, model identities, or settings:
these are intentional audit content, not automatically scrubbed free text.
The JSONL export and SQLite ledger are not a cross-file transaction. A crash
may leave an intent or action without the final JSONL record; use the ledger
for inspection. This is a single-process demonstrator, not crash recovery or
exactly-once actuator infrastructure.

## Optional real backend and next milestone

The implemented adapter targets `AlexWortega/openjev/qwen3.5-4b-nli-v2` at a
pinned commit. **Real 4B inference was verified on Windows/CUDA on 2026-09-27**,
reusing an existing local runtime and model cache without any downloads or
installs. The local runner records hashes and runtime versions, enforces offline
loading, and rejects input truncation. See [OpenJev setup and evidence](https://github.com/jemsbhai/pollard-jev/blob/main/docs/openjev.md)
and [benchmark method, command and results](https://github.com/jemsbhai/pollard-jev/blob/main/docs/BENCHMARKING.md).

Across 32 held-out synthetic cases, model proposals matched 12 of 20 expected
actions. Calibration-only threshold selection yielded 11 correct simulated
actions, no incorrect accepted actions, and 21 abstentions, with a 0.534-second
median end-to-end decision time. The state machine matched all expected outcomes;
this run does not establish a model advantage. Package policy defaults are
unchanged. Recorded-sensor validation and the 0.8B comparison remain outstanding.

The [follow-up input experiment](https://github.com/jemsbhai/pollard-jev/blob/main/docs/REPRESENTATION_EXPERIMENT.md) tested factual
text and fixed task-rule prompting. Plain text did not help on development cases.
Rules improved fresh-test proposal accuracy from 60% to 67.5%, but obstacle,
near-threshold and observation-order failures remain. Both candidates are opt-in;
JSON remains the default. The state machine still matched all expected outcomes.

Next: improve numerical/priority interpretation with new held-out validation
and recorded sensor data. Add process-level inference cancellation before
device integration. Measure energy with an actual meter when available; keep
TOML and other proxies separate from joules. Native sensor encoders, controller
connections, and MCU deployment are later work.

See [implementation status](https://github.com/jemsbhai/pollard-jev/blob/main/docs/IMPLEMENTATION.md) for verified behavior and remaining limitations.
