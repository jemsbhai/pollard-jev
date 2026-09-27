# Changelog

## 0.11 — 2026-09-27

Reproducible local OpenJev benchmarks and opt-in input experiments, with the
offline demo and original JSON input retained as defaults.

- Opt-in versioned factual-text and fixed robot-rule input variants; original
  JSON rendering stays the default and representation is recorded in identity.
- Paired fresh evaluation and 60 synthetic challenge cases cover boundaries,
  combined conditions, observation order and invalid evidence. Rules improve
  proposal accuracy modestly but retain obstacle/boundary/order failures;
  see `docs/REPRESENTATION_EXPERIMENT.md` for full results and limitations.
- Reproducible synthetic benchmark with disjoint calibration/test splits,
  proposal errors, policy coverage, abstention, latency and state-machine comparison.
- Local-only benchmark runner reusing an existing Python environment and pinned
  cache, with runtime/model/source provenance and input-truncation checks.
- Input representation is read-only after construction to keep rendered inputs
  consistent with audit identity; missing Git no longer blocks local benchmarks.
- First real 4B CUDA run on 2026-09-27: 64 measured cases, 60% held-out proposal
  accuracy on action-eligible cases, 11 correct accepted actions and no wrong
  accepted actions. Full limitations and reproduction command in `docs/BENCHMARKING.md`.

## 0.1 — 2026-09-23

Initial public release of the offline robotics/IoT decision companion.

- Validated observations, permitted choices, provider results, policies, and outcomes.
- Pollard 1.6.0 integration for request budgets, registered actions, bounds, and audit records.
- Dispatch-time freshness, timeout and cancellation checks; late results cannot execute actions.
- Synthetic fixture provider, bounded robot simulator, and state-machine baseline.
- Six offline demonstration scenarios, JSONL/SQLite records, read-only history,
  policy simulation, and explicit reevaluation without dispatch.
- Optional OpenJev NLI adapter with independent score semantics and a pinned local-cache loader.
- Release tooling for decimal minor (+0.01) and major (+0.10) increments, with
  versions at or above 1.0 prohibited until the project owner explicitly approves.

Model inference and physical hardware operation are outside this release's
verification. Optional backend tests validate mapping with test doubles.
