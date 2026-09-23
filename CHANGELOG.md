# Changelog

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
