# Implementation status

The first milestone is implemented as an offline Python package with a bounded
mobile-robot supervisor. Pollard 1.6.0 supplies real execution governance;
the provider scores, sensors, and robot effects in the default demo are synthetic.

The original implementation passed 88 tests on Windows/Python 3.12.2. The six
demonstration scenarios produced one accepted simulated move, two requests for
more evidence, and three deferrals. Pollard verified all 26 nodes in the
resulting ledger without findings. Editable installation, wheel installation,
the documented example, history, and policy simulation were also verified.
Initial release validation passed 120 local tests, including version-policy
checks. Cross-platform GitHub CI exercises Linux and Windows installations.

Coverage includes action allowlists, parameter bounds, malformed/non-finite
outputs, units, request-batch accounting, inference failures, expiry at dispatch,
cancellation/deadline changes during validation, discarded late replies,
bounded outstanding workers, and history without actuator side effects.

The optional OpenJev adapter has 22 mapping/cache-loader tests using test doubles.
No model weights, real model inference, physical controller operation, or energy
measurement are implied by those tests. No selected checkpoint or optional model
runtime was available during implementation.

The resource budget counts logical provider batch attempts, not tokens, GPU
forward passes, or joules. Native inference may continue after timeout; late
results cannot dispatch. Records are single-process and do not provide
cross-file transactions or exactly-once actuator recovery.

Next: explicitly prepare and benchmark pinned 0.8B/4B model candidates against
the state-machine baseline on noisy/missing/stale observations. Measure latency,
decision errors and abstention, calibrate thresholds on held-out cases, and add
process cancellation before adapting the interface to devices. Keep measured
energy separate from TOML and other proxies.
