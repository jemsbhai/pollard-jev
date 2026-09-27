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
The original offline milestone did not run model weights. On 2026-09-27 a separate
real 4B CUDA benchmark reused the existing Shellhacks environment/cache with no
downloads or installs. It completed 32 calibration cases, 32 held-out cases and
one warmup. Model proposals matched 12/20 held-out action labels; the calibrated
policy executed 11 correct simulated actions, no wrong actions, and abstained
on 21 cases. The state machine matched all 32 expected outcomes. Median model
end-to-end decision time was 0.534 seconds. See [benchmark details](BENCHMARKING.md).
These are synthetic feature cases; physical controllers and energy remain untested.

The benchmark and representation additions pass 206 offline tests. Version consistency, the
non-isolated wheel/source build and strict Twine checks passed without package
downloads. This work is unreleased; the package version remains `0.1`.

The resource budget counts logical provider batch attempts, not tokens, GPU
forward passes, or joules. Native inference may continue after timeout; late
results cannot dispatch. Records are single-process and do not provide
cross-file transactions or exactly-once actuator recovery.

The follow-up [representation experiment](REPRESENTATION_EXPERIMENT.md) tested
two development candidates, then compared fixed task rules against JSON on 64
fresh test and 60 challenge cases with separate calibration. Raw proposal
accuracy improved from 60% to 67.5% on the fresh test set and to 70% on challenges.
No wrong actions were accepted, but near-threshold/obstacle errors and one
observation-order inconsistency remain. JSON defaults were retained.

Next: improve numerical/priority interpretation on development data and reserve
new held-out and recorded-sensor cases for evaluation. The 0.8B model
was not found in the inspected local stores and was not downloaded, so that
comparison is pending. Calibration now uses only a separate calibration split;
the held-out split evaluates its selected policy. Add process cancellation
before adapting the interface to devices. Keep measured energy separate from
TOML and other proxies.
