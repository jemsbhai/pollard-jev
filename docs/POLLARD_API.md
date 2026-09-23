# Pollard API inspection

Inspected on 2026-09-23. This milestone targets the published **Pollard 1.6.0**
package, inspected after installation in this project's virtual environment.
`pollard.__version__`, distribution metadata, live Python signatures, and installed
source all identify that version. Reference: [PyPI release](https://pypi.org/project/pollard/1.6.0/)
and [upstream project](https://github.com/jemsbhai/pollard).

An existing sibling Pollard checkout was also read
without modification. It had clean Git status, commit
`ddd4b85ea77b4cc46bbf5fb9432efd9883123557`, and declared version **0.4.0** in
`pyproject.toml` and `src/pollard/__init__.py`. Its older registry does not support
integer bounds. Do not assume that checkout is interchangeable with the pinned
published dependency.

## Verified integration surface

These are actual public APIs, observed in 1.6.0's `runtime.py`, `registry.py`,
`governor.py`, `meters/__init__.py`, `stores/sqlite.py`, `replay.py`, and `verify.py`:

- `Runtime(store=None, *, meters=None, registry=None, policies=None, dry_run=False,
  mode="record", refuse_duplicate_recordings=False, ...)` accepts an in-memory
  store, a `SQLiteStore`, or a SQLite path.
- `runtime.run(label, budget=Budget(...))` returns a `Run` with a root and cursor.
- `run.model_call(payload, fn=callable)` calls `callable(payload)` and returns a
  `Node`. The provider response is in `node.result`.
- `Registry([ActionSpec(name, version, description, schema, side_effects,
  handler)])` defines permitted tools. `run.tool_call(name, args, version=...)`
  validates the registered schema and invokes `handler(args)`.
- `run.note(payload)` appends application policy/outcome evidence.
- `run.report()["spent"]` recomputes recorded charges; `Budget(extra={...})`
  applies named custom meter limits.
- `BudgetExceeded` and `PolicyViolation` expose `refusal_id`; Pollard records a
  `refusal` node before raising either exception. There is no public
  `Run.refuse()` method. Application deferrals can be explicit note records.
- `store.walk(root_id)` and `store.get(node_id)` inspect persisted nodes.
  `SQLiteStore(path, read_only=True)` supports historical access without writes.
  `verify(store, node_id).ok` checks the node's recorded ancestry.

The public custom meter protocol is `name`,
`precheck_estimate(node_kind, payload)`, and
`charge(node_kind, payload, result, meta)`. A request meter returns one for a
`model_call` and zero for other node kinds in both methods. One `model_call`
containing several questions is therefore one provider batch request. Question
count and request count must remain distinct. This is an exact request ledger,
not token accounting or measured energy.

## Integration constraints

1. Identity payloads admit dictionaries, lists, strings, integers, booleans, and
   null; floats are rejected. This implementation wraps typed input and
   configuration JSON in canonical JSON strings in identity payloads. Decoding
   those strings preserves fractional values. Results and metadata support
   finite JSON numbers.
2. Registry schemas support integer `minimum`/`maximum` bounds in **1.6.0**.
   The `number` schema type remains unsupported. Integer millimetres,
   milliseconds, and similar explicit units fit the simulated skill interface.
3. An ordinary exception escaping the model callable releases its reservation
   without a successful model-call record. To charge attempted batches equally
   on success, timeout, malformed output, or provider failure, catch expected
   provider errors inside that callable and return a structured failure result.
   Do not persist arbitrary exception messages that may contain secrets.
4. Pollard governs dispatch and records its result; it does not impose this
   application's inference deadline or sensor freshness rules. The companion
   must reject late results and recheck expiry/cancellation immediately before
   the registered simulator handler changes state.
5. Policy callbacks implement `decide(PolicyContext) -> Decision`. They can deny
   registered tools. This milestone's evidence policy has richer dispositions
   and should record those explicitly.
6. `mode="record"` can invoke a duplicate call again by default. Use unique run
   identities or the explicit duplicate guard when reusing a store. `hybrid`
   mode can execute missing calls and must not implement historical inspection.
   Historical inspection should walk recorded nodes without an actuator.
7. Passing an explicit meter list replaces Pollard's defaults. Include
   `StepMeter()` explicitly if step totals are wanted alongside request totals.
   Do not equate step totals with requests: an action is also a step.
8. Pollard run roots bind a registry digest. Changing registered action specs
   while reusing a root can produce an integrity error. Record policy and model
   identity in the companion's decision records as well.

## Inspection verification

A standalone smoke program was executed with this project's
`.venv\Scripts\python.exe` against the installed 1.6.0 wheel. It verified:

- a synthetic failed batch with two question IDs consumed one request;
- a second batch was refused before its callback ran;
- unknown actions and out-of-bounds integer arguments created refusal nodes;
- no refused callback ran;
- a permitted bounded action ran once and did not consume a model request;
- a note could still be appended with the exact request budget consumed;
- the full recorded chain passed `verify`.

Observed charges were `requests=1`, `steps=2`; the chain contained a root, model
call, note, three refusals, and tool call. This API smoke test used only local
callables and an in-memory store; it did not call a model or operate hardware.
