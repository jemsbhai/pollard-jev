# Optional OpenJev backend

The offline simulator does not need model weights, model packages, credentials, or a GPU.
The optional adapter targets `AlexWortega/openjev`, checkpoint
`qwen3.5-4b-nli-v2`, at inspected repository revision
`552759daad712f1af6c4c13dabcb1e047886fc9c` (2026-09-23).

## Inspected interface and scope

The [upstream helper](https://huggingface.co/AlexWortega/openjev/blob/552759daad712f1af6c4c13dabcb1e047886fc9c/modeling_openjev.py)
provides `OpenJevCrossEncoder(path, subfolder=None, device=None, dtype=..., bs=32,
max_len=4096)` and `predict_hypotheses(premise, hypotheses)`. Its output has one
row per hypothesis, with three softmax probabilities in this order:
contradiction, entailment, neutral. Upstream ranking selects the largest
entailment score. The adapter preserves these independent hypothesis scores;
it does not renormalize entailment across actions or describe it as physical
success probability.

The [checkpoint config](https://huggingface.co/AlexWortega/openjev/blob/552759daad712f1af6c4c13dabcb1e047886fc9c/qwen3.5-4b-nli-v2/config.json)
specifies `Qwen3_5ForSequenceClassification`, those label IDs, and
`transformers_version: 5.15.0`. The current
[model card](https://huggingface.co/AlexWortega/openjev/blob/552759daad712f1af6c4c13dabcb1e047886fc9c/README.md)
now recommends v5 for typed decisions; v2 remains a documented NLI checkpoint.
This first adapter deliberately exposes that NLI interface and supports text
features only. It does not implement image input, sensor encoders, v5 typed
rubrics, probability calibration, or the separate SemIf project. The benchmark
can select policy thresholds on a separate calibration split without changing
the model's scores. OpenJev is an independent
implementation, not TypeSafe Jev's disclosed implementation or weights.

The model card declares MIT. No standalone LICENSE file appeared in the inspected
repository listing. The referenced base model
[Qwen/Qwen3.5-4B LICENSE](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE)
contains Apache-2.0 terms. These are the observed upstream declarations; model
and dependency licenses remain separate from this companion's code.

## Validation boundary

Normal tests inject an encoder with the verified method signature and exercise
request serialization, label mapping, malformed outputs, and lazy loading.
They do not load weights or demonstrate model quality. Separately, the pinned
4B checkpoint ran on Windows/CUDA on 2026-09-27 using an existing local runtime
and model cache with no downloads or installs. See [the local benchmark](BENCHMARKING.md)
for measured latency, GPU allocation, proposal errors, abstention and calibration
results. Those synthetic cases do not establish physical success or energy savings.

The adapter's synchronous encoder can continue computing after its caller is
cancelled. The supervisor must disregard late results and own all dispatch;
the adapter itself has no actuator API. A production inference process should
provide worker isolation and explicit cancellation or termination for resource
reclamation.

## Experimental input representations

The constructor and `from_local_cache` accept `representation="json-v1"`
(the unchanged default), `"text-v1"` (the same facts as prose), or
`"robot-rules-v1"` (facts plus fixed default-demo operating rules). The selected
version is recorded in `identity.settings.input_representation` and is read-only
after construction; create a new provider to select another representation.
Score semantics and permitted action parameters remain unchanged.

The rules variant is experimental and supplies additional task information.
Its fixed 10% battery/0.5 m clearance rules do not follow custom `PolicyConfig`
settings and do not encode the entire evidence policy. See the
[paired experiment and limitations](REPRESENTATION_EXPERIMENT.md). No alternative
has been promoted to the default.

## Reuse an existing local installation

Prefer the [offline benchmark command](BENCHMARKING.md#run-again-without-downloads)
when the model and optional dependencies are already available locally.
It uses the selected interpreter and cache without modifying the existing
installation. It never installs dependencies or downloads a missing checkpoint.

## Optional setup for a new installation

These commands install model dependencies and download the selected checkpoint.
They are only for an explicitly requested new installation; do not use them
for the existing local benchmark or under a no-download constraint:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[openjev]"
.\.venv\Scripts\hf.exe download AlexWortega/openjev --revision 552759daad712f1af6c4c13dabcb1e047886fc9c --include "modeling_openjev.py" "qwen3.5-4b-nli-v2/*"
```

The dependency constraint starts at the checkpoint's recorded Transformers
5.15 version. This is an inspected configuration, not a runtime compatibility
or Windows CUDA certification. Install the appropriate PyTorch build for the
actual machine when preparing a GPU benchmark.
The [published Transformers 5.15.0 package metadata](https://pypi.org/pypi/transformers/5.15.0/json)
also requires Hugging Face Hub `>=1.5,<2`; pip resolves that transitive constraint.
Optional acceleration libraries are not installed by this companion.

Then construct the provider in Python and pass it to the same governed loop:

```python
from pollard_jev.providers.openjev import OpenJevProvider

provider = OpenJevProvider.from_local_cache(device="cuda", max_length=4096)
# loop = DecisionLoop(provider=provider, ...)
# Use the normal typed requests and deterministic policy; provider.infer() returns data only.
```

The factory executes the inspected upstream Python helper from the local
commit-pinned cache. Heavy imports happen only here. Its
[`snapshot_download(local_files_only=True)`](https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download)
lookup cannot fetch missing files. The helper is given a local snapshot path;
the factory checks the required tokenizer, config, and single weights file,
and the configured NLI label order before executing the helper. A partial cache
fails at that check. Ordinary `infer` calls never download
weights. One logical provider invocation can contain multiple questions; the
adapter calls the encoder once per nonempty question and leaves request-budget
accounting to the supervisor.

Input observations retain units, timestamps, unknown markers, and validity
limits in their JSON premise. Action hypotheses are copied from the request.
Upstream tokenization can truncate at `max_length`, which is included in model
identity alongside the revision, dtype, and resolved device. Long-context
adequacy, real model behavior, and cancellation of native compute still need
benchmarking before deployment.

The initial milestone audit found no checkpoint in the default Hugging Face
cache or optional model packages in this project's venv. The 2026-09-27 audit
located an existing local runtime and model cache and successfully reused them.
The benchmark saves exact runtime versions and source/model hashes. The factory
unit tests remain small local doubles; actual CUDA evidence is recorded separately in the
benchmark report. No 0.8B checkpoint was found in the inspected local stores.
