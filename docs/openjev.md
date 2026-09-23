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
rubrics, calibration, or the separate SemIf project. OpenJev is an independent
implementation, not TypeSafe Jev's disclosed implementation or weights.

The model card declares MIT. No standalone LICENSE file appeared in the inspected
repository listing. The referenced base model
[Qwen/Qwen3.5-4B LICENSE](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE)
contains Apache-2.0 terms. These are the observed upstream declarations; model
and dependency licenses remain separate from this companion's code.

## Validation boundary

Normal tests inject an encoder with the verified method signature and exercise
request serialization, label mapping, malformed outputs, and lazy loading.
They do not load weights or demonstrate model quality. A real checkpoint has
not been executed in this milestone. No latency, VRAM, energy, or robotics
success claims follow from these mapping tests.

The adapter's synchronous encoder can continue computing after its caller is
cancelled. The supervisor must disregard late results and own all dispatch;
the adapter itself has no actuator API. A production inference process should
provide worker isolation and explicit cancellation or termination for resource
reclamation.

## Explicit setup for a later real-model run

These optional PowerShell commands install model dependencies and download the
selected checkpoint. They are separate from the offline milestone commands:

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

The milestone environment audit found no OpenJev checkpoint in the configured
Hugging Face cache and none of the optional model packages in the project venv.
The factory tests use small local test doubles; they do not exercise Transformers
loading or a CUDA kernel. Keep an environment lock and benchmark the real
checkpoint before making deployment or memory claims.
