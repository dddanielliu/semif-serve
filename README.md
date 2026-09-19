# semif-serve

**A drop-in Jev endpoint backed by open models.**

[SemIf](https://github.com/dddanielliu/SemIf) reads typed option probabilities straight out of
a model's logits, but it is a Python library with no server and a hard ceiling of 16 options
per decision. [Jev](https://docs.typesafe.ai/api) is an HTTP service with no such ceiling.
This serves the Jev wire protocol on top of SemIf, so an existing Jev client changes its base
URL and nothing else.

```
POST /v1/systemone          choice · score · noul, mixed freely in one request
GET  /health
```

## What it does

- **All three primitives.** `choice`, `score`, and `noul` all reduce to one readout: score a
  fixed option set, read the distribution back. Requests mixing them are answered in one pass.
- **No option ceiling.** Decisions with more options than one answer-slot alphabet are resolved
  by a runoff in SemIf's `runoff.score_options`: options are split into balanced groups, every
  group of every question is scored in one batch, and the winners run off. A decision that
  already fits one alphabet is a verified pass-through.
- **One prefill per round.** Every question in a request shares one state, so adding questions
  costs suffix tokens, not another prefill.

## Run it

```bash
git clone --recurse-submodules <this repo>
cd semif-serve
uv sync
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True uv run semif-serve
```

SemIf requires **exactly one visible CUDA device**, hence `CUDA_VISIBLE_DEVICES`.
`expandable_segments` is not optional on a 12GB card: without it the allocator fragments and a
4B model OOMs mid-batch.

Install `flash-linear-attention` too. Qwen3.5 uses gated delta rule, and without it
transformers silently falls back to a reference PyTorch implementation:

```bash
uv pip install flash-linear-attention
```

This needs a Python with development headers. A distro Python without them leaves Triton
unable to compile at runtime, so prefer a uv-managed interpreter
(`UV_PYTHON_PREFERENCE=only-managed uv venv --python 3.12`).

## Performance

Measured on one RTX 3080 Ti (12GB) with Qwen3.5-4B, driving a 40-control page through
jev-ultrafast's own client, 3 runs:

| Configuration | Decision latency |
| --- | ---: |
| Reference kernels, prefill per round | 1978 ms |
| `flash-linear-attention` + reused prefill | 1273 ms |
| ...plus batched answer-slot verification | **1164 ms** |
| Hosted Jev, published median | 178 ms |

The server issues a warmup decision at startup. Without it the first request pays ~10s of
Triton compilation.

Profiling one realistic browser-agent request (2532-token state, three questions, 40 targets)
shows where the remaining time goes:

| Cost | Share |
| --- | ---: |
| State prefill | 47% |
| Suffix forward | 42% |
| Tokenization | 9% |
| Cache replicate, other | 1% |

Roughly 89% is model forward compute. That is a property of running a 4B model over a
2500-token state on this card, not of the protocol, and no amount of serving work removes it.
Closing the rest of the gap to Jev means a smaller model or faster hardware. `causal_conv1d`
would remove one more fallback kernel, but this box's nvcc is CUDA 13 against a cu128 torch,
so it was not built.

Point a client at it:

```bash
curl -s localhost:8077/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": "The export button crashes the settings page in Safari.",
  "questions": {
    "severity": {"type": "score", "instructions": "How severe is this?",
                 "criteria": ["Cosmetic", "Workaround exists", "Blocking"]}
  }
}'
```

`uv run semif-serve --stub` serves deterministic scores with no model and no GPU, which is
enough to develop a client against the wire format.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `SEMIF_HOST` | `127.0.0.1` | Binding anything else requires `SEMIF_API_KEY` |
| `SEMIF_PORT` | `8077` | |
| `SEMIF_MODEL` | `Qwen/Qwen3.5-4B` | |
| `SEMIF_REVISION` | pinned commit | SemIf refuses unpinned remote models |
| `SEMIF_MAX_TOKENS` | `16384` | Over-limit prompts return 422; nothing is silently truncated |
| `SEMIF_MAX_BATCH` | `12` | Rows per scoring batch. See below |
| `SEMIF_API_KEY` | unset | When set, `Authorization: Bearer` is required |
| `SEMIF_QUEUE_SECONDS` | `20` | Wait for the GPU before shedding load with a retryable 529 |

**`SEMIF_MAX_BATCH` is a memory limit, not a throughput knob.** SemIf replicates the state
prefix across the batch, so peak memory grows with batch size. Measured on a 12GB RTX 3080 Ti
at a 2918-token prefix: batch 12 peaks at 10.7GiB, batch 16 OOMs. On a larger card, raise it.

## Model support

`Qwen/Qwen3.5-4B` is the default and the only model verified here.

**MiniCPM5-2B is not usable for this workload.** Not for lack of capability: SemIf rebuilds
the shared state prefix by serialising the evidence and dropping exactly one token to clear
any merge at the boundary. MiniCPM's vocabulary merges `]}` into a single token, so one token
is not enough and the reconstructed prefix stops being a prefix of the real prompt. It passes
on small states and fails on states ending in `[]`. jev-ultrafast's state always ends with
`recent_actions`, which is empty on the first step, so it fails there routinely. Qwen's
vocabulary does not form that merge.

Qwen3-0.6B runs but chooses poorly.

## Fidelity and its limits

The request and response shapes match the published Jev schema, including `usage` and the
401/422/429/529 error codes. Three things are honestly different:

- **`confidence` matches Jev for `score`, and is inferred for `choice`.** Jev's docs call it
  "a statistic computed from the probability distribution" and decline to give the formula.
  Their published score example pins it down: levels at `0.0/0.7/0.3` give `sigma = 0.4583`
  against a maximum of `(n-1)/2`, so `1 - sigma/sigma_max = 0.5417`, and Jev publishes `0.54`.
  Max-probability, normalised entropy and top-two margin give 0.70, 0.44 and 0.40, so the
  match is not a coincidence. Choice options are unordered and sigma is meaningless for them,
  and no worked choice example publishes a non-degenerate confidence, so choice uses
  normalised entropy `1 - H/log(n)` — the standard categorical dispersion measure, matching
  the documented "flat is low, peaked is high". That one is inferred, not confirmed.
  Separately, the underlying probabilities remain uncalibrated, as SemIf states.
- **Runoff distributions are a product, not a single softmax.** For decisions above the option
  ceiling, `P(option) = P(its group) · P(option | group)`. It is normalised over every option
  and no candidate is dropped, but it is not what one pass over all of them would produce.
- **`output_tokens` is always 0.** Nothing is sampled. The answer is a logit lookup.

## Development

```bash
uv run pytest        # offline, no GPU: the stub engine covers the whole wire path
uv run ruff check .
```

Tests never load a model. SemIf's own scoring is tested in the submodule.
