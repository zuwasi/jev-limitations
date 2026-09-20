# Finding routing: context filtering and explicit review

An application-level experiment for the limitations documented at
https://docs.typesafe.ai/model-jaggedness/jev-1.13. This is not a change to Jev's
weights, a vulnerability verifier, or a production safety gate. It adds no
runtime dependencies and changes no adapter library behavior. The repository's
MIT license applies to this example and its synthetic fixtures.

## Run locally, without keys or network calls

From a source checkout with Python 3.10 or newer:

```sh
pip install -e .
python -m examples.finding_routing
```

Or use the repository's locked development environment:

```sh
uv sync --locked --all-extras
uv run python -m examples.finding_routing
uv run python -m pytest tests/test_finding_routing.py -q
```

For the complete upstream suite on Windows, allow only loopback sockets needed
by Python's async event loop. External network calls remain blocked:

```sh
uv run python -m pytest -q --allowed-hosts=127.0.0.1
```

The default demo uses the actual adapter with a scripted provider that always
prefers `dependency_investigation` with probability 0.97. It deliberately ignores
input and labels. It demonstrates plumbing and policy behavior, **not Jev's
accuracy, calibration, latency, or resistance to misleading text**.

On the eight supplied synthetic cases, both model arms misroute six important
findings. Review gates defer three of these, leaving three important misroutes
visible. Correct automatic routes remain two. This is not a measured improvement
in any real model. In particular, misleading descriptions and conflicting
evidence can still produce confidently wrong routes.

## What the experiment compares

1. **Unfiltered:** description, all evidence, and background.
2. **Filtered:** description plus all evidence with an exact matching component
   identity. Background and other components are excluded.
3. **Guarded:** reuse the filtered answer, but return `needs_review` for absent or
   incomplete evidence, a low selected probability, or model abstention.

All arms reject missing answers, invalid probability distributions, SDK failures,
and oversized context. They retain every case in the report. No findings are
closed, suppressed, or marked safe. Two model calls per case are made when both
contexts fit the size limit; call order alternates between cases. The guarded arm
does not make a third call.

The filter retains conflicting evidence for the target, source references, and
the complete description. It does not use labels or keyword matching. The caller
must map evidence to the correct component identity before running it. Cross-
component relationships can be relevant; attach that evidence to the target
explicitly or leave the finding for review. Filtering is **not redaction** and
does not neutralize instructions inside retained text.

`route_finding(client, finding)` is the standalone guarded path. Missing evidence
stops it before any model call. The evaluation deliberately still queries missing-
evidence cases in its two model arms to measure the review gate's effect.

The default threshold of 0.9 is illustrative, not a 90% accuracy guarantee. It
uses the selected Choice probability, **not** the separate `confidence` field.
Tune thresholds on a separate calibration set for each model, question, and
input format. Never transfer a threshold between Jev and an adapter provider
without validation. Invalid distributions are rejected, not renormalized.

The 8,000-character default cap measures serialized state only, not tokens or
the full request. Exceeding it causes review rather than silent truncation.
It is not a guarantee that a provider's token limit will be met.

## Evaluate previously reviewed cases

No customer data or historical reviews are included. `finding_cases.jsonl`
contains invented component names and synthetic expected routes. The fixtures
cover missing evidence, unrelated evidence, missing source references,
contradictions, misleading descriptions, and ordinary specialist routing.

Create a **private file outside this checkout**, using one JSON object per line:

```json
{"id":"sanitized-001","provenance":"reviewed","review_reference":"local-review:001","finding":{"component":"component-identity","description":"Sanitized finding description","evidence":[{"component":"component-identity","source":"sanitized-source-reference","text":"Relevant evidence"}],"background":"Optional unrelated context"},"expected_route":"source_review","important":true}
```

Allowed expected routes: `dependency_investigation`, `configuration_review`,
`source_review`, `needs_review`. An independent reviewer supplies the labels,
importance, and reference before model evaluation. The reference field records
provenance but does not prove a review happened. Labels and review metadata are
never sent to the model. Duplicate IDs, unknown fields, and mistyped values fail
validation rather than disappearing from the denominator.

Start with a separate calibration set, then evaluate a held-out reviewed set.
Include difficult and misleading cases. Do not tune against the held-out results.
Repeat live comparisons to account for model variability before claiming a gain.

Live execution requires an explicit model, an environment key, and consent to
send data to the paid external API:

```sh
# Set TYPESAFE_API_KEY through your shell or secret manager, never in source.
python -m examples.finding_routing --backend jev --model jev-1.13.0 --allow-network --cases /path/to/private-reviewed.jsonl

# Optional comparison through the adapter (install the openai extra first).
# Set OPENAI_API_KEY. Use a model ID available to your account.
python -m examples.finding_routing --backend openai --model YOUR_MODEL_ID --allow-network --cases /path/to/private-reviewed.jsonl
```

The Anthropic path uses `--backend anthropic`, its optional extra, and
`ANTHROPIC_API_KEY`. Neither comparison is an automatic fallback. The current
Jev model/API availability must be checked with your account.

Both unfiltered and filtered content leave the machine during a live evaluation.
Review and redact the **whole dataset**, not just the selected evidence. Console
JSON contains case IDs, labels, decisions, probabilities, reasons, character
counts, returned model IDs, and local elapsed times. Treat it as private when evaluating real cases.
It excludes source text, API keys, SDK debug traces, and exception messages.

## Read the results without rewarding abstention

- `missed_important`: important cases automatically sent to the wrong queue,
  including automatic routing when the expected outcome is `needs_review`.
  This is a routing-error measure, not a count of undetected vulnerabilities.
- `missed_important_rate`: that count divided by all important cases, or `null`
  when the set contains none. Read alongside `important_deferred`.
- `important_deferred` / `needs_review`: pending work, never successful verification.
- `coverage`: the fraction automatically routed.
- `auto_route_accuracy`: correct automatic routes divided by automatic routes;
  `null`, not 100%, when everything is deferred.
- Per-case `error`: abstention reason, including API failure and context overflow.
  API outages can reduce misroutes by deferring everything; that is not improvement.
- Per-call `elapsed_seconds`: local wall time, including SDK retries where used.
  Guarded decisions reuse a response and have no independent model latency.
  These are diagnostic observations, not statistically established speedups.

Compare missed important findings **and** review workload, coverage, and correct
routes before considering latency. Deterministic checks, tests, and observed
system state still establish whether any later action succeeded.
