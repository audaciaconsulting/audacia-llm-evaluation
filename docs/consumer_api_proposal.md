# Proposal: a consistent consumer API

Status: draft for review. Raised from integrating the package into the MHRA `bp-ai-poc`
eval suite.

## Motivations

1. **Misuse currently fails open.** The pass/fail key differs per evaluator
   (`faithfulness_result`, `similarity_result`, `llm_as_judge_result`, or a caller-supplied
   one), so reading the wrong key returns `None` and the suite goes green forever. Same for
   a missed `await` on an async evaluator: the assertion never runs. A test library whose
   misuse produces a passing test is the worst case there is.
2. **The package makes work unrelated to evaluation.** Eight modules call
   `logging.basicConfig()` at import and reconfigure the consumer's root logger; the
   unpinned credential produces a 400 naming neither credential nor tenant; results carry
   whole source documents; no `py.typed`, so the type hints don't reach consumers.
3. **Azure AI Foundry reporting must stay cheap to enable.** Postponed, not declined — so
   readiness needs to be demonstrable, not assumed.
4. **The window is now.** Pre-production, one consumer, seven thin suites. Breaking changes
   cost a version bump today and a deprecation cycle later.

Most of this is not invention. `azure.ai.evaluation.SimilarityEvaluator` already has a sync
`__call__`, config-at-init, and a `credential` parameter; ragas exposes `single_turn_score`
alongside `single_turn_ascore`. Our wrappers invert or ignore all four.

## Current state

| Family | `__call__` | `assert_result` | `evaluate` | Pass/fail key |
| --- | --- | --- | --- | --- |
| `RagasBaseEvaluator` | async | async | — | `{metric}_result` |
| `BaseScoreEvaluator` | async | async | — | caller-supplied |
| `FormatBaseEvaluator` | sync | sync | `evaluate()` | — |
| `TransformerBaseEvaluator` | sync | sync | `evaluate_against_*()` | — |
| `RunSimilarityEvaluator` | sync | sync | `evaluate(assert_result)` | `similarity_result` |
| `RunLlmAsJudge*` | sync | sync | `evaluate(assert_result)` | `llm_as_judge_result` |

Three axes of inconsistency: whether you await, what the result key is called, and what
`evaluate` means. The split is documented (`README.md:184` plus an "Await?" column), so it
isn't a docs gap — but it must be looked up per evaluator, and no generic code is possible.
`tests/evaluators/test_similarity.py` shows the maintainers paying the same cost.

## Phase 1 — one interface

Breaking. Ship as `0.3.0`.

1. **An `Evaluator` base** all families inherit, defining `__call__`, `evaluate`,
   `assert_result`.
2. **Sync public API**, wrapping async internals with a loop-safe runner
   (`async_run_allowing_running_loop`, already in the tree — plain `asyncio.run` raises
   inside a running loop). Keep `assert_result_async()` / `evaluate_async()` public too: a
   service already in an event loop should await rather than block it.
3. **A typed result** — `EvalResult(name, score, threshold, passed, reason, inputs, raw)`.
   `result.passed` works everywhere and a typo raises instead of returning `None`.
   `raw` keeps the existing flat dict verbatim, following Azure's `{metric}` /
   `{metric}_result` / `{metric}_threshold` convention. Keep `EvalResult` a plain model, not
   dict-like — `evaluate()` needs a Mapping, so hand it `raw` at that one boundary.
4. **Config at init, data at call** — Azure's own convention. Ours takes data at `__init__`,
   so an evaluator can't be reused across rows: every case rebuilds its grader client via
   `get_ragas_wrapped_azure_openai_llm()` or `get_azure_ai_evaluation_model_config()`. Costs
   ~20 classes and a consumer migration, but Phase 1 already breaks, so it's one migration
   rather than two.
5. **Foundry readiness** (~half a day): an `as_azure_callable()` adapter returning `.raw`,
   plus a local `evaluate()` test. `azure_ai_project` is optional, so that test runs against
   `output_path` with no project and no RBAC — making readiness something CI checks.

Dropped from an earlier draft: threshold normalisation. "Document the ranges" is a doc edit,
not an API change. Note `threshold=False` is currently a sentinel on
`RunStringPresenceEvaluator` / `RunExactMatchEvaluator`; binary metrics deserve their own
construction path rather than overloading the parameter.

## Phase 2 — consumer ergonomics — **done**

Non-breaking.

6. **Remove the eight `logging.basicConfig()` calls.** A library reconfiguring the root
   logger on import is the same class of problem as the module-level `load_dotenv()` removed
   in `7a20ef5`. Use a `NullHandler`; add opt-in `configure_logging()` for notebooks.
   Check `red_teaming.ipynb` doesn't depend on the current behaviour.
7. **Ship `py.typed`** (PEP 561), and add `[tool.setuptools.package-data]` so it reaches a
   built wheel — otherwise type checkers ignore the package's hints entirely.

Considered and rejected: truncating large fields in returned results. Logging already
truncates contexts, which is right for logs, but the return value is data. Diagnosing a
failed faithfulness score needs the full retrieved text, and a consumer who wants it
shortened can do that themselves — the library cannot undo it.

## Phase 3 — authentication and config — **done**

9. **Let the tenant be pinned.** `get_credential()` accepts an explicit credential, so a
   consumer can pass `AzureCliCredential(tenant_id=...)` or a service principal. Previously
   nothing was passed and the SDK built an unpinned `DefaultAzureCredential`, resolving
   whichever credential it found first — Azure CLI on one machine, `azd` on another —
   failing with `400 Tenant provided in token does not match resource token`. That cost the
   MHRA integration time twice in one session from two different causes.

   Deliberately *not* added: an `LLM_EVAL_TENANT_ID` env var. `DefaultAzureCredential` has
   no tenant argument reaching `AzureCliCredential` or the `azd` credential — only
   `interactive_browser_`/`shared_cache_`/`visual_studio_code_`/`workload_identity_` — so
   such a var would silently do nothing for the credentials developers actually use. A knob
   that appears to work is worse than none. Building our own `ChainedTokenCredential` would
   work but re-implements the SDK's chain for a case `az login --tenant <id>` already fixes.
10. **Fail fast on missing config** — `_require_env` raises naming the unset variable
    instead of surfacing a 401 from the grader mid-run.

Dropped: a parameter for every env var. The `LLM_EVAL_*` names are documented in the README
and a single set is enough for now; revisit if a consumer needs two endpoints in one process.

## Phase 4 — LLM-as-judge

12. **Make `check_prompt_output` advisory.** It requires prompt markers and numbered input
    labels matching the `inputs` keys *in order*, but `with_structured_output` already
    guarantees the schema. The ordering requirement silently breaks a prompt when a consumer
    reorders inputs. Warn, don't raise.
13. **Offer per-criterion verdicts**, so multi-rule criteria don't collapse to one pass/fail.

## Sequencing

Phase 2, 3, 1, 4. Phase 2 has no API risk; Phase 3 removes a recurring support cost; Phase 1
is the breaking one, and its migration cost scales with the number of consumer suites — so if
it happens at all, it should happen before the real test cases are written, not after.

## Generality

Only one consumer exists, so items were checked against convention rather than preference:
sync `__call__` and config-at-init match Azure's evaluators; the flat dict is Azure's
convention, already mirrored via `camel_to_snake`; `credential` is a parameter the package
ignores; no-`basicConfig` and `py.typed` are standard Python.

Considered but unserved: a **batch/reporting** consumer would want aggregation across rows
(mean score, pass rate) — Azure does this via `_aggregate_results`, and we have no
equivalent. A **non-Azure** consumer can inject `llm`/`embeddings` into most evaluators, but
`RunSimilarityEvaluator` requires an `AzureOpenAIModelConfiguration` and `model_tools` is
Azure-only; Azure should be the default, not the assumption.

Still opinionated, with no external convention to appeal to: the `EvalResult` field set, and
per-criterion judge output.

## Appendix: async support in `evaluate()`

Not obvious, and worth recording. `evaluate()` picks one of three batch clients:

| Client | Async custom evaluator |
| --- | --- |
| `run_submitter` (default) | **Not awaited** — the metric becomes a coroutine object |
| `pf_client` | Works — `ProxyClient._should_batch_use_async` sniffs `iscoroutinefunction` |
| `code_client` | No handling |

The default only takes the async path for objects implementing the private
`HasAsyncCallable` protocol (`_to_async()`), whose sole implementer is Azure's own
`EvaluatorBase`. A custom async evaluator therefore fails *silently* there. Item 2's sync API
behaves identically on all three and relies on nothing private.
