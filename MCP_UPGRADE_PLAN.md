# ehrapy MCP — SOTA Upgrade Plan (Aug 2026)

**Audience:** an implementation agent working on branch `feat/add-mcp-tooling`.
**Scope:** everything under `ehrapy/mcp/`, `tests/mcp/`, plus `pyproject.toml` and MCP docs pages.
**Goal:** keep the existing catalog + namespace-dispatch architecture (it is the validated 2026 pattern — do NOT rewrite it into per-function tools or a code-execution server) and fix the specific defects and gaps listed below. Every task has acceptance criteria. Work through the phases in order; each phase should be a separately reviewable commit or PR.

Grounding: this plan implements the findings of the 2026-08-21 review, which was based on the MCP 2026-07-28 spec, Anthropic's "Writing effective tools for agents" / "Code execution with MCP", OpenAI GPT-5.6 + Google Gemini 3.7 + xAI function-calling guidance, FastMCP 3.4.x docs, and published production case studies (BioMCP "We Deleted 35 Tools", Harness dispatch redesign, Cloudflare Code Mode, scmcp, biocontext-ai/anndata-mcp).

---

## 0. Design decision: Markdown or JSON tool outputs?

**Answer: both, on separate channels — small typed JSON for machine state, Markdown for anything a model reads and reasons over. Never a JSON string that encodes what should be prose or a table.**

Why (this is the reasoning; the rules follow from it):

1. **MCP already gives you two channels.** Since spec 2025-06-18, a tool result carries `content` (text blocks the model reads) *and* `structuredContent` (JSON validated against the tool's `outputSchema`). The Software-3.0 framing ("English/Markdown is the programming language of agents") and the JSON-schema framing are not in conflict — the spec resolved it by giving each its own lane. Servers that pick only one lane are leaving the other's benefits on the table.
2. **Markdown wins for anything bulk or prose — measurably.** During the review we measured a 200×50 DataFrame serialized the current way (`orient="records"`, `indent=2`): **~424K chars ≈ 106K tokens**. The identical data as CSV/Markdown-table text: **~48K tokens (4.4× cheaper)**, because records-orient repeats every column name on every row and JSON escaping/indentation adds ~2× on top. Research agrees with the intuition: forcing models to *emit* rigid JSON measurably taxes reasoning (Tam et al. 2024, ~27-pt drops; the 2026 "Natural Language Tools" replications), and models *read* Markdown at least as reliably as JSON. Tables, summaries, docstrings, workflow guides, error explanations → Markdown.
3. **JSON wins for anything the model must pass back verbatim or a program must consume.** Handles (`edata_id`), status, shapes, counts, file paths, error codes, suggested next calls. These need to be exact, enumerable, and schema-validatable — that's what `structuredContent` + `outputSchema` are for, and typed output schemas are also what make code-mode/programmatic-tool-calling stubs work (both Anthropic PTC and the MCP client best-practices doc generate typed stubs from `outputSchema`).
4. **The current code does the worst of both:** tools return a Python `str` containing a JSON blob, so FastMCP's generated `outputSchema` is the meaningless wrapper `{"result": string}` — double-encoded JSON with no validation and a 2×-inflated token bill.

**Concrete rules for this codebase (apply everywhere):**

- Every tool returns a **`fastmcp.tools.tool.ToolResult`** (or a typed dict/dataclass where the whole result is small and structured — FastMCP auto-generates `outputSchema` from return annotations):
  - `structured_content`: a **small, flat** JSON envelope — `status`, `edata_id`, `namespace`, `function`, shapes/counts, `plot_path`, `suggested_next`, error fields. Target < 400 chars. No nested tables, no row data.
  - `content`: compact **Markdown** — the human/model-readable rendering: a heading line, a Markdown table for tabular data (pipe table, ≤ 20 rows default), truncation/steering notes, next-step hints as prose.
- **DataFrames/Series → Markdown pipe tables** in `content`, never `orient="records"` JSON. Shape and truncation info goes in both channels.
- **Reference/docs tools** (`get_workflow_guide`, `get_function_help`, `list_ehrapy_functions`) → Markdown-first; their structured envelope is just `{status, ...counts}`.
- **`json.dumps(..., indent=2)` is banned on the wire.** Where JSON must be emitted, use compact separators.
- Never embed untrusted data (file contents, dataset values) into descriptions or instructions — tool output content is fine, descriptions are a security surface.

---

## Phase 1 — Correctness & token safety (P0)

### T1. Fix `run_get` rewriting the dataset on every read  (F2)

`catalog.py` marks the `get` namespace `kind="edata"`, and `run_get` hardcodes `in_place=True`, so `_dispatch_edata_or_plot` calls `persist_edata` → a **full `write_h5ed` of the entire dataset on every `obs_df` read**.

- Add a new namespace kind `"get"` (read-only): dispatches like `edata` but **never** persists and never rebinds `session.edata_id` to a new handle.
- While here: remove the `in_place` parameter from **all** tool signatures and from `run_dispatch`. Rationale: `in_place=False` currently mutates in memory, silently discards the result, and still reports `status: "ok"` — a lie. The documented pattern for preserving state is `fork_edata_handle` first; keep only that. One less parameter in every schema, one less foot-gun for fast models.
- **Accept:** a test loads a dataset, records `mtime_ns` of the cached `.h5ed`, calls `run_get("obs_df", ...)` and `get_edata_snapshot`, asserts `mtime_ns` unchanged. `in_place` appears nowhere in `tools/`.

### T2. Smart serialization: profile-first, information-ranked truncation  (F1)

Measured baseline (real client session, mimic_2, see Appendix A): `run_preprocessing("qc_metrics")` returns **11,590 tokens**, `run_io("to_pandas")` **66,746 tokens**, `run_get("obs_df", keys=2)` **3,482 tokens**. The fix is not just a byte cap — it is choosing *what information survives*. Head-truncation alone is wrong for EHR data (skewed distributions, missingness is the signal). Rewrite `serialization.py` around three tiers:

**Tier 1 — pass-through.** If the fully rendered Markdown table fits the budget (≤ 2,500 chars for the table portion), emit it whole. Small results must never be summarized away.

**Tier 2 — profile, don't sample (the default for anything big).** A DataFrame that would blow the budget is replaced by a **column-profile table** — one row per column:

| column | dtype | non-null % | unique | summary |
|---|---|---|---|---|
| numeric | — | — | — | `mean ± std [min, max]` |
| categorical/bool | — | — | — | top-3 values with counts |
| datetime | — | — | — | min → max |

This is what the analyst-model actually needs from a wide EHR table (it's also what `biocontext-ai/anndata-mcp` ships as `get_descriptive_stats`). A 46-column mimic_2 profile ≈ 1–2K tokens and is *more* useful than 200 raw rows. When even the profile exceeds budget (very wide data), rank columns and keep the top K by an **information score** — this is the principled-selection piece:

- `score(col) = w_m · missing_frac + w_v · dispersion + w_r · relevance`, with
  - **dispersion**: numeric → coefficient of variation on z-scored values (fallback: std of min-max-scaled); categorical → normalized Shannon entropy `H/log(k)`;
  - **missingness** weighted *up* (for EHR QC, high-missing columns are precisely what the agent must see);
  - **relevance** = 1 for columns named in the call's `params` (`keys`, `groupby`, `duration_col`, `event_col`, …), else 0 — requested columns always survive.
- Weights as module constants (`w_r=2, w_m=1, w_v=1`), deterministic tie-break by column name. Always state how many columns were dropped and that the score ranks by missingness/variance.

**Tier 3 — rows on request.** When the agent explicitly asks for rows (`response_format="detailed"`, or a `rows` param on `run_get`), return **head 5 + tail 5 + seeded stratified sample** to the row budget — stratified by the first categorical among the relevant columns (`groupby`/first key), else uniform. Head-only sampling hides the tails of skewed clinical distributions; a fixed seed keeps calls reproducible. Mark sampled rows (`sample` column: head/tail/random).

Deliberately **not** doing: coresets, sketching, mutual-information-vs-target, or topological descriptors. Entropy/variance/missingness ranking + stratified sampling is the right cost/benefit for a serialization layer; anything heavier belongs in analysis functions the agent calls explicitly (and would make responses nondeterministic and slow). Revisit only if T21 evals show the profile tier failing.

**Plumbing (all measured failures must be covered):**

- Tuple/dict-of-DataFrames results (e.g. `qc_metrics` returns *two* metric tables — the 11.6K-token offender) → one profile per table, sharing a single payload budget.
- **Whole-table returns** (`to_pandas` — the 67K-token offender): always Tier 2, with the steering line "full table not returned — use `export_edata` to write it to disk, or `run_get` with `keys=[...]` for specific columns."
- **Empty-result guard**: a 0-column/0-row DataFrame from a function that took no selection params (measured: `obs_df` with no `keys` returns shape `[1776, 0]` with `status: ok`) must NOT report plain success — return `status: "ok_empty"` with `agent_action` listing up to 30 available column names.
- Hard payload cap stays as backstop: rendered `content` > **10,000 chars** → truncate at a row boundary + steering line with exact remaining counts. Env override `EHRAPY_MCP_MAX_RESULT_CHARS`.
- ndarrays → shape/dtype only; `repr` fallbacks capped at 1,000 chars; nested containers share one budget.
- **Accept (regression-tested against Appendix A):** on mimic_2 — `qc_metrics` response ≤ 1,500 tokens (chars//4) and names the highest-missingness columns; `to_pandas` ≤ 1,500 tokens with the export steering line; `obs_df` with no keys returns `ok_empty` + column list; `obs_df` with 2 keys ≤ 600 tokens; a 5×3 DataFrame passes through whole; profile of a 46-col frame lists every column; column-ranking test: a column that is 60% missing survives Tier-2 ranking of a 100-col frame while a constant column does not.

### T3. `response_format` parameter on dispatch tools

Add `response_format: Literal["concise", "detailed"] = "concise"` to `run_preprocessing`, `run_analysis`, `run_get`, `run_io`. (`concise` = T2 defaults; `detailed` = raised caps.) This is the Anthropic-documented ~3× token lever; enum keeps it fast-model-safe.

- **Accept:** schema for `run_get` shows the enum with default `"concise"`; `detailed` returns more rows in a test.

### T4. In-memory dataset cache  (F3)

Every dispatch currently does a full `read_h5ed` and every mutation a full `write_h5ed`. Add to `edata_store.py` an in-process LRU keyed by `edata_id` (default capacity 3 datasets, env `EHRAPY_MCP_CACHE_ENTRIES`):

- `load_edata` hits the LRU first; validates staleness via the registry's `mtime_ns` before trusting a cached entry.
- Mutations still **write through** to disk (durability across restarts stays) but update the LRU so the next call skips the read.
- `kind="get"` dispatches read from LRU without any disk write (T1).
- **Accept:** test that two consecutive `run_get` calls trigger exactly one `read_h5ed` (monkeypatch-count it); write-through test that a `run_preprocessing` mutation is visible after clearing the LRU.

### T5. Loud rejection of unknown arguments  (F6)

Delete the silent-strip behavior in `AgnosticFastMCP.call_tool` (`server.py`). Replace with FastMCP v3 **middleware** (`on_call_tool`): if arguments contain keys outside the tool schema (allow the known orchestration key `wait_for_previous`), return a steering execution error — do not silently drop, do not run the tool:

```
status=error, error_code=UNKNOWN_ARGUMENT,
reason="Unknown argument(s): parameters. Valid arguments for run_preprocessing: function, edata_id, params, response_format.",
agent_action="Rename the argument and retry. Function kwargs go inside `params`."
```

This matches MCP SEP-1303: validation failures must be *execution* errors the model can self-correct from. (Spec-behavior note: models that pass `parameters` instead of `params` currently get a silent success with default kwargs — the worst possible outcome.)

- **Accept:** test calling `run_preprocessing` with a bogus top-level kwarg gets `UNKNOWN_ARGUMENT` naming valid keys; `wait_for_previous` is still tolerated; the `AgnosticFastMCP` subclass is deleted.

### T6. Structured output (kill the double-encoding)  (F7)

Convert every tool from `-> str` (JSON blob) to `ToolResult` per §0. Restructure `dispatch.py`'s `_ok` into an envelope builder that returns `(structured_content: dict, content_markdown: str)`. Errors follow the same dual-channel shape (envelope keeps `error_code` / `agent_action` / `details`; content is a two-line Markdown rendering: what failed → what to do). Keep returning error results rather than raising, so the structured steering fields survive; FastMCP's `mask_error_details` never applies because we never leak raw tracebacks (wrap the generic `except Exception` message at 300 chars).

- **Accept:** live `list_tools()` dump shows real per-tool `outputSchema`s (no `x-fastmcp-wrap-result` string wrapper); an MCP client round-trip test (`fastmcp.Client` in-memory) sees both `structuredContent` and Markdown text content on `run_get`. Update all existing tests that `json.loads` tool returns.

---

## Phase 2 — Agent ergonomics (P1)

### T7. Numpydoc-powered `get_function_help`  (F4)

This is the single highest-leverage change in the module. `function_help` currently returns the first docstring paragraph plus raw `str(annotation)` — the model never sees what a parameter *means* or which values are valid, and wrong argument **values** are the #1 measured failure class in 2026 tool-calling evals.

Rewrite `catalog.function_help` to:

- Parse the numpydoc `Parameters` section (ehrapy/ehrdata docstrings are numpydoc; use `numpydoc.docscrape.NumpyDocString` — add `numpydoc` to the `mcp` extra — with a regex fallback if unavailable).
- Emit per-parameter: name, type string, **description**, default, and — when the annotation is `Literal[...]` or the docstring enumerates options — an explicit `choices` list.
- Include the `Examples` section if present (first example only, ≤ 600 chars).
- Render as **Markdown** (signature line, then a parameter table) with a small structured envelope; total ≤ 2,500 chars, truncating the longest descriptions first.
- Append one synthesized example call: `run_preprocessing(function="knn_impute", params={"n_neighbours": 5})` using 1–2 salient params.
- **Accept:** `get_function_help("preprocessing", "knn_impute")` output contains the `backend` choices `scikit-learn`/`faiss` and a prose description for `n_neighbours`; payload ≤ 2,500 chars.

### T8. Rewrite all tool descriptions  (F5)

Target style (write-once-run-anywhere, per Anthropic/OpenAI/Google convergence): **3–4 sentences, front-loaded** — (a) what it does + returns, (b) when to use it, (c) when NOT to / what it can't do, (d) one caveat or precondition. Active voice. **No "CRITICAL/MUST/ALWAYS"** (frontier models overtrigger on it); rails for fast models go into enums/types, not prose. ≤ 2KB each (Claude Code truncation limit), realistically ≤ 500 chars.

Exact texts for the six load-bearing tools (use verbatim; keep docstring = description):

- **`run_preprocessing`** — "Run one ehrapy preprocessing function (ep.pp.*) on a cached dataset: QC, encoding, imputation, normalization, filtering, PCA, neighbors. Pass the function's keyword arguments as the `params` dict; call `get_function_help('preprocessing', function)` first if you are unsure of valid parameters. Mutates the dataset for `edata_id` in place — call `fork_edata_handle` first to keep the previous state. Returns a result summary and the `edata_id` the change was applied to."
- **`run_analysis`** — "Run one ehrapy analysis function (ep.tl.*) on a cached dataset: survival (kaplan_meier, cox_ph), causal inference (iptw, g_computation), embeddings (umap, tsne), clustering (leiden), feature ranking. Pass function kwargs in `params`; use `get_function_help('analysis', function)` for parameter details. Most functions store results on the dataset (e.g. in .obsm/.uns) rather than returning data — read them back with `run_get` or `get_edata_snapshot`. Not for preprocessing (use run_preprocessing) or plotting (use run_plot)."
- **`run_get`** — "Read tabular results from a cached dataset via ep.get.*: obs_df / var_df (observation or variable tables, filtered by `keys`), rank_features_groups_df (ranked features after run_analysis('rank_features_groups')). Read-only — never modifies the dataset. Returns a Markdown table truncated to the first rows; narrow with params (e.g. keys=[...]) rather than requesting everything."
- **`run_plot`** — "Render one ehrapy visualization (ep.pl.*) for a cached dataset and return the image. Most plots require prior computation (e.g. run_analysis('umap') before plotting 'umap'); use `get_function_help('plot', function)` for parameters. The PNG is returned as image content and also saved on the MCP host (path in the result). Not for computing results — plotting only reads the dataset."
- **`list_ehrapy_functions`** — "List the callable ehrapy/ehrdata functions behind the run_* dispatch tools, grouped by namespace (preprocessing, analysis, get, plot, io, demo). Omit `namespace` for the full catalog with one-line namespace summaries; pass one to get just its function names. Use this when you don't know the exact function name; then call get_function_help for its parameters."
- **`get_function_help`** — "Return the signature, parameter descriptions, valid choices, defaults, and an example call for one dispatchable function, rendered as Markdown. Call this before run_preprocessing/run_analysis/run_plot whenever you are not certain of a function's parameters — parameter names and values must match the underlying ehrapy API exactly. Namespaces: preprocessing, analysis, get, plot, io, demo."

Rewrite the remaining tools (`get_workflow_guide`, `get_runtime_context`, `ingest_dataset`, `load_demo_dataset`, `fork_edata_handle`, `export_edata`, `get_edata_snapshot`, `run_io`) in the same four-part pattern. Must-mention caveats: `ingest_dataset`/`export_edata` operate on the **MCP host** filesystem (not the agent sandbox); `load_demo_dataset` lists 2–3 real dataset names; `get_edata_snapshot` is the cheap first look after loading; `fork_edata_handle` is how you checkpoint before destructive steps.

- **Accept:** every description is 3–4 sentences; total `tools/list` still ≤ 4K tokens (T16 enforces); no MUST/CRITICAL/ALWAYS anywhere in `ehrapy/mcp/`.

### T9. Namespace/tool naming alignment  (F11)

A fast model *will* call `list_ehrapy_functions(namespace="analysis")` today and get an error, because the internal namespace is `tools`.

- Rename catalog namespaces to match the tool surface 1:1: `preprocessing`, `analysis` (was `tools`), `get`, `plot`, `io`, `demo` (was `dt`).
- Accept old names as silent aliases in `list_functions` / `get_callable` / `function_help` (one alias dict, no deprecation noise — this is unreleased).
- **Accept:** `list_ehrapy_functions(namespace="analysis")` and `...(namespace="tools")` both work; catalog output and all descriptions/prompts use the new names.

### T10. Plot results as image content  (F9)

`run_plot` currently returns only a host filesystem path — useless to any non-local client.

- Return the PNG via FastMCP's `Image` helper inside the `ToolResult` content (alongside the Markdown summary), with `plot_path` kept in `structured_content` as supplementary metadata.
- Guard size: render at dpi 100; if the file exceeds ~800KB, re-render at dpi 72 before embedding; always keep the host path either way.
- **Accept:** in-memory client test on a small demo plot receives an `ImageContent` block plus the path in `structuredContent`.

### T11. Tool annotations  (F10)

Add `annotations=` at registration (`server.py` — switch registration from the bare loop to explicit `mcp.tool(annotations={...})(fn)` with a per-tool table in `tools/__init__.py`):

| Tool | readOnlyHint | destructiveHint | idempotentHint |
|---|---|---|---|
| get_workflow_guide, get_runtime_context, list_ehrapy_functions, get_function_help, get_edata_snapshot, run_get | true | — | true |
| run_plot | true (writes only to its own cache) | — | — |
| load_demo_dataset, ingest_dataset, fork_edata_handle | false | false | false |
| run_preprocessing, run_analysis | false | **true** (in-place mutation) | false |
| run_io, export_edata | false | **true** (host writes) | false |

- **Accept:** live dump shows annotations on every tool; zero added description tokens.

### T12. Session hygiene + misc  (F12)

- **Echo the resolved handle:** every dispatch envelope already returns `edata_id`; additionally add `used_latest: true` when the handle came from the session fallback rather than an explicit argument, and mention the fallback semantics in the run_* descriptions' params line ("omit edata_id to use the most recent dataset").
- **Delete `get_package_info`** (duplicates `list_ehrapy_functions`); move `ehrapy_version` into `get_runtime_context`. The module is unreleased — no alias needed. Update `tests/mcp/` and `docs/api/mcp_index.md`.
- **Pin `fastmcp>=3.4,<4`** in `pyproject.toml` (the current `>=2.0` spans a major-version boundary; v2 lacks APIs this code calls; v4 changes session semantics — revisit deliberately).
- Compact JSON everywhere per §0.
- Keep `tools/inspection.py`'s `summarize_edata` alias or delete it — decide once; recommendation: delete (nothing released depends on it) and drop it from `__all__`.
- **Accept:** `uv pip install -e ".[mcp]"` resolves fastmcp 3.4.x; 14 tools registered; no `indent=2` in `ehrapy/mcp/`.

### T13. Schema portability floor (Gemini-safe)  (F8)

- Keep `X | None` optionals (ecosystem sanitizers collapse `anyOf:[T,null]` to nullable) but **forbid** everything else that breaks the Gemini subset: no heterogeneous unions, no `$ref`/`$defs`, no `oneOf`/`allOf`, no nested objects beyond the single open `params` dict.
- Add a unit test that walks every generated `inputSchema` and asserts the floor (allow `anyOf` only in the `[T, null]` form; allow `additionalProperties: true` only on `params`).
- **Accept:** the schema-floor test passes and runs in CI.

---

## Phase 3 — Security & clinical-data posture (P2)

### T14. Filesystem confinement + read-only mode  (F13)

- `EHRAPY_MCP_ALLOWED_ROOTS` (colon-separated absolute paths): when set, `ingest_dataset`, `export_edata`, and every `run_io` path argument must resolve (after `expanduser` + `resolve()`, i.e. symlink-safe) inside one of the roots; otherwise return `PATH_NOT_ALLOWED` with an `agent_action` naming the allowed roots. Unset = current behavior (document that explicitly).
- `EHRAPY_MCP_READ_ONLY=1`: `export_edata` and write-side `io` functions (`write_*`) return `READ_ONLY_MODE` errors; mutating dispatch still works (it only touches the private cache). Filter at dispatch time, not registration, so the tool list stays cache-stable.
- Extract the path-check into `errors.py`/a small `policy.py` so all three entry points share it.
- **Accept:** tests for traversal (`root/../../etc/passwd` rejected), symlink escape rejected, read-only blocks `write_h5ed` but not `run_preprocessing`.

### T15. Cache location, permissions, lifecycle  (F14)

EHR data currently lands in world-readable shared temp (`/tmp/ehrapy_mcp`) with no cleanup.

- Move the cache to `platformdirs.user_cache_dir("ehrapy-mcp")` (add `platformdirs` to the `mcp` extra), env override `EHRAPY_MCP_CACHE_DIR`. `mkdir(mode=0o700)` and `chmod` the tree on startup.
- Registry gains `purge(older_than_days: int)`; call it on server start with env `EHRAPY_MCP_CACHE_TTL_DAYS` (default: no purge). Remove orphaned records whose `cache_path` no longer exists (today these raise raw `FileNotFoundError` → make `load_edata` translate to the `EDATA_ID_UNKNOWN` envelope with "reload or re-ingest" steering).
- Add a short **"Data at rest"** paragraph to `docs/installation.md`'s MCP section: where data lives, permissions, TTL, and the two env flags from T14. Clinical deployers must be able to answer this without reading code.
- **Accept:** fresh server creates a 0700 user-scoped cache dir; stale-handle test gets a steering envelope, not a traceback.

---

## Phase 4 — SOTA-plus differentiators

### T16. CI token-budget gate (BioMCP pattern)

New test `tests/mcp/test_budgets.py`:

- Serialize the full live `tools/list` (names + descriptions + inputSchema + outputSchema + annotations); assert **≤ 16KB and ≤ 4,000 tokens** (chars//4 estimate is fine).
- Assert every individual description ≤ 2KB.
- Assert the T2 response budget on a worst-case DataFrame.
- **Accept:** runs in the normal pytest suite (importorskip fastmcp, as today).

### T17. Next-step steering in success envelopes

Add a small static workflow graph (dict in `prompts.py` or new `steering.py`): map `(namespace, function)` → up to 3 suggested next calls with 1-line reasons. Cover at minimum: load/ingest → snapshot + qc_metrics; qc_metrics → encode / missing_values plots; encode → pca; pca → neighbors; neighbors → umap + leiden; leiden → rank_features_groups; rank_features_groups → rank_features_groups_df + plots; kaplan_meier → plot; cox_ph → forestplot; iptw/aipw → love_plot + covariate_balance. Emit as `suggested_next: ["run_analysis(function='leiden')", ...]` in `structured_content` and one prose line at the end of `content`. Unknown functions → omit the field (never fabricate).

- **Accept:** `run_preprocessing("pca", ...)` response suggests neighbors; a function with no entry has no `suggested_next` key. BioMCP measured this exact mechanism halving call counts on canonical flows.

### T18. MCP prompts + rewritten server instructions

- Register **three MCP prompts** via `@mcp.prompt` (they surface as slash commands in Claude Code / Cursor / VS Code): `ehrapy-explore` (load → snapshot → QC → encode → missingness plots), `ehrapy-clustering` (hvf → pca → neighbors → umap/leiden → rank features → read back), `ehrapy-survival` (table one → kaplan_meier → cox_ph → forest plot). Each takes an optional `dataset` argument; content = the flow with exact tool calls, expected outputs, and success criteria. Verb-first names, descriptions written for a human picking from a menu.
- Keep `get_workflow_guide` as a tool (some clients ignore prompts) but rewrite `WORKFLOW_PROMPT`: fix the namespace table to post-T9 names, drop the `in_place` paragraph (removed in T1), add one line on `response_format`, `suggested_next`, and host-vs-sandbox paths.
- Rewrite server `instructions` (≤ 2KB; first sentence doubles as the tool-search trigger in Claude Code):

  > "Electronic health record analysis with ehrapy: load patient cohorts (CSV/h5ed or built-in demos), run quality control, imputation, clustering, survival and causal analysis, and produce plots. Use this server whenever a task involves EHR/clinical tabular data analysis. Datasets are cached server-side and referenced by `edata_id` handles returned from every call; omit `edata_id` to reuse the most recent dataset. Start: `load_demo_dataset` or `ingest_dataset`, then `get_edata_snapshot`. Discover the ~150 dispatchable functions with `list_ehrapy_functions`, get parameters with `get_function_help`, execute via `run_preprocessing` / `run_analysis` / `run_get` / `run_plot` / `run_io`. Guided flows: `get_workflow_guide` or the ehrapy-* prompts."

- **Accept:** in-memory client lists 3 prompts; instructions ≤ 2KB; guide/tool/instructions all agree with the post-T9 surface (grep for `run_analysis`/`analysis` consistency).

### T19. Provenance op-log (scmcp pattern)

On every successful mutating dispatch, append to `edata.uns["ehrapy_mcp_ops"]` (create as list): `{tool, namespace, function, params: <keys + scalar values only, truncate values at 100 chars>, edata_id, parent_id, timestamp}`. It persists inside the `.h5ed`, survives forks, and makes any agent-produced dataset auditable/reproducible — a differentiator no general MCP guidance ships. (A future `export_script` tool can replay it; out of scope here.)

- **Accept:** after two preprocessing calls, `uns["ehrapy_mcp_ops"]` has two ordered entries; fork preserves the log.

### T20. Progress reporting for slow functions

Maintain a `SLOW_FUNCTIONS` set (`knn_impute`, `miss_forest_impute`, `neighbors`, `umap`, `tsne`, `leiden`, causal estimators, `read_csv` on large files). In `run_dispatch`, when `ctx` is a real FastMCP `Context` and the function is slow, call `await ctx.report_progress(...)` before/after execution and `ctx.info(...)` with the function name. This requires making `run_dispatch` async or adding an async wrapper — take the async path (the tool layer is already async). Note FastMCP v4 background tasks (`task="optional"`) as the eventual home for this; do not adopt v4 now (T12 pin).

- **Accept:** a test with a stub Context records a progress call for `neighbors` and none for `obs_df`.

### T21. Eval harness (evals before further opinions)

Add `scripts/mcp_eval.py` + `tests/mcp/eval_tasks.yaml` (not run in CI): ~12 realistic multi-tool tasks with programmatic checks — e.g. "load mimic_2, run QC, report the 3 most-missing variables" (check: response names columns that are actually most-missing), "cluster mimic_2 and return top features per cluster", "KM curve by service unit → image returned", plus 3 adversarial ones (bogus function name → does the model recover via `list_ehrapy_functions`; wrong param name → does the T5 error steer it; huge table request → does truncation steering lead to a narrowed retry). The script drives any MCP-capable agent CLI against the server and grades the checks. Document in the script header how to run against one frontier + one fast model. Every published gain in this space (Anthropic, BioMCP, Harness) came from transcript-reading, not taste — this is the mechanism for all future description tuning.

- **Accept:** script runs end-to-end against a local server with a manual model; YAML schema documented in-file.

---

## Sequencing & PR breakdown

| PR | Contents | Risk |
|---|---|---|
| 1 | T1, T2, T3, T4 (correctness + budgets) | Low — internal semantics |
| 2 | T5, T6, T7 (validation, structured output, help) — the big surface change; update all tests | Medium |
| 3 | T8–T13 (descriptions, naming, plots, annotations, hygiene, schema floor) | Low |
| 4 | T14, T15 (security posture + docs) | Low |
| 5 | T16–T21 (budget gate, steering, prompts, provenance, progress, evals) | Low |

Constraints for the implementing agent:

- **Do not** add new tools beyond this plan (14 total after T12), convert to per-function tools, or add a code-execution tool — explicitly out of scope (revisit only after T21 evals show chain-length failures).
- Match existing code style (ruff config, numpy-style docstrings, `from __future__ import annotations`).
- Keep tool **docstring = description** as the single source (FastMCP reads docstrings); the T8 texts become the docstrings.
- Run `pytest tests/mcp -x` after every task; the suite must stay green per phase.
- After PR 3, regenerate the live `tools/list` dump and confirm the T16 budget by hand once before relying on CI.
- After PR 1 and again after PR 5, re-run the Appendix A dogfooding session and update the table — the measured numbers are the ground truth this plan is graded against.

---

## Appendix A — Measured baseline (dogfooding session, 2026-08-21)

Method: real MCP client round-trip (`fastmcp.Client` in-memory against `ehrapy.mcp.server.mcp`), mimic_2 demo dataset (1776 obs × 46 vars), measuring the text content a client actually receives. Token estimate = chars // 4.

| Call | Received | Should be | Defect exercised |
|---|---|---|---|
| `load_demo_dataset("mimic_2")` | 198 chars (~49 tok) | ✓ fine | — |
| `get_edata_snapshot` | 256 chars (~64 tok) | ✓ fine | — |
| `run_preprocessing("qc_metrics")` | **46,360 chars (~11,590 tok)** — two full metric DataFrames as records-JSON | ≤ 1,500 tok profile + top-missing summary | T2 tier 2, tuple handling |
| `run_get("obs_df")` no keys | 317 chars, `status: ok`, shape **[1776, 0]** — silent empty success | `ok_empty` + available-columns steering | T2 empty guard |
| `run_io("to_pandas")` | **266,987 chars (~66,746 tok)** | ≤ 1,500 tok profile + export steering | T2 whole-table guard |
| `run_get("obs_df", keys=["age","gender_num"])` | 13,930 chars (~3,482 tok) for 2 columns | ≤ 600 tok | T2 tier 1/3, records-orient + indent |
| `run_plot("missing_values_matrix")` | 297 chars, host path only, no image content | image block + path | T10 |
| Static: `tools/list` | 7,695 chars (~1,923 tok), 15 tools | ✓ under budget; keep ≤ 4K tok | T16 gate |
| Static: 200×50 DataFrame at serializer limits | ~424K chars (~106K tok) | ≤ 10K chars hard cap | T2 backstop |

Also verified live: `outputSchema` is the `{"result": string}` wrapper on every tool (T6); `anyOf:[T,null]` unions present in every optional param (T13); descriptions are single sentences (T8); `params` is an open `additionalProperties: true` object (inherent to dispatch — mitigated by T7).

---

## Phase 8 — Merge with `origin/feat/add-mcp-tooling` (2026-08-21)

Origin advanced 5 commits (`887bd60`..`f5ac5b2`) fixing numbered issues #2–#12 while this
rewrite was in progress. The two sides have **incompatible return contracts** — origin's
`dispatch.py` returns `dict` payloads from `_ok()`, this rewrite returns `ToolResult`
(dual-channel, see §0). A line-level merge is therefore not meaningful: 13 files conflict
and nearly every hunk would be resolved by discarding one side wholesale.

**Recorded as:** `git merge -s ours` (marks origin merged, keeps this tree) followed by a
port commit applying each origin fix to the new architecture. This checklist was written
*before* the `-s ours` merge — after it, origin's fixes are invisible to `git log`.

### Port checklist

| Issue | Origin commit | Status in rewrite | Action |
|---|---|---|---|
| #2, #12 obs/var names in snapshot | `887bd60` | ✅ already present | none |
| #3 fold top-level kwargs into `params` | `58d6d9f` | ❌ **inverted** (rewrite rejects) | **hybrid**: fold when tool has `params`, else `UNKNOWN_ARGUMENT` |
| #10 thread-safe session | `58d6d9f` | ❌ lost (plain dict) | port lock + `request_id` fallback |
| #4 plot `return_fig=True` | `e54c354` | ❌ lost (only `show=False`) | port |
| #5 kmf / balance / positivity injection | `e54c354` | ❌ lost — **proven broken** | port (bounded cache, prefer `uns`) |
| #11 base64 image naming | `e54c354` | ✅ works (`Image(path=)`) | none |
| #6 `classify_exception_error` | `811efe9` | ❌ lost | port + wire into **all** dispatch branches |
| #7, #8 export contracts, guide | `811efe9` | ✅ verified empirically | none |
| #9 `igraph` in `mcp` extra | `811efe9` | ❌ lost | port |
| runtime `Context` import | `f5ac5b2` | ✅ already present | none |

### Decisions taken (2026-08-21, with repo owner)

- **#3 → hybrid.** Fold plausible function kwargs into `params` when the tool exposes a
  `params` property and say so in the response; reject with `UNKNOWN_ARGUMENT` when the tool
  has no `params` or the key survives folding. Always strip `wait_for_previous`.
  Implemented in `ArgumentValidationMiddleware`, *not* by resurrecting `AgnosticFastMCP` —
  middleware is the composable hook and is already wired.
- **`summarize_edata` dropped.** The 14-tool surface (T5) is deliberate and
  `get_edata_snapshot` covers it. `tools/inspection.py` deleted rather than left as dead
  code. This is a decision, not a merge casualty.

### Own bugs found while dogfooding (not origin regressions)

- **Session state leaked across clients.** Every tool called `get_session()` with no `ctx`,
  so per-client isolation in `session.py` was dead code and client B saw client A's active
  `edata_id`. Fixed by threading `ctx` through *and* making the session thread-safe —
  fixing only the former converts a latent bug into a live race.

### Deferred (recorded, not blocking)

- Tier-1 check renders the whole DataFrame to Markdown before measuring length;
  short-circuit on `shape` first. Invisible at MIMIC-II scale, ugly on a real cohort.
- `_sample_rows`: O(n²) `remaining_idx` membership test; `df.loc[idx]` breaks on duplicate
  index values.
- `mcp = create_server()` at module scope purges the cache as an import side effect.
- `load_edata` returns the shared cached object. Object invariance across read-only ops was
  verified to hold today (`uns` and `obs.columns` unchanged after `run_get`/`run_plot`), so
  this is a known-safe assumption rather than a fix — but it is an assumption.
