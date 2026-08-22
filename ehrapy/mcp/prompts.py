"""Agent-facing workflow guidance."""

WORKFLOW_PROMPT = """# ehrapy MCP workflow (v0)

Call `get_runtime_context` before file operations and `get_package_info` for the full API map.

## Namespaces → MCP tools

| ehrapy module | MCP tool | Functions |
|---------------|----------|-----------|
| `ep.pp.*` | `run_preprocessing` | QC, encode, impute, normalize, filter, PCA, neighbors |
| `ep.tl.*` | `run_analysis` | survival, causal, embedding, leiden, feature ranking |
| `ep.get.*` | `run_get` | obs_df, var_df, rank_features_groups_df |
| `ep.pl.*` | `run_plot` | all plots → PNG in cache |
| `ehrdata.io.*` | `run_io` | read/write h5ed, csv, zarr, pandas |
| `ehrdata.dt.*` | `load_demo_dataset` | mimic_2, physionet2012, … |

Discovery: `list_ehrapy_functions`, `get_function_help(namespace, function)`.

## Typical flows

### Exploration
1. `load_demo_dataset("mimic_2")` or `ingest_dataset(path)`
2. `get_edata_snapshot`
3. `run_preprocessing("qc_metrics", params={...})`
4. `run_preprocessing("encode", params={"autodetect": true})`
5. `run_plot("missing_values_matrix")`

### Clustering
1. `run_preprocessing("encode", params={"autodetect": true})`
2. `run_preprocessing("simple_impute")`
3. `run_preprocessing("highly_variable_features")`
4. `run_preprocessing("pca")`
5. `run_preprocessing("neighbors")`
6. `run_analysis("umap")`
7. `run_analysis("leiden")`
8. `run_analysis("rank_features_groups", params={"groupby": "leiden"})`
9. `run_get("rank_features_groups_df", params={"group": "0"})`

### Survival
1. `run_analysis("stratified_table_one", params={...})`
2. `run_analysis("kaplan_meier", params={...})`
3. `run_analysis("cox_ph", params={...})`
4. `run_plot("cox_ph_forestplot", params={...})`

### Causal
1. `run_analysis("positivity_check", params={...})`
2. `run_analysis("covariate_balance", params={...})`
3. `run_analysis("iptw", params={...})`  (or aipw, g_computation, t_learner, …)
4. `run_plot("love_plot", params={...})`

## Handles
- `edata_id` — cached EHRData on the MCP host (h5ed in cache_dir)
- `fork_edata_handle` before destructive edits
- `export_edata` to write results to a user path

## Params
Pass function kwargs as a JSON `params` dict. Lists become tuples when needed.
Most preprocessing/analysis mutates EHRData in place; set `in_place=false` to skip persist.
"""
