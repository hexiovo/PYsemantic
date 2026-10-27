# Advanced parameter validation

## Implemented scope
All six remaining option groups are implemented: token length and number/URL/case cleaning; topic/profile keyword and representative counts; independent neighbor and silhouette switches; sublinear TF and IDF smoothing; model convergence controls and optional LDA priors. Settings are included in presets, last settings, projects, worker requests, result caches and exported parameters.

## Source checks
- Compilation of semantic_toolkit, scripts and tests passed.
- Unittest discovery: **24 tests passed**, including ten new advanced-parameter tests and an extended worker/preset/project/export round trip containing all nondefault settings.
- All-method self-test: all **8 methods passed**, including local ONNX BGE and synthetic external vectors (`build/advanced-analysis-check`).
- Runtime self-test passed: real background workers, projects, logs, history, presets, ETA and cancellation (`build/advanced-runtime-check`).
- Existing result UI smoke passed with **361 documents**, pagination, source text, profiles, rename, project restoration, method changes and text-free caches (`build/advanced-result-ui`).
- Advanced parameter UI smoke passed: every new control, saved settings, old-settings defaults, invalid values, disabled state and configured profile representatives (`build/advanced-parameter-ui-final`).
- Visually inspected the grouped settings and configured profile on Windows at the current monitor scale. Other scales and a separate clean machine were not tested.

## Correctness coverage
Reference comparisons verify word and character TF-IDF weighting for smoothing/logarithmic options and external-vector aggregation against the weighted matrix. Tests prove that URL/case preparation leaves source text unchanged, BGE receives prepared sentences rather than token-filtered fragments, disabled calculations do not call estimators, ETA omits disabled stages, estimator tolerances/priors are forwarded, LDA probabilities normalize, and profile/CSV counts match configured limits. Older result configs retain default profile behavior.

## Repository preparation
The user specified the empty public repository `hexiovo/PYsemantic`. Connected account metadata confirms push/admin permission. Local Git uses main and the verified account's GitHub noreply identity, scoped to this repository. Environment, generated builds, runtime data, temporary directories, EXEs and ONNX weights are ignored; pinned model metadata and tokenizer resources are included. README describes preparing a fresh clone and downloading the pinned model before offline use.

## Recovery
Pre-edit snapshot: `build/before-all-parameters-20261027-090104/`. Final packaging log: `build/advanced-build-final.log`.

## Empty cleaned text regression
A targeted test reproduced a URL-only document being represented by its replacement space in character TF-IDF. Text preparation now maps an all-whitespace URL-cleaned result to the empty string; the document receives a zero representation and is excluded from comparisons. The regression test and the packaged character-worker check cover this case.

## Final packaged executable
- Final build completed and updated `dist/ChineseSemanticToolkit.exe` and the desktop APP copy (173,659,878 bytes).
- Final EXE all-method self-test exited 0: eight methods and exports (`build/advanced-exe-analysis-final`).
- Final EXE UI smoke exited 0 with eleven screenshots, real background execution, project restore and logs (`build/advanced-exe-ui-final`); the packaged advanced settings page was visually inspected.
- Packaged NMF, LDA and character TF-IDF workers passed all nondefault parameter, filtering, independent-switch, topic/profile-count and empty-cleaned-document checks (`build/advanced-exe-workers-final/checks.json`).
- Desktop and dist EXE SHA256 match: `741079f8e2b1f55026e8f56b8bdd56563efa3af51187023c36413d3f19fc5941`.
- Staged source inventory contains 42 files; largest file is the third-party notices (1,142,503 bytes). No generated environments, model weights or runtime output are staged. New tests, scripts, plan and validation documentation pass the scoped whitespace check; inherited formatting in upstream notices and existing files is preserved.
