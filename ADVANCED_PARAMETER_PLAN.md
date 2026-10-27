# Complete advanced analysis settings and Git publication

## Background and context
The user approved every remaining option discussed: minimum token length and numeric/URL/case cleaning; keyword and representative counts; independently optional neighbors and silhouette; sublinear TF and IDF smoothing; convergence tolerances and LDA priors. The user explicitly requested committing and pushing the completed project to https://github.com/hexiovo/PYsemantic.git. The remote is empty and the connected account has admin/push permission.

## Acceptance criteria
- Every setting is validated, connected to computation or result display, persisted through presets/projects/workers and exported in parameters.json.
- Original source text remains available for inspection and is never rewritten. URL removal and case handling act on analysis text; minimum token length and numeric-token filtering act on token-based features. Character/BGE behavior is documented.
- Independent neighbor/silhouette switches skip actual work and ETA stages, preserving clustering and theme results as appropriate.
- Topic terms, profile keywords, representative IDs and representative excerpts honor their configured counts in UI and exports.
- TF-IDF options affect word and character feature computation and applicable external-vector weighting; LDA retains raw count input.
- NMF, KMeans, LDA local-update/perplexity tolerances and optional LDA priors reach their intended estimators. LDA perplexity stopping is active only when evaluation is enabled.
- All tests, source and packaged integration checks pass. Windows screenshots cover new controls and configured profiles. Rebuild the desktop EXE.
- Commit the complete source, pinned resource manifest/tokenizers, tests and documentation to main and push without overwriting remote history. Exclude environments, runtime output, generated EXEs and model weights; explain model download for a fresh checkout.

## Implementation details
Extend AnalysisConfig, pure text preparation, vectorizers, estimators, skipped-stage handling and ETA signatures. Add grouped controls using existing CustomTkinter factories, retaining older settings defaults. Parameterize profile/representative rendering and export. Add focused numerical-oracle, bypass, persistence and UI checks. No new dependencies or delegated agents.

## Verification and recovery
Use the project .venv, unittest discovery, all-method self-test, runtime self-test and application-owned Windows captures. Preserve a timestamped source snapshot in build/ before modifications. Record completed checks in ADVANCED_PARAMETER_VALIDATION.md. Initialize local Git, use a repository-local identity verified against the connected account, inspect the exact staged file inventory, then commit/push and verify the remote SHA.
