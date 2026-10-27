# Common analysis parameters

## Background and context
The user requested the six prioritized parameter groups identified in the current application: maximum document frequency, n-gram ranges, neighbor count, optional clustering, model iteration limits, and random seed. Existing defaults and saved projects must remain compatible.

## Acceptance criteria
- Expose validated maximum document proportion (0, 1], independent word (1–3) and character (1–5) n-gram bounds, neighbor count (1–100), clustering switch, NMF/LDA iteration limits (1–1000), and a nonnegative 32-bit random seed.
- Preserve previous defaults, pass settings through workers, presets, project round trips and exports, and retain old settings compatibility.
- Word n-grams apply to word TF-IDF, LSA, NMF and LDA; BGE, PPMI and external word vectors retain unigram auxiliary matrices. Document these semantics and BGE preprocessing behavior.
- Disabled clustering skips both KMeans and silhouette computation, with an explicit result status and an accurate ETA plan.
- Meaningful parameter, matrix, seed, worker and persistence checks pass; capture and inspect the Windows methods page and existing workflow pages.

## Implementation details
Extend AnalysisConfig and validation, wire vectorizers, neighbor search, topic estimators and randomized stages. Add matching settings controls using existing UI factories and persist them via existing JSON settings/result formats. Update ETA signatures and iteration cost estimates, avoiding clustering stages when disabled. Add focused unittest and UI smoke coverage plus user documentation. No new dependencies.

## Environment and verification
The directory is not a Git checkout. A timestamped pre-edit source snapshot is retained under build/. Use the existing .venv, unittest discovery, integration self-tests, Windows application-owned screenshots and the existing build.ps1 packaging workflow. Record validation in PARAMETER_VALIDATION.md.
