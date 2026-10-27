# Result exploration implementation plan

## Background and context
The application already computes document metrics, nearest neighbors, topic weights and KMeans labels. Its result tables currently truncate at 300 rows and do not link to source text. Users need to inspect the evidence behind results and understand each group. The approved scope is recommendations 1 and 2 from this chat.

## Acceptance criteria
- Document and neighbor tables support literal ID filtering, cluster filtering and complete pagination; neighbors additionally support a validated cosine range.
- Double click, Enter and an explicit button open the selected document or both neighbor documents with full text and source information. Text-free historical caches show an explicit unavailable state.
- Cluster profiles show counts, mean TF-IDF keywords, representative documents and all members. Names are editable per method and retained in saved projects.
- Topic profiles show keywords and members using explicit method-specific semantics, including separate positive and negative LSA poles.
- Switching methods or corpora cannot display stale profiles or raw text from another result.
- Unit and integration checks pass, existing runtime/UI smoke checks pass, and Windows screenshots cover the new workflows.

## Implementation details
- Add a pure result-exploration module for filtering, pagination and group summaries. Keep sparse matrices sparse; compute representatives against their cluster centroid, without an all-pairs matrix.
- Add reusable paged tables and a result explorer component, using existing UI factories and style tokens. Compute profiles in a background thread and ignore superseded work.
- Keep custom cluster names in existing per-method project runtime metadata. Pass names to exports so CSV output remains useful; retain compatibility with older projects.
- Update the result page and clear its state when a new corpus is loaded. Use the corpus attached to the selected result, never an unrelated corpus with matching IDs.
- Add focused unittest coverage and a dedicated UI smoke driver; run existing self-tests and build the Windows executable.

## Environment and verification
This directory is not a Git checkout; no worktree or repository fast-gate script is available. Preserve a local pre-edit snapshot of existing changed files. Use the project Python environment, unittest, compilation, runtime smoke checks and UI screenshots as the local validation gate. No new dependencies are needed.
