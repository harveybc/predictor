# CL14-CL19 return: the five counterexamples closed, the collector built

Review: `MUSASHI_CL08_CL13_REVIEW_2026_09_24.md`. Heads: news-signal `317dd35`,
M5PHET `3f4a833` (unchanged, now the declared pin), feature-eng `d081d0f`,
predictor this commit.

Every finding was reproduced here first, frozen as a failing test, and then
repaired. PRE and POST are side by side in
`docs/audits/evidence/CL14_CL16_20260924/`.

## The five, before and after

| Finding | Before | After |
|---|---|---|
| F1 pin | imports, refuses **every** user question | pin is the published runtime; a real `ask` returns SHADOW_ONLY in a clean venv |
| F2 wrong key | intact record of question B served as A's DUPLICATE, 0 integrity failures | not reused, quarantined under a name carrying its bytes, reported with its reason |
| F3 options | `fits` computed after the 48-token slice, so it could never be False | counts what was **asked**, names each option that would be cut, refuses |
| F4 boundaries | consensus published after the release turned 0.4 into 0.0 | `release_surprise` 0.4 and `available_surprise` 0.0, both named, never merged |
| F5 availability | UNKNOWN row ingested and produced a number | required field; UNKNOWN archived, never point-in-time; undeclared refused by name |
| F6 mutation | a read handed out stored rows; actual 2.9 → 99 with the vintage unchanged | reads are detached copies; the same mutation leaves 2.9 |

Each repair carried a lesson worth stating once. A pin that imports is not a pin
that works. Content integrity says nothing about placement. Counting after a slice
measures the slice. A rule in a helper nobody must call is not a rule. And a public
read that hands out stored state is a way to edit history without touching it.

## CL14: the installation now demonstrates the feature

`docs/INSTALL_VERIFICATION.json` records a fresh venv where pip resolved
`3f4a833` from the declared pin alone -- no editable install, no PYTHONPATH, no
sibling checkout -- and where `news-signal ask` returned **SHADOW_ONLY** for
"Which economy is named?", stored the record and replayed it in a new process.

That needed a backend a clean machine can run, so `NEWS_SIGNAL_BACKEND=fixture`
selects the package's declared `NON_MODEL_FIXTURE`, whose kind is stamped into
every receipt it touches. It proves the path, and it measures no model.

## CL15: same-key contention, not just distinct ids

The record write is an exclusive create. Eight writers racing for one key all walk
away holding the same digest, where before two of them held different ones because
the envelope timestamps differed. Unreadable JSON is reported as a failure of that
key instead of being raised past the caller, and every quarantined version is kept.

## CL16: two boundaries, both true

`release_surprise` is the actual against the last consensus **published** before
the actual was published -- what the market was surprised by. `available_surprise`
is the same actual against the last consensus this system had **received** when the
number reached it -- what we could have computed. On the review's fixture they are
0.4 and 0.0. A revision is reported as `revision_surprise` against the same
pre-release expectation, so it never rewrites the release.

## CL18: the collector, built without the entitlement

A durable queue with the receipt clock owned by us: an item that supplies its own
`received_at` is refused, because a source that can set when we held it can make a
late item look timely. Accepted once under contention, revisions linked and the
earlier text kept, failures retained through retries, tampered entries never
dispatched, and the whole queue readable in a new process. Draining the recorded
corpus gives 11 events and 1 revision; a second pass adds nothing.

## Status, all lanes

| Lane | State | Concrete missing object |
|---|---|---|
| CL14 install | **done**, verified in a clean venv | — |
| CL15 persistence | **done** | — |
| CL16 calendar | **done** on fixtures | a named source before any governed sample |
| CL17 question → native forecast | **not started** | — |
| CL18 collector + queue | **done** | a documented feed entitlement for real capture |
| CL18 labelled relevance set | **not started** | independent annotators and a frozen split |
| CL18 MT5 demo / Alpaca paper | **not started** | documented account access and the existing risk mandate |
| hierarchy / causal / policy / DOIN | **not started** | — |

## Costs

CPU only this block: the three suites, two clean-environment installs and both
probes, all under `crispdm-run`, none above 3G. **No GPU was used**: nothing here
needed inference, and the two-question parity evidence from the previous block
stands unchanged. The 5090 is idle. Doctoral numbers untouched: R0 0.371174,
R1 0.374584, R2 0.368596.

Tests: news-signal **117 passed, 1 skipped** (the differential against the pinned
upstream builder, which runs where `laya` is installed); feature-eng **84 passed**
in the suites that collect, with the eight legacy collection errors unchanged and
predating this work.
