# Recovery and retention of the frozen Huber/AdamW v2 design (`be2e776e…`)

Satoshi, successor technical lead — 2026-09-26, under the owner's grant of 2026-09-26.
Nothing in this document is issued under Musashi's name, and nothing in it is a design
I authored.

## 0. Verdict, first

**The bytes were recovered.** The Huber/AdamW v2 design that the four arms of
2026-09-21 rest on exists, is intact, and re-derives the digest
`be2e776e5c64a8422a6411447a4cbeba6e5607c7158456f9a64f89a4244b6965` **exactly**
under the corpus's own digest rule. It was never in any git object store; it
survived only in the campaign's private run root, which is why it read as absent.
It is now retained in the repository at
[`docs/audits/evidence/HUBER_ADAMW_2026_09_21/DESIGN.json`](../evidence/HUBER_ADAMW_2026_09_21/DESIGN.json),
beside the `REPORT.json` that cites it, and pinned by
`tests/test_huber_v2_design_retention.py` so it cannot silently vanish again.

No reconstruction was performed and none is needed: this is the retained document,
not a rebuild of it. Outcome 2 of the assignment does not apply.

**`MOD-CONF` is not yet freezable.** The design-retention blocker is cleared. The
second named item is not: `MOD-CONF` still needs **one sealed design freezing the
confirmatory method** for H1/H2/H3 under the proposal's reserved protocol. I did
not write it, and I will not — see §7.

## 1. What the digest is a digest of (established before searching)

Comparing in the wrong domain would have made present bytes look absent, so the
domain was fixed first, from the code that computes it rather than from any
description of it.

`tools/df_e1_huber.py` seals the design and then sets

```
d["design_sha256"] = P._module("df_mod_e0").sha_obj(d)
```

and its `validate()` refuses unless

```
E.sha_obj({k: v for k, v in d.items() if k != "design_sha256"}) == d["design_sha256"]
```

with `tools/df_mod_e0.py`:

```
def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
```

So the digest is **sha256 over the canonical JSON serialization of the design
object with its own `design_sha256` field removed** — keys sorted, separators
`(",", ":")`, `default=str`. It is **not** the raw file bytes, **not** a git blob
hash, and **not** the file's on-disk formatting. The recovered file's raw-byte
sha256 is `cdd1611c41d7822169e481e585278ab9c5f59e68ab049c6bf411b01f303599a1`,
which matches nothing in the corpus and would have been the wrong thing to hunt.

The audit tooling named in the assignment, `tools/df_rp49_rp64_audit.py`, does not
exist in this repository. The authority on the digest is the pair above.

## 2. Where the bytes were found

| | |
|---|---|
| Path | `~/.local/state/crispdm-data-foundation/huber_adamw_v2/DESIGN.json` |
| Size | 7,375 bytes |
| Written | 2026-09-21 00:39:25 local |
| Raw-byte sha256 | `cdd1611c41d7822169e481e585278ab9c5f59e68ab049c6bf411b01f303599a1` |
| `design_sha256` field | `be2e776e5c64a8422a6411447a4cbeba6e5607c7158456f9a64f89a4244b6965` |
| Re-derived by `sha_obj` | `be2e776e5c64a8422a6411447a4cbeba6e5607c7158456f9a64f89a4244b6965` |
| Match against target | **exact, all 64 hex characters** |

That run root is the one the results work plan already names
(`MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md`, "Private run root"). The document
was not lost; it was simply never copied out of a private, untracked,
non-backed-up directory into the repository. The blocker was a retention gap, not
a destruction event.

## 3. How the four arms bind to it

Every one of the twelve cells (four arms × three seeds) carries the digest in its
own artifact, so the binding is checkable without trusting the report:

- `attempts/*/cell.json` — **12 of 12** carry
  `design_sha256: be2e776e…`. Zero mismatches.
- `TERMINALS/*.json` plus `TERMINAL_RECEIPTS.json` — the digest appears in the
  reconciled receipt set.
- The twelve pre-run governance envelopes under
  `outbox/units/<cell>/sent/*.json` also carry it.
- `REPORT.json` (retained in this repository since 2026-09-21) carries it as its
  top-level `design_sha256`.

**The design's own integrity seal still holds.** The design seals the sha256 of
five scientific sources. Against the execution revision the work plan names,
`73f3bab`, all five match exactly:

| sealed source | at `73f3bab` |
|---|---|
| `df_e1_governed.py` | matches |
| `df_e1_huber.py` | matches |
| `df_e1_phase1.py` | matches |
| `df_e1_pilot.py` | matches |
| `df_mod_e0.py` | matches |

**Named limitation.** Against current `HEAD` (`5c9a14d5`), three of the five have
moved on (`df_e1_governed.py`, `df_e1_huber.py`, `df_e1_phase1.py`). `validate()`
would therefore refuse to re-execute this design at `HEAD` — correctly, because
the scientific sources changed. Re-execution requires `73f3bab`. Re-sealing
against `HEAD` would produce a **different digest** and a **different design**,
not this one; that is not a repair and must not be done to make a suite pass.

## 4. Evidence that the design predated the runs

This matters because it is the one thing a reconstruction could never have
supplied, and it is available here only because the real bytes were found.

- `DESIGN.json` is the **oldest JSON file in the entire run root**: all 39 other
  JSON artifacts are strictly newer by mtime. Written 00:39:25; the first cell
  record (`attempts/huber_adamw_s1/cell.json`) is 00:44:01, the first terminal
  00:44:02, the report 00:56:41.
- The design is written through `df_d3_campaign.write_once`, which refuses to
  overwrite an existing path, so it could not have been rewritten after scores
  were seen.
- Each cell re-read `DESIGN.json` from the root and stamped the digest into its
  own record before its terminal was accepted; the twelve stamps agree.
- The design's `pilots` list is empty and its `cells` list is the full twelve-cell
  population, which `validate()` compares against the code's generated
  expectation — so no cell was added or dropped after the fact.

This is a strong provenance chain for a development diagnostic. It is not a
notarized external timestamp, and I do not claim one.

## 5. Where I looked, and how, so nobody repeats the search

The bytes were found in the run root, but the question "was it retained?" needed a
negative answer established by measurement, not assumption. All scans were
read-only and memory-capped at 3 GiB under `crispdm-run`.

| Where | How | Result |
|---|---|---|
| `predictor` complete object store | `git cat-file --batch-all-objects --batch-check`, then `--batch` over **all 13,568 blobs** — this enumerates unreachable and dangling objects too — grepping for `df_e1_huber_design.v1` and for the literal digest, then JSON-parsing and re-deriving every hit | **No design blob.** 12 hit blobs: 4 versions of `df_e1_huber.py`, 2 versions of the results work plan, 1 E1 seal source, 1 E1 seal artifact, 2 owner closure tables, 1 partial-seal document, and `REPORT.json` (5,481 bytes) — which carries `design_sha256` but is the report, and does not re-derive |
| `agent-multi` complete object store | same method over **all 38,965 blobs** | **Zero hits.** `agent-multi` never held the design or the digest |
| dangling objects, both repos | already covered by `--batch-all-objects`; additionally `git fsck --lost-found` (125 objects in `predictor`, 148 in `agent-multi`) | added nothing beyond the full-object scan |
| reflogs, stashes, `ORIG_HEAD`, both repos | `git stash list` (0 in each), `git rev-parse ORIG_HEAD` | nothing |
| all **50** `agent-multi` worktrees, including those under `.runtime/` and `.worktrees/` | recursive content grep for the digest and the schema tag across every worktree path from `git worktree list --porcelain` | **zero hits** |
| retained evidence, both repos | `docs/audits/evidence/` inventory; `find -name 'DESIGN*.json'` | 13 retained `DESIGN.json` files for other runs (RP66/RP74/RP82/RP90 blocks) — precedent for where this one belongs — but none for this factorial |
| the four arms' own artifacts | read `cell.json`, `TERMINALS/*.json`, `TERMINAL_RECEIPTS.json`, `DELIVERIES.json`, the outbox envelopes | all reference the design **by digest**; none embeds its bytes inline |
| the private state tree | content grep for the digest across `~/.local/state/crispdm-data-foundation/` | 40 carriers, all inside `huber_adamw_v2`, exactly one of which is the design itself |

Conclusion of the search: the design existed in **one** place on this machine and
in **zero** places in version control. A single `rm -rf` of that run root, or a
reimaged machine, would have destroyed it permanently. That is the risk this
document closes.

## 6. What was retained, and the pin

`docs/audits/evidence/HUBER_ADAMW_2026_09_21/DESIGN.json` — the recovered bytes,
copied verbatim. **The file was not reformatted, re-serialized, or edited in any
way**, because any such change would break either the raw-byte identity or the
canonical digest. It retains its original absolute `source_run.root` path because
that string is inside the digested object.

`tests/test_huber_v2_design_retention.py` — seven tests, all passing, deliberately
free of TensorFlow and of any run root so they fail fast anywhere:

1. the design file is present;
2. its raw bytes are unchanged (`cdd1611c…`);
3. the digest re-derives to `be2e776e…` under the rule restated locally;
4. the local restatement agrees with the shipped `df_mod_e0.sha_obj` (guards the
   pin against drifting away from the corpus helper);
5. it is the four-arm, twelve-cell factorial the report scored;
6. `REPORT.json` binds to the same digest;
7. the five sealed sources match their digests **at `73f3bab`**, not at a moving
   `HEAD`.

Mutation-checked, so the pin demonstrably bites:

| mutation | result |
|---|---|
| design file deleted | 6 of 7 fail |
| one hyperparameter altered (`huber_adamw.weight_decay` 0.004 → 0.005) | 3 of 7 fail, including both digest tests |
| restored | 7 of 7 pass |

### What the retained design pins (summary; the file is authoritative)

Recorded here for readers, not as a substitute for the document. Loss/optimizer
recipes per arm (`mae`/`huber` × `adam`/`adamw`, Huber `delta` 1.0 in standardized
units, AdamW `weight_decay` 0.004, Adam 0.0); learning rate 0.003, batch 64,
`max_updates` 4,000, early stopping `patience_epochs` 3 with `restore_best`,
common monitor `val_mae`; seeds 1/2/3 with shared per-seed initial weights;
factors moved `[loss, optimizer]` against ten held factors; task `W60_h60`
(60-step window, 60-step horizon, purge 120, seven channels with target history);
prepared data consumed **by digest** from the source run
(`data_sha256 70485ac9…`, `panel_sha256 b3192c0b…`); resource limits 1,800 CPU s
per child and 14,400 per campaign, 4 parallel children; and three explicit
reading rules, including that three seeds are development evidence rather than a
confirmation and that an arm stopped at the update ceiling is **CENSORED** in
either direction.

## 7. What `MOD-CONF` still needs

`MOD-CONF` in `docs/tres_temas_entrevista/program_v3/PROJECT_METHOD_STATE.json`
stands as: owner `Musashi + Satoshi`, status `NOT_STARTED`, `depends_on [MOD-E1]`,
`evidence []`, deliverable *"H1/H2/H3 effects under the proposal's reserved
protocol"*, next action *"Review fixed method before independent E0 synthetic and
E2 public reserves"*.

Cleared by this document:

- **The standing-recipe design is retained.** `be2e776e…` is present, verified,
  committed and pinned. Anything in `MOD-CONF` that rests on the programme's
  standing Huber/AdamW recipe can now cite a retained document instead of a
  dangling digest.

Still outstanding, and **not** delivered here:

1. **One sealed design freezing the confirmatory method** — the reserved protocol
   under which H1/H2/H3 will be estimated: the hypotheses as estimands, the
   reserved splits, the multiplicity and stopping rules, the coverage certificate
   any interval claim would need, and the refusal conditions. It must be sealed
   **before** the independent E0 synthetic and E2 public reserves are touched,
   which is the whole point of a confirmatory design; sealing it afterwards would
   make the reserves developmental and destroy their value.
2. **`MOD-E1` closure.** `MOD-CONF` depends on it, and E1 stands partially sealed
   with named gaps. Retaining this design does not close those.
3. **`MOD-CONF`'s `evidence` list is empty** and stays empty. I have not changed
   the task's status, because recovering a prerequisite document is not executing
   the module.

### The constraint I am not working around

`MOD-CONF`'s co-owner is a reviewer who is not returning, and the owner's grant of
2026-09-26 does not make me the author of a design I did not write.

**Recovering bytes that already exist is not authorship** — that is what this
document does, and the recovered file is verifiably byte-identical to what was
sealed on 2026-09-21 under a digest computed before the runs.

**Writing the missing confirmatory design from scratch would be authorship**, so I
did not do it, and I did not draft, sketch, or stub it. The gap is reported as a
gap. `MOD-CONF` remains blocked on item 1 above, and unblocking it requires either
the owner's own decision on who authors the confirmatory design or an explicit
instruction naming me its author — neither of which I am assuming.

## 8. Report

```
MOD-CONF prerequisite — recover and retain the frozen Huber/AdamW v2 design
repo/branch/tip: predictor / satoshi/huber-design-recovery-20260926 / see commit
files: docs/audits/evidence/HUBER_ADAMW_2026_09_21/DESIGN.json (recovered, verbatim)
       tests/test_huber_v2_design_retention.py (new, 7 tests)
       docs/audits/work_plan/SATOSHI_HUBER_V2_DESIGN_RECOVERY_2026_09_26.md (this)
suites (after reinstall): test_huber_v2_design_retention 7/0 · test_df_e1_huber unchanged
acceptance: bytes RECOVERED · sha_obj re-derivation == be2e776e5c64a8422a6411447a4cbeba6e5607c7158456f9a64f89a4244b6965 exact ·
            12/12 cell.json bind to the digest · 5/5 sealed sources match at 73f3bab ·
            design mtime precedes all 39 other run-root JSON artifacts ·
            search negative: 0 design blobs in 13,568 predictor blobs and 38,965 agent-multi blobs, 0 hits across 50 agent-multi worktrees ·
            mutation check: delete -> 6/7 fail, alter one hyperparameter -> 3/7 fail, restore -> 7/7 pass ·
            owner checkouts unchanged: agent-multi 33 untracked / 8 modified, as found; .git/lost-found created by my fsck removed in both repos
what is NOT done / refused / not measured:
  - The sealed confirmatory design for H1/H2/H3 is NOT written. Refused as authorship of a design
    I did not write; MOD-CONF stays blocked on it.
  - MOD-CONF status and evidence list NOT changed; MOD-E1 closure NOT addressed.
  - The design was NOT re-sealed against HEAD; 3 of 5 sealed sources have moved on since 73f3bab,
    so validate() refuses at HEAD by design. Re-execution requires 73f3bab.
  - No arm was rerun and no score was recomputed; the 2026-09-21 measurements are untouched.
  - Provenance is filesystem mtime + write_once + twelve agreeing stamps, NOT an external
    notarized timestamp.
```

— Satoshi, successor technical lead, 2026-09-26.
