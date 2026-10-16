# Short R2 Recovery Return

Code-only sidecar, 2026-10-15. Independent actual-path review is required before
adoption. Scope: new `tools/i6dmo/r2_recovery.py`, `tests/test_r2_recovery.py`,
and this note. No campaign, donor, scientific source, service or release hash
was changed. No fit, network request, mirror generation or commit was performed.

## API

```python
build_manifest(source_root, *, mirror_roots, source_pins, r0_target)
verify_manifest(manifest, *, source_root, mirror_roots, source_pins, r0_target)
load_manifest(path, *, expected_sha256, r0_target)
```

All return a private `i6dmo.r2_recovery.v1` document. Builders/verifiers perform
read-only operations and do not save the document. The parent owns exclusive
private manifest publication and freezes its file digest. `load_manifest`
requires that externally frozen digest; a self-resealed JSON document is not
an adoption authority. Failures raise typed `RecoveryError` with `.code`.

`source_pins` is exactly:

```python
{
    "effective_sha256": "<original EFFECTIVE_CONFIG.json file digest>",
    "plan_sha256": "<original PLAN.json file digest>",
    "code_commit": "<original source code pin>",
}
```

The intended short source retains code
`c34b1e48c17d83d0f2eaa08cfc8daed4f1afb861`. Actual roots, staged views and proof
paths remain private. `mirror_roots` contains two explicitly supplied,
locally readable mirror views. Neither may alias the source or the other;
hardlinked source files are not independent custody copies.

**Physical provenance is parent-owned.** One supplied view can be the existing
primary mirror, the other a private staging view of the secondary-host mirror.
Before invoking this helper, the parent must compare every staged file with
the SHA computed remotely through the existing authenticated transport, retain
that remote proof, and bind its digest plus original physical-mirror provenance
into the successor's recovery evidence. Two local directories alone do not
prove physical-host separation or remote staging provenance. The helper does
not read private deployment credentials, run SSH/rsync or attest those facts.

`r0_target` contains:

```python
{
    "contract": target_contract,
    "code_root": "<current target source checkout>",
    "expected_code_files": current_training_file_hashes,
    "expected_scales_sha256": target_scales_identity,
    "current_monitor_identity": target_monitor_identity,
}
```

The target is explicit on every verification. Its code files are read and
hashed, not trusted from a supplied hash map alone. Do not substitute the
source receipt's hash map for the target's actual `T.CODE_FILES` hashes.

## Returned Contract

- `source`: original campaign ID, code, externally pinned effective/PLAN hashes
  and private source root; `close_required=False` does not fabricate CLOSE.
- `custody.files`: authenticated source-relative file hashes, after direct byte
  comparison against both supplied views; publication count and trust scope.
- `pretraining_refs[feature]`: original stored-terminal `{path, sha256,
  unit_id, identity_sha256, identity, child_terminal_sha256}`, plus
  `donor_terminal` and `donor_npz` `{path, sha256}` refs.
- `unit_specs[feature]`: exact original pretraining PLAN unit, including its
  original identity and code. Parent factory adoption policy is separate;
  this sidecar does not rewrite those scientific identity components.
- `donors[feature]`: `{"terminal": {"path": ..., "sha256": ...}}`, the existing
  train/model import reference format.
- `donor_context`: reconstructed original branch context: ordered features,
  shapes/weight schema, original population/scaler/objective and provenance.
- `donor_provenance`: the same original provenance, including feature-to-unit
  and learned-corpus mappings. Campaign ID/code/ruling are never relabelled.
- `r0_reuse`: exact `r2_design.r0_reuse_decision` result and component failures.
- `r0_fit`: original authenticated R0 stored-terminal ref only if every reuse
  predicate agrees; otherwise `None`. `original_r0_fit` retains its custody ref
  for audit even when it cannot be adopted into this target.
- `failed_r2`: original failed stored-terminal ref and unchanged child terminal,
  including its original empty/null training state. It is not adoptable work.
- `manifest_sha256`: canonical document digest excluding this field.

Status is `ADOPTABLE` only when the donors authenticate and R0 is compatible;
otherwise authenticated donors can return
`DONORS_ADOPTABLE_R0_INCOMPATIBLE`. The latter retains the unmodified existing
predicate `FRESH_R0_REQUIRED` and names its failed components. The sidecar never
silently fits, queues a replacement or relaxes a code-file predicate. In
particular, changed `model.py`/`train.py` bytes for the new long mean monitor
must prevent adoption of an old short R0 under the strict predicate.

## Authentication

The effective config seal, externally pinned original file digests, PLAN binding
and closed TEST flag are verified. Exactly 19 original short features and
pretraining units must agree in order. Every completed source unit binds the
PLAN identity, original code, CHILD envelope, all child artifact hashes and its
LOCAL queue terminal/COMPLETE attempt. SQLite backups are opened read-only with
`immutable=1`; no sidecars or locks are created, and journaled backups are refused.

`REPORTS.json`, `PUBLICATION.json` and the outbox backup must agree. Every
published report's actual body must equal its sealed canonical body and its
DELIVERED row must carry the existing exact-readback row digest. Each completed
adopted unit's reports must occur in that verified publication population and
bind that unit's identity. A receipt saying DELIVERED is not sufficient alone.

Custody compares actual files, not a mirror's self-asserted inventory: effective
config, PLAN, publication reports, queue/outbox backups, stored/child terminals,
all donor artifacts and R0 receipt artifacts. Every used custody file must be
byte-identical in both supplied views. Digests are recorded for future binding,
and files are checked again before return. `mirror.manifest` is never called.

Donor terminal provenance, objective and ruling, feature order, seed, corpus
seal, saved CORPUS document, NPZ hash, branch identity and each actual NPZ tensor
digest are checked. The original O1 INNER-only stopping qualification remains
explicit; it is not relabelled as the new long mean FIT/INNER objective. The
source learned-corpus digest is authenticated, not recomputed from training
arrays here. At actual import, train must still recompute its target corpus and
network/scaler context and use the existing donor loader's identity checks;
recovering original provenance is not permission to bypass target compatibility.

R0 uses `donor_trust._verify_fit` for the existing scientific receipt, runtime,
capacity, model/weights and source/publication checks, after independent custody
verification. Only then is the locally established per-fit acceptance passed to
the existing R0 predicate. The whole-root donor verifier is not falsely called
accepted: the failed source has no CLOSE and is not a closed campaign donor.

## Failed R2 Production Shape

Production `_finish` stores a typed FAILED child and calls `q.fail`; it does not
insert a queue terminal row. A read-only inspection confirmed this source shape:
the original failed `r2.fit` has on-disk stored/child terminal bytes, unit state
FAILED, latest attempt FAILED, and **no queue terminal row**.

The helper does not invent that row or require the failure's reports to have
been published. Instead it binds the failed stored identity/envelope/artifacts
to the custody-verified original PLAN and queue unit/latest failed attempt,
including stored attempt index/kind/token/role. It then checks the source
attempt's `UNIT_SPEC.json` against original identity, campaign, the PLAN's
complete-effective-document digest, exact `C.arm_view(...)`, source code and
TEST closure. Child views inherit the root seal; they are not independently
sealed documents. This distinction was confirmed against the actual source.

Attempt files are not in the existing mirror payload. The failed UNIT_SPEC is
therefore explicitly recorded as **source-only**, bound to the mirrored
PLAN/queue identity, and its private path/hash are retained on the failure
reference. That scope is not silently promoted to mirrored custody. Failed R2
training must remain empty/null and fit time absent, including any saved
receipt; the source attempt is never patched or retried by this component.

## Verification And Handoff

Tests were written first; initial collection failed at the missing-module import.
The first production-shaped no-terminal test failed on `RECOVERY_QUEUE_SOURCE`
before that path was implemented. Final focused run:

```bash
python -m pytest --noconftest -p no:cacheprovider tests/test_r2_recovery.py -q
```

Observed **22 passed in 4.88 seconds**, under two CPU slots, 2-GiB virtual memory,
60-second timeout, hidden CUDA and one BLAS/OMP thread. Tests use only temporary
synthetic custody roots. Their read-only snapshot assertion checks bytes and
mtimes across source, both views and target code; a guarded-import test refuses
TensorFlow/model imports.

Bounded read-only production probes also passed: all 19 actual donor
terminal/corpus/NPZ contracts, 115 report bodies/readbacks, the actual failed
attempt without a queue terminal, and the existing R0 scientific per-fit
verifier (zero problems).

After the parent captured the remote mirror view, a real read-only
`build_manifest` probe also passed: **19 donors**, **115 used custody files**
directly compared across both views, and **115 delivered reports/readbacks**.
The remote view was checked against the parent's retained remote-custody file
hash map before invocation; remote transport/host provenance remains parent-owned.
The result was `DONORS_ADOPTABLE_R0_INCOMPATIBLE`, with
`r0_reuse.decision = FRESH_R0_REQUIRED` and the sole failed component
`code_files_sha256`. Current `design.py`, `model.py` and `train.py` differed from
the original R0 training-file hashes. The original failed R2 remained FAILED.

That smoke probe used the authenticated original scale identity and current
local code/monitor identity, without importing TensorFlow. It explicitly
asserted R0 incompatibility and published no manifest. The parent must derive
the full actual target and reauthenticate before freezing/adopting; the probe
is not authorization to reuse R0 or bypass target corpus/scaler checks. No fit,
network operation, TEST read, source mutation or manifest publication occurred.

| Requirement | Evidence |
| --- | --- |
| Original refs/context, no source writes | successful custody round trip and byte/mtime snapshot |
| Both views and delivered readback required | mirror file corruption, changed queue/outbox, forged readback |
| Donor identity/content checks | resealed wrong campaign/feature/objective/population/corpus/weights/budget |
| Strict R0 compatibility, no rerun | changed target train bytes yield named code-file incompatibility |
| Manifest is not a trust anchor alone | self-resealed altered context rejected; external file pin checked |
| PLAN hash includes the effective seal | full-document digest accepted; inner seal rejected even with matching custody copies |
| FAILED source without queue terminal | production-shaped absent terminal and unmirrored UNIT_SPEC |
| No source relabelling | changed failure UNIT_SPEC campaign rejected |

Parent handoff: supply the verified mirror views, build/freeze the private
manifest, reauthenticate it before copying, retain original unit identities as
ADOPT_ONLY, and bind remote staging proof outside the helper. R2 reporting is
fresh. Factory, transport, queue adoption and any target-authorized fresh R0 fit
remain parent-owned. Independent review and actual custody acceptance precede
dispatch; no recovery state has been written into the original campaign.
