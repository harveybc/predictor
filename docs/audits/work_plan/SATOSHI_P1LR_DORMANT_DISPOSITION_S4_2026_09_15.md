# P1LR: one canonical path, a dormant disposition, and a costed plan — no launch

S4 of `docs/handoffs/MUSASHI_TO_SATOSHI_COUNTERS_ARCHIVE_AND_TERMS_2026_09_15.md`:

    "Record one canonical candidate path based on its governing design and a dormant
     disposition; prepare a separate costed collection plan, not a launch. Do not schedule
     repeated decision attempts without evidence. Do not change running service configuration
     under this order."

Nothing was launched, no gate was set, no verdict was written and **no service configuration
was changed**. The inventory of the missing evidence stands as filed
(`SATOSHI_P1LR_DECISION_GATE_INVENTORY_2026_09_15.md`); this adds the three things S4 asks for.

## 1. The canonical path — and why both declared ones are wrong

I reported two divergent gate paths and said it was not mine to choose between them. Reading
the governing design settles it, and the answer is that **neither of them is right**.

First, a fact that changes the shape of the question: `tools/p1_difficulty_lr_factorial.py`
does **not** write a verdict file of its own accord. `_emit()` prints the payload to stdout and
writes a file only when `--output` is given. So the gate path is an **operator convention**,
not a producer output, and it has to be derived from where the design puts the rest of the
family. The producers name their artefacts inside the collection root:

    p1lr_collection_manifest_<experiment_identity>.json    (tools/p1lr_collect.py)
    p1lr_replica_proof_<experiment_identity>.json          (tools/p1lr_collect.py)

and the contract names that root and that identity itself:

    replica.collection_root  = ~/.local/share/agent-multi/p1lr_v2_collections_20260815
    experiment               = p1_difficulty_lr_factorial_20260815_v2
    replica.replica_host     = dragon

The one candidate path consistent with that design is therefore, beside its own inputs:

    ~/.local/share/agent-multi/p1lr_v2_collections_20260815/screen_verdict_<experiment_identity>.json

written by `--output`, where `<experiment_identity>` is the token the collector stamps on the
manifest and the replica proof in the same root. Against that:

| where it is declared | path | verdict |
|---|---|---|
| unit default (`p1lr-decision@.service`) | `p1lr_v3_explicit_close_collections_20260818/screen_verdict_bfbfd6443b849275.json` | wrong root — a **v3** tree the contract does not name |
| per-instance (`p1lr-decision@202.env`) | `p1lr_collection_886b776e022d0d7c_20260812/screen_verdict.json` | wrong root **and** wrong shape — undated, unidentified filename |
| the contract's own replica root | `p1lr_v2_collections_20260815/` | **canonical** |

This is a candidate, not an applied change: S4 forbids touching running configuration, and the
identity token must come from a real collector run rather than from me.

## 2. The dormant disposition, and what is happening right now

The order says not to schedule repeated decision attempts without evidence. **They are already
scheduled, and they are running.** Measured on dragon, 2026-09-15:

    p1lr-idle-guard.timer      active,  OnUnitActiveSec=15min, Persistent=true
    p1lr-idle-guard.service    failed,  status=2/INVALIDARGUMENT
    p1lr-decision@202.service  failed

`journalctl --user -u p1lr-idle-guard.service --since '24 hours ago'` matches a failure **130
times**. The timer has been waking every fifteen minutes for at least a day to re-attempt a
decision whose screen evidence does not exist, and failing each time. The gate is doing its
job — nothing false has been produced — but the retries are noise that hides a real signal and
they are exactly what the order names.

**Disposition: dormant.** The front is deferred from this acceptance, so the units should stop
being woken, and should stop being woken in a way that is *visible* rather than deleted: the
failed state is evidence and must survive. That means masking the timer, not clearing the
units.

Not applied — S4 forbids configuration changes under this order, and these are live units on
another machine. The exact operator commands, on **dragon**:

    systemctl --user stop p1lr-idle-guard.timer
    systemctl --user mask p1lr-idle-guard.timer     # dormant, and obviously so
    # deliberately NOT run: `systemctl --user reset-failed`, which would erase the evidence
    #                       that p1lr-decision@202 and the idle guard are refusing

Reversal is `systemctl --user unmask p1lr-idle-guard.timer && systemctl --user start
p1lr-idle-guard.timer`, and it belongs with the collection campaign below, not before it.

On **omega** nothing is scheduled: `p1lr-idle-guard.timer` is inactive and
`p1lr-decision@101.service` is inactive/dead. No action there.

## 3. The collection plan, costed — as a plan

This is what it would take to produce a real adjudication. It is **not** authorized by this
document and nothing in it was run.

**What must exist before any verdict:** 16 `(seed, cell)` screen records — four seeds
(101, 202, 303, 404) against the contract's cyclic Latin square, which assigns each of
`P1N_LR1E4`, `P1N_LR3E5`, `P1E_LR1E4`, `P1E_LR3E5` to every within-seed position exactly once
— plus the typed `p1lr_replica_proof` produced by `tools/p1lr_collect.py`, which finding 225
makes mandatory.

**The compute, in the vocabulary S1 established.** The contract's `budget_knobs` are executable
and exact: `epoch_timesteps: 20000`, `phase1_epochs: 1`, `phase2_epochs: 1`, and the runner
refuses anything else.

| quantity | value | where it comes from |
|---|---|---|
| training transitions per cell | 40,000 | one phase-1 epoch + one phase-2 epoch at 20,000 |
| cells | 16 | 4 seeds × 4 cells |
| **training transitions, total** | **640,000** | the product, exactly |

Two costs are deliberately **not** stated as numbers, because I have not measured them and a
plausible figure is worse than an absent one:

* **wall-clock.** No measured rate exists for this contract on this hardware. The one honest
  way to get it is a single pilot cell, measured with the S1 compute contract
  (`training_transitions_observed`, `optimizer_step_calls`, wall seconds) — which is itself a
  run, and so is the first costed step rather than an estimate;
* **memory.** Same: to be taken from the pilot cell under `crispdm-run -m`, not guessed.

**Sequence, each step gated on the previous one:**

1. one pilot cell, bounded, with the compute contract recording what it actually spent;
2. from that measurement, the full 16-cell cost, and only then a go/no-go on the campaign;
3. the 16 cells to `output_root`;
4. `tools/p1lr_collect.py --collection-root ~/.local/share/agent-multi/p1lr_v2_collections_20260815`
   to stage, seal and replicate, producing the manifest and the typed replica proof;
5. the replica host (`dragon`, per the contract) recomputes the tree digest and loads all 16
   terminals — finding 225's requirement, and the step that makes the proof typed rather than
   asserted;
6. `--screen-verdict --contract … --replica-proof … --output <the canonical path above>`;
7. whatever the verdict says. Including that it does not advance.

**Who owns it:** scientific campaign acceptance is Musashi's. The implementation is mine when
it is authorized. It is not authorized now, and this document does not ask for it to be.
