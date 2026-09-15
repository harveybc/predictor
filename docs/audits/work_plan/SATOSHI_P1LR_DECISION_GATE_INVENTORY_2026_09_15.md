# The missing screen decision for `p1lr-decision@202`: what exists, what does not

§5 of `docs/handoffs/MUSASHI_WORKER_ACTIVATION_AND_R1_R6_COMPLETION_2026_09_15.md`:

    "locate the exact caller, required decision schema, governing design, consumed evidence
     and producer command... If an input is missing, name it and its producer... Do not set a
     gate true, invent a review record or launch the dependent experiment just to remove a
     refusal."

Nothing was set true and nothing was produced. This is the inventory.

## The caller

`p1lr-decision@.service` on **dragon** runs, as `ExecStartPre`:

    examples/systemd/p1lr_decision_gate_check.sh ${P1LR_SCREEN_GATE}

The service never starts unless that check exits zero. Its failure at 2026-09-14 23:30–23:32
was `status=4/NOPERMISSION`, repeated eleven times until systemd gave up with "Start request
repeated too quickly". The refusal it printed:

    REFUSED_SCREEN_GATE_MISSING: ...screen_verdict_bfbfd6443b849275.json does not exist

The gate is doing its job. The unit is refusing to run a DECISION budget without the screen
that authorizes it.

## The required decision

| | |
|---|---|
| schema | `agent_multi.p1_difficulty_lr_screen_verdict.v1` |
| required outcome | `SCREEN_VIABLE_REGION` — any other outcome is refused by the same script |
| governing design | `p1_difficulty_lr_factorial_v2.json`, schema `agent_multi.p1_difficulty_lr_factorial.v2`, in the pinned runtime checkout `.runtime/agent-multi-p1lr-v4-8758273f` |

## The producer, exactly

    tools/p1_difficulty_lr_factorial.py --screen-verdict \
        --contract <p1_difficulty_lr_factorial_v2.json> \
        --replica-proof <p1lr_replica_proof.v1 file>

`screen_verdict()` evaluates the contract's `mechanics_screen.requires` gates over **all 16
(seed, cell) records** — four seeds (101, 202, 303, 404) by the contract's `cell_order`. The
typed replica proof is **mandatory** (finding 225): without it, or with any unbound,
duplicate, foreign, altered or unloaded entry, the verdict is a typed refusal and
`replica_terminal_loads` stays false. That proof is produced by:

    tools/p1lr_collect.py --collection-root <root>      # agent_multi.p1lr_collection.v1

## What is missing, named

* **The 16 (seed, cell) screen records.** Searched on dragon and on omega under
  `~/.local/share/agent-multi`: `find ... -path '*p1lr*' -name '*record*.json'` returns **zero**
  on omega, and on dragon the only P1LR artefact of that family is
  `p1lr_v4_live_state_zero_update_genesis_20260818`, which is a **zero-update genesis**
  artefact — initial state, not screen records.
* **The typed replica proof.** No `p1lr_replica_proof` file exists on either machine.
* **The verdict itself.** No `screen_verdict*.json` exists anywhere under
  `~/.local/share/agent-multi` on either machine.

## A configuration defect found while looking

The gate path is declared **twice, differently**, and the two do not agree:

| where | path |
|---|---|
| unit default (`Environment=` in `p1lr-decision@.service`) | `p1lr_v3_explicit_close_collections_20260818/screen_verdict_bfbfd6443b849275.json` |
| per-instance override (`~/.config/agent-multi/p1lr-decision@202.env`) | `p1lr_collection_886b776e022d0d7c_20260812/screen_verdict.json` |

Neither directory exists on dragon. Whichever of the two is effective, the answer is the same
refusal — but an operator reading one file would be looking for the wrong artefact. This is
worth fixing regardless of the verdict, and it is not mine to decide which path is canonical.

## Can a deterministic adjudication be produced from existing evidence?

**No.** The design requires 16 records plus a typed replica proof, and neither exists. There is
nothing to adjudicate, and a verdict written without them would be the fabrication the order
forbids. The honest next step is to run the producer chain — collection, then screen verdict —
which is a campaign, not a repair, and belongs to whoever owns that front.

## What was NOT done, deliberately

The gate was not set true, no review record was invented, the dependent experiment was not
launched, and the failing units were left exactly as they are. `p1lr-decision@202` and
`p1lr-idle-guard` remain in `failed` on dragon; they are visible there rather than hidden by a
restart.
