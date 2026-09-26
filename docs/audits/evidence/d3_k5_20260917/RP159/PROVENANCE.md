# The design authority retained beside the nine-cell contrast

`DESIGN.json` here is the accepted design of the ECL modular contrast
`ecl-modular-contrast-20260924b`, retained beside its evidence so that the run can be
closed again without any data.

## Why it had to be retained now

The run itself was produced before `score_contrast` retained the design it authenticated,
so its design was only ever *re-derived*, and its run directory
(`.../ecl-modular-contrast-20260924b`) no longer exists on this host. RP158 named that gap
rather than papering over it: with no design beside the evidence, the pure closure refuses,
because inventing the authority it is about to check against is the defect, not the fix.

## What this file is, and what it is not

It is `seal_contrast(<the governed ECL delivery>, pred_len=96, seeds=(2021, 2022, 2023))`
re-derived on CPU, with **no fit, no checkpoint load and no inference of any kind**: the
seal reads the CSV header for the channel order and builds an unfitted model to report its
shapes. Its canonical content hashes to

    b5d5eee1b5fce981f4627352ed516d5745b1e16996507a8c63fffc53518cadd9

which is exactly the `design_sha256` the run recorded in `../RP155/CONTRAST.json` when it
produced these cells, and exactly what RP155, RP156, RP157 and RP158 each recorded. That
equality is the reason this object is admissible at all: a re-derivation is not authority
because it was re-derived, it is authority only when it reproduces the digest the run
recorded. A design that did not reproduce it would be refused by `authenticate_design`, and
there is a test for precisely that case
(`test_B1_design_refuses_a_re_derivation_that_is_not_the_runs_recorded_authority`).

It is **not** a new design, a new seal of a new run, or a claim that the original file was
recovered. Its operational `at` field carries the 2026-09-25 re-derivation time and is
excluded from the identity by name, as it has been since RP152.

    file sha256   a265bb6d7611dff4917cb247b58a93641e030929ed8063c6b25acccda86146ba
    re-derived    2026-09-26T01:20:28Z (UTC), CPU only, under crispdm-run
