# Adversarial verification: addendum lane "alignment" (I6N-AUD-06, -09, -16, -17)

Run date 2026-10-11. CPU only, through crispdm-run. No market data was read and no model was trained. Lane files were not modified.

## Reproduction
- No COMMANDS file exists in the lane. The command is embedded in ALIGNMENT_ADDENDUM.md and was rerun exactly from the lane dir,
  with $AUDITED = the i6d-weekly worktree (HEAD 269c2d13).
- Rerun out.json sha256 is cdd25c09de2f0b01b176808d494159c751eac868f64e2ca33ede0ce631c8665b. It is byte-identical to the retained file (cmp).
- The script sha256 is 67c44afa..., which matches.
- All 6 imported modules match their sha256 and git blob at 269c2d13. The audited worktree has no uncommitted .py changes, only docs.
- The proposal (d4341e72), i6e_clock_windows (d5d53f2d), assembly (6b1601a2), probes/cde/out.json (42e7cdcb) and check.py (ea521a73) all match.
- The audited worktree was left clean: no new files.

## Independent checks
These are in scratchpad/verify_alignment. independent_check.py has sha256 366b0669...; independent_out.json has sha256 a39be506....

1. **Keras `same` stride 2 (AL-02/AL-05).** It reads t on every n tested, which confirms 0 losses. But the reduced grid is NOT uniform and NOT t-anchored:
   - right-edge steps are {1,2} for n=5, 9 and 55 at k=2, and for every n at k=3;
   - for example, n=55 k=2 ends with blocks [52,53],[54];
   - earlier outputs read inputs later than their t-anchored label (t-2j).

   Origin-anchored padding gives steps {2} and no such reads. So `same` passes only the lane's t-membership test. The lane's acceptance test also drops the original AUD-06 "future-perturbation" half. `same` is not equivalent to origin-anchored padding: physical-position encoding after reduction assumes constant spacing.
2. **AL-04 wording.** transformer_conv ACCEPTS odd n when the factors divide it: n=9 [3,1,1], n=55 [5,1,1], and [1,1,1]. It refuses n not divisible by the factor product, which means odd n only under factor-2 stages. In all accepted cases t is preserved. It also refuses blocks=0 ("blocks must be a positive integer"), which is concrete evidence for AL-11 that was not retained.
3. **AL-07.** The lane omitted the constant per-feature branch (proposal:78, Conv1D causal k=3). The route's RF is 7, not 5, leaving 48 of 55 and 426 of 433 unread. A Conv1D-only route at n=55 or 433 is also not an N1 configuration, since N1 is n=5. The RISK holds qualitatively, but the numbers are not tied to the concrete route.
4. **AL-06.** The inherited probe_alignment with random-init weights passes on a stride-1 causal conv for 5 of 5 seeds, and the original 84/84 random-weight results agree with known weights. "Cannot tell weights apart from wiring" is overstated:
   - The concern applies to degenerate weights (zero, one-hot, or constant under LayerNorm).
   - For the coverage direction, the probe fails closed.
5. **AL-03.** The class does drop t on odd n, and its docstring is false. But every caller (fs4 build_predictor, and i6a ARCH_B through the same spec) enforces window = 4*latent_steps (fs4_temporal_predictor.py:82-83), so lengths are always even. The proposal does not name the class. For the plan it is a RISK, which mirrors AL-04's retraction. A docstring/contract defect is demonstrated in inherited code only.

## Classification and attribution
- **AL-13.** OWNER_REQUIREMENT is defensible, but the lane cites only proposal:37. The owner contract BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md:66-67 (sha256 604f9480...) explicitly requires preserving market/availability masks and not inventing prices. Cite it.
- **AL-07.** The correction "reject configurations whose RF < n" would override the owner's admissible search space (proposal:117-118, 85; "No exigir una capa costosa a toda ruta"). Downgrade it to declare/flag, as a METHODOLOGICAL_ALTERNATIVE.
- **AL-10.** It is marked RETRACTED, but the original understated the effect (Nyquist is exceeded from delta_x=4, not 7). It is a correction, not the withdrawal of an overclaim.
- **Final-only reduction.** The lane assumed factor 2; the proposal states factor 2 only for progressive reduction (:130).
- **Citations.** "proposal:97" should read :98.
- **Scope.** No calendar or market input is involved, so all numbers come from code and the proposal grid. Nothing in the lane overrides early stopping, seeds or the selection redo.

## Privacy
No host names, IPs, UUIDs, usernames, absolute home paths or tokens appear in the lane files.

## Verdict
REPRODUCED. The output is byte-identical. Most claims hold. Four need qualification before the lane is merged into the addendum: AL-02/05 (`same`), AL-04 wording, AL-07 numbers and "reject", and AL-06 wording. AL-03 also needs a plan-level RISK qualifier.
