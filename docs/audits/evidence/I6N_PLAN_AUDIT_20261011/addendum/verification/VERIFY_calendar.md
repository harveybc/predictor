# Adversarial verification: addendum lane "calendar" (I6N-AUD-07/-08/-10/-11/-29; owner point 6a)

Date: 2026-10-11. CPU only, each job run through crispdm-run, one at a time. No lane file was modified.
Only ts members and pre-2025 rows were read.

## 1. Reruns (per COMMANDS.sh, outputs written to this scratch dir)
All seven outputs are byte-identical to the retained files (cmp):
identify_input.out, rerun_bar_clock.out, rerun_gaps_2024.out, rerun_wmc_calendar.out/.json (also identical to probes/a),
calendar_attribution.out/.json, bar_label_check.out.
COMMANDS.sh is NOT runnable as written: it uses $AUD and $AUDITED but never sets them, and under `set -eu` it stops
with "AUD: parameter not set". It ran only after both were exported by hand.

## 2. Digests
Every lane file matches MANIFEST.json (MANIFEST.json itself = 3329324c...). Every input matches INPUTS.json:
INPUT.npz 897caee1, I6-E-M INPUT d9015867 (v1 and v2), DESIGN.json c58e9223, the audited modules 5f6a06d4/e4ed00fc/7fcaad67/656275b7,
the documents MWC 604f9480 / PROP d4341e72 / I6E 129cea8c, and the probes 20f54499/2eecc3db/5a2e6934. The audited worktree is at
HEAD 269c2d13, and none of the cited tools is modified.

## 3. Independent checks (scripts and outputs in this dir)
- indep_sessions.py: splits weeks at gaps over 24 h instead of Saturday 00:00 UTC. It confirms that all 22 DST-gap weeks in 2012-2018
  are normal (Sun 18:00 / Fri 17:00 NY) and that 19/21 gap weeks in 2019-2024 open Sun 17:00 and close Fri 16:00 (the other two are
  Sun 18/Fri 15 and Sun 18/Fri 16). The count of normal aligned weeks depends on how weeks are split: 270 with this split, 272 with the
  lane's split.
- postopen_usage.py: counts how the post-open bars (O < END label < the Mon 00:00 UTC fit_cutoff) enter i6m_data.partitions.
  Under NY there are 86 bars in 52 segments; under OBS-END there are 89. They enter NO fit (gradient) row. They enter inner-validation
  rows only as TARGETS: h=72 uses 86 (NY) or 89 (OBS-END) of them, h=2 uses 3, h=48 uses 1 (NY) or 2 (OBS-END), and h=3,4,5,6,24 use 0.
  No inner-validation row has a post-open ORIGIN under NY; under OBS-END, 3 rows do (h=2, in the DST-gap weeks).
- Effective converter: recomputing i6e_statistical_campaign.code_digest() from the HEAD bytes gives 62dcebe0..., which equals
  statistical72_v1/DESIGN.json code_sha256. So the converter that produced INPUT is byte-identical to the one inspected (7fcaad67).
  However, the INPUT ts member comes from fs4_weekly_wrapper.DataStore over the feature-eng PS1 parquets
  (tools/i6e_statistical_campaign.py:57-67), not from the converter. The converter constrains it only through exact_price_values,
  which raises if any ts is missing from the END-labelled converter output.
- Arithmetic re-derived by hand, all correct: B-A=96 and A->cut=67 under NY; 53x24=1,272; the straddle-week split 36/2/56; the
  binding condition B-cut<24 and the 4/9 NY settings; OBS-END A->cut=68 in the March gap weeks; NY first week h72 M=11.

## 4. Classification review
- 08b (DST gap): the demonstrated part is the inconsistency of session boundaries against the assumed NY calendar and against the
  dataset's own 2012-2018 gap weeks. The words "systematic one-hour clock offset", "as-of joins off by 1 h" and "A/B misaligned" assume
  that the bar clock, not the session, is wrong, and the lane itself leaves that cause unverified. Those consequences are a RISK
  until the raw source is checked. That source is identifiable by DESIGN.json original_price_sha256 c746f344..., and only its pre-2025
  rows would be needed.
- 11b: "fits can see 86 observed post-open bars" overclaims. See postopen_usage: the bars enter only inner-validation targets, almost
  entirely at h=72, and enter no fit row. The non-conformance with MWC:14-19 itself stands. It concerns an object that predates the
  MWC, and the MWC states that it does not claim compliance of existing runners. Whether it bears on I6-N is the RISK in 11a.
- 08a: the retraction stands, but its evidence chain names the wrong producer of ts (see section 3). The lane should retain the
  code_digest match and the exact_price_values constraint.
- 10a: classed "RETRACTED", but the residual (which split a row belongs to when t and t+h straddle) is essentially the original
  concern. QUALIFIED would be more accurate.
- No instruction overrides an owner decision: there is no early-stopping change, no mandatory seeds and no automatic redo of
  selection.

## 5. Privacy
No host names, IPs, GPU UUIDs, usernames, absolute home paths or tokens in the lane files. Paths are written as $STATE/... and
$HOME/.local/state is used only as a shell default in COMMANDS.sh. The matches on "4090" are inside hashes.

## Verdict
REPRODUCED (all byte-identical; digests match). The substantive claims survive independent checks. Corrections needed: qualify the
11b "visible to the fits" figure, downgrade the clock-offset consequences in 08b to RISK, retain the converter-to-ts provenance link
for 08a, reclass 10a as QUALIFIED, and fix COMMANDS.sh so it defines AUD and AUDITED.
