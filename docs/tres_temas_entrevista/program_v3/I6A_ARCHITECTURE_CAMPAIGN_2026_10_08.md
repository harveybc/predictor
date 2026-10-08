# I6-A: matched temporal architecture controls

Status: COMPLETE for the selected-input EURUSD `Y_s_1h` business adaptation;
the general E0 architecture acceptance gate remains open.

Origin-alignment correction: the first ARCH-C code used Keras `Conv1D` with
stride 2, which sampled positions 0,2,...,22 and dropped the origin at 23.
The old ARCH-C cell bytes and logs are preserved under an archive manifest and
are excluded from closure. The corrected branch applies a causal stride-1
convolution followed by indices 1,3,...,23 at each halving. A test now proves
that both the earliest input and the origin can influence the last core step,
and that the origin cannot influence the first fused step. ARCH-A/B and ARCH_0
are unchanged. Only ARCH-C cells are repeated; its architecture digest changed.

Scope: this is the selected-input BUSINESS adaptation of the ARCH controls,
not the general E0 mechanism study. In particular ARCH-A retains the existing
FS4 three-convolution 24-to-12-to-6 branch, whereas the original E0 design
sheet names a one- or two-layer local detector. H2/H3, contextual sensitivity,
the literature models and R0/R1/R2 remain separate acceptance work. The common
core's causal residual dilations give its last position access to the full
24-hour window in the learned-branch arms; ARCH_0 deliberately sees only six
sampled positions, so its deficit cannot isolate "learning" from information
loss. Do not call this bank the full ARCH acceptance gate by itself.

The frozen EURUSD `Y_s_1h` RAW winner supplies four selected inputs. The
business protocol fits a new model before each 2024 validation week from the
preceding four years, with a purged chronological inner-validation tail. Every
arm uses the same 24-hour causal windows, target, row population, seed 0,
optimizer, 30-epoch maximum, batch size 256, patience 5, fusion, temporal core
and scalar head. The zero-return naive is scored on exactly the same rows.
Only the branch stack differs:

| Arm | Branch before the common six-step fusion grid |
| --- | --- |
| ARCH_0 | Fixed causal 4:1 sampling, no learned feature branch |
| ARCH_A | Independent grouped causal Conv1D branches, 24 to 12 to 6 |
| ARCH_B | Independent residual dilated causal Conv1D branches, 24 to 12 to 6 |
| ARCH_C | Independent causal Conv1D branches plus per-feature GRU, 24 to 12 to 6 |

The fixed sampler is a negative architectural control, not a claim that it
retains all information. ARCH-C's GRU returns a sequence. No arm flattens time
before fusion. Parameter counts and fit time are reported rather than assumed
equal. Complete weeks are assigned to one device, so all four arms of each
week share a host. Two of every three weeks are assigned to the preferred GPU;
the remaining third to a second GPU. Input file hashes are equal across the
two workers. Each worker is bounded by a 5 GiB cgroup and writes an atomic,
digest-checked result per cell, a status with measured ETA and a failure log.
There is no automatic retry of a failed cell.

The TRAIN-2023 week-0 implementation diagnostic completed on 120 identical
rows. The feature set was selected using later validation, so these numbers
are **not** prospective model quality evidence:

| Arm | MAE | Same-row naive MAE | Parameters |
| --- | ---: | ---: | ---: |
| ARCH_0 | 0.007030384 | 0.000921990 | 3,297 |
| ARCH_A | 0.000924921 | 0.000921990 | 3,953 |
| ARCH_B | 0.000946530 | 0.000921990 | 4,785 |
| ARCH_C, corrected | 0.001092393 | 0.000921990 | 4,225 |

The first 2024 validation week also used identical rows (97) and naive MAE
0.000612503. It is one of 52 weeks and cannot order the arms: ARCH_0
0.004883628; ARCH_A 0.000617189; ARCH_B 0.000614759; corrected ARCH-C
0.000693208. The original ARCH-C value 0.000664281 is withdrawn because it
omitted the origin. None of the four beats the naive in that week. No
strategy simulation or TEST score follows from it.

`tools/i6a_campaign.py` deterministically partitions complete weeks and
resumes from verified cell receipts. `tools/i6a_close.py` recomputes outer and
inner digests, verifies all 208 cells and refuses any aggregate if a cell or
paired population is missing. Its final table compares each arm with ARCH-A
and the same-row naive over the full weekly validation population. The
selected feature set was chosen on this validation year, so the comparison is
development evidence; TEST remains sealed for a single later confirmation.

## Complete 2024 validation result

Closure `d25425393ae537288b09cdeb21382cc7ea253292cc161bff0866be039ccc0b0b`
accepted 208/208 current-revision cells and all 52 matched weeks. The 27
superseded ARCH-C cells remain separately archived and do not enter this
denominator. Mean weekly naive MAE is 0.000492891 for every arm.

| Arm | Mean weekly MAE | Mean weekly skill vs naive | Weeks beating naive | Parameters | Fit seconds, 52 weeks |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARCH_0 | 0.005474226 | -10.4003 | 0 | 3,297 | 793 |
| ARCH_A | 0.000511972 | -0.0413 | 7 | 3,953 | 606 |
| ARCH_B | 0.000500321 | -0.0151 | 12 | 4,785 | 646 |
| ARCH_C, origin-aligned | 0.000677526 | -0.3960 | 0 | 4,225 | 1,635 |

ARCH-B's mean paired difference from ARCH-A is -0.000011651 MAE (-2.28 %
relative to ARCH-A's mean); B is better in 34/52 weeks. This is descriptive,
not a significance or generalization claim. ARCH-B remains 0.000007430 MAE
above the same-row naive mean. Therefore no arm from this bank is eligible for
strategy testing or trading. ARCH_0 also removes three quarters of the input
times by design, so its poor result does not by itself measure the value of
learned branches.

The deployed warehouse read back 208 idempotent reports and 1,664 metric rows.
Their lineage is explicitly `UNVERIFIED` because these cells consumed retained
local parquets, not a data-gov delivery. Physical cell receipts, both source
archives, the closure and the freeze remain retained outside Git. The source
architecture file hash agreed across both workers:
`ac2165aef9926c1859b53ad1ca2d30b687e5ce5b62bf9aa512d79951c98ab876`.

Warehouse correction: a replay exposed a DuckDB-specific false `rowcount` on
`INSERT ... ON CONFLICT DO NOTHING`. The same 208 report rows acquired a second
copy of each of their 1,664 child metrics. With the service stopped, a
byte-matched physical backup was retained outside Git. All 1,664 duplicate
pairs had identical content; only their second I6-A metric row was removed.
The DuckDB backend now checks report identity under its write lock before the
insert, and the publisher requires both report and metric counts on readback.
The restarted service returned `already_stored` for a retained report without
adding a row; readback was again 208 reports and 1,664 metrics. No model was
rerun for this repair.
