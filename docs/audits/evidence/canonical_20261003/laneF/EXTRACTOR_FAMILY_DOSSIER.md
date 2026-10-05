# Lane F-prep — alternative extractor families dossier (2026-10-03)

Authority: master plan v3 §6–§7, FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30
§6.1 and FS17, handoff SATOSHI_CANONICAL_SELECTION_FIRST_EXECUTION_2026_10_03 §7
(all read at commit 2edc2684). Author role: lane F technical lead. No GPU work in
this lane. No third-party code was installed or executed; repositories were read
through the GitHub contents API only, papers through arXiv.

## 0. Verdict in one table

| Candidate | Official code @ commit | Licence | Commercial use | Time per step? | Verdict |
| --- | --- | --- | --- | --- | --- |
| TimeSiam (ICML 2024) | github.com/thuml/TimeSiam @ c06f05df6197af0d780fae16b272ee4bd7d54ed9 (2024-06-11) | MIT (© 2021 THUML) | Permitted (attribution) | NO — patch tokens (len 12, stride 12) | ADMISSIBLE licence; clean-room Keras implementation, per-step, no adapter needed in our version |
| Ti-MAE (arXiv 2301.08871) | **none official**; only unofficial github.com/asmodaay/ti-mae @ f9c6c7f8ce486f7268503e627b1d0b645dec0610, self-described "unofficial", no LICENSE | none (all rights reserved by default) | NOT permitted | YES — point-level embedding | Code NOT admissible; method implemented clean-room from the paper |
| TimeMAE (WSDM 2026) | github.com/ustc-time-series/TimeMAE @ 31690745c2149f41c33afd5879ab57da8894e5cc (2026-03-04); paper cites github.com/Mingyue-Cheng/TimeMAE which now resolves to the same repo | **none** (GitHub API: license null, /license 404) | NOT permitted | NO — non-overlapping sub-series tokens (δ∈{4,8,12,16}, default 8) + mean pooling downstream | Code NOT admissible; idea only as a later variant |
| TS2Vec (AAAI 2022) | github.com/zhihanyue/ts2vec @ b0088e14a99706c05451316dc6db8d3da9351163 (2023-06-05) | MIT (© 2022 Zhihan Yue) | Permitted (attribution) | YES — timestamp-level (B,T,D) | ADMISSIBLE licence; encoder is NOT causal by default (see §4) |
| PatchTST self-sup (ref.) | github.com/yuqinie98/PatchTST @ 204c21ef… | Apache-2.0 | Permitted | NO — patches | Out of scope for lane F-prep; listed for completeness |
| MOMENT (ref.) | github.com/moment-timeseries-foundation-model/moment @ 38f7310a… | MIT | Permitted (weights: separate terms not audited here) | NO — patches | Out of scope; corpus audit required before any comparison (§6.1) |

Own repositories (predictor, feature-extractor) are MIT (© 2024 owner), so MIT
is **not a new licence** for this programme. Licence admissibility is therefore
not the blocker for TimeSiam/TS2Vec. What remains an owner decision is
**executing** their code (PyTorch, pinned old stacks: TS2Vec pins torch 1.8.1,
numpy 1.19.2; TimeSiam pins sktime 0.4.1, reformer_pytorch, unpinned torch).

## 1. TimeSiam

- Paper: J. Dong et al., ICML 2024, PMLR 235; arXiv 2402.02475 (v1 2024-02-04, v2 2024-06-07).
- Code: thuml/TimeSiam, MIT. No pretrained checkpoints are distributed in the repo.
- Objective (from `models/PatchTST.py::pretrain_timesiam` and the paper): sample a
  *past* window x_p and a *current* window x_c from the same series with distance
  d ∈ [0, sampling_range·seq_len]; mask current patches (mask_rate 0.25); a shared
  (Siamese) encoder embeds both; a learnable **lineage embedding** chosen by the
  distance bucket (`lineage_search`) is added to the past tokens; a decoder does
  self-attention on current tokens and **cross-attention to past tokens**; a
  Flatten head reconstructs the full current window (MSE). Fine-tuning feeds one
  window with each lineage token and averages (`representation_using=avg`) or
  concatenates encoder outputs.
- Shapes: input [B, L, C] (channel-independent), L=96; tokens [(B·C), L/12+1, d_model=512];
  the representation is **per patch**, not per step, and the head flattens all
  patches. Encoder attention is bidirectional (`mask_flag=False`), so token t sees
  the future inside the window. Per-instance normalisation over the window.
- Reported benchmark (paper, in-domain fine-tuning, avg over horizons 96–720):
  ETTh1 MSE 0.429 (PatchTST backbone), 0.440 (iTransformer), random init 0.473;
  Exchange 0.353 / 0.355 vs 0.367. (Extracted from arXiv HTML; re-verify against
  the PDF table before citing in a closure table.)
- Reproduction command (official, NOT executed): `bash scripts/TimeSiam/ETT_script/PatchTST_ETTh1.sh`
  (pretrain `--task_name timesiam --seq_len 96 --mask_rate 0.25 --sampling_range 6 --lineage_tokens 2 --train_epochs 50 --d_model 512 --d_ff 1024`,
  then `--task_name fine_tune --pred_len {96,192,336,720} --load_checkpoints ./outputs/pretrain_checkpoints/ETTh1/ckpt_best.pth`).
  Data from the authors' Google Drive / Tsinghua Cloud links.
- Pretraining data: ETT, Weather, Electricity, Traffic, **Exchange (daily FX of 8
  currencies vs USD, 1990–2016)**, and cross-domain TSLD-500M / TSLD-1G.
  Contamination risk: relevant ONLY if foreign weights were used. We use none —
  our clean-room model is pretrained TRAIN-only on our own series, so FS17's
  "foreign-corpus weights are not TRAIN-only" condition does not arise.
- Compute: single A100 80GB, 5 repetitions. Our univariate T=168, D=8 pilot is
  orders of magnitude smaller (fits trivially on the 5070 Ti 16 GB or 4090 24 GB).
- Adapter required for the official model: yes (patch→step). For our clean-room
  variant: no — we tokenise per step.

## 2. Ti-MAE

- Paper: Z. Li, Z. Rao, L. Pan, P. Wang, Z. Xu, "Ti-MAE: Self-Supervised Masked
  Time Series Autoencoders", arXiv 2301.08871 (v1 2023-01-21). No official code
  link in the paper; the only public code is an unofficial, unlicensed repo.
- Objective: MAE-style random masking of **point-level** embedded steps (optimal
  ratio ≈75%), encoder sees visible steps only, decoder (2-layer Transformer,
  4 heads, d=64) reconstructs all points with MSE; the decoder also emits the
  forecast directly.
- Shapes: per-step tokens; time preserved (B,T,64). Encoder bidirectional.
- Reported benchmark (Table 1, multivariate, vs representation learners):
  ETTh MSE 0.2629 / 0.3520 / 0.3977 / 0.4266 / 0.4493 / 0.5091 at H = 12/24/48/96/128/168;
  Exchange 0.0697 … 0.2123 at H = 24…196; TS2Vec on ETTh in the same table 0.5817 … 0.7621.
  Splits 6:2:2 (ETT), 7:1:2 others; batch 64; V100 32GB.
- Reproduction command: none exists (no official code). NOT reproducible as a
  reference → per master plan §6 it may NOT be called a reference; it is a
  method description we implement clean-room as a *control family*.
- Pretraining data: ETT, Weather, Exchange (FX), ILI, UCR. Same contamination
  remark as §1 (no foreign weights used).

## 3. TimeMAE

- Paper: M. Cheng et al., WSDM 2026, pp. 498–508; arXiv 2303.00320 (v4 2026-02-27).
- Code: ustc-time-series/TimeMAE, **no licence** → reading permitted, copying /
  deriving / executing not permitted without the authors' grant.
- Objective: non-overlapping sub-series (window-slicing, δ default 8), mask 60%,
  **decoupled** encoders for visible vs masked positions, two targets: masked
  codeword classification (learned tokenizer, vocab 192) and masked representation
  regression against a momentum (EMA, 0.99) target encoder.
- Shapes: tokens [B, T/δ, 64]; downstream `forward` mean-pools over tokens
  (`torch.mean(x, dim=1)`) → global vector. Fails the temporal contract without a
  declared adapter; the pooled mode never satisfies FS17.
- Benchmark: classification only (HAR, PhonemeSpectra, ArabicDigits, Uwave,
  Epilepsy). E.g. FineLast accuracy HAR 91.31±0.10, PS 19.25±0.22, AD 95.76±0.51,
  Uwave 95.88±0.28, Epilepsy 97.88±0.20 (3 runs). No forecasting benchmark.
- Reproduction command: `bash run.sh` (HAR, 3 seeds, `--mask_ratio 0.6 --vocab_size 192 --wave_length 8 --layers 8 --d_model 64`).
  Defect observed by reading: the published `run.sh` ends the python command with
  a trailing `\` before `done`, so the loop does not parse as written; the script
  also hard-codes `--device cuda:5`.
- Verdict: not first, not adoptable as code. Its MRR/EMA-target idea can be a
  later clean-room variant of the masked AE.

## 4. TS2Vec (additional past-to-current/contrastive candidate)

- Paper: Z. Yue et al., AAAI 2022, arXiv 2106.10466. Code MIT.
- Objective: hierarchical contrastive loss (temporal + instance) between two
  overlapping random crops with timestamp masking (binomial p=0.5), max-pooled
  over scales.
- Shapes: input [B, T, C] → `encode` returns [B, T, repr_dims=320] per timestamp.
  Encoder is a dilated Conv1D stack with **same (two-sided) padding** — not causal.
  Causality is obtained only through sliding inference (`causal=True,
  sliding_length=1, sliding_padding=200` in `tasks/forecasting.py`), i.e. one
  forward per step.
- Benchmark (univariate ETTh1, ridge on last-timestamp representation):
  MSE 0.039 / 0.062 / 0.134 / 0.154 / 0.163 at H = 24/48/168/336/720
  (MAE 0.152 / 0.191 / 0.282 / 0.310 / 0.327). RTX 3090; training 60.42 s for ETTm1.
- Reproduction command (NOT executed): `python -u train.py ETTh1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval`.
  The `forecast_csv` loaders append calendar covariates to the input.
- Pretraining data: whatever dataset is trained (no shipped weights). No foreign weights.
- Verdict: admissible licence and timestamp-level; it is the cleanest *reference
  reproduction* target because it has an exact command and a univariate hourly
  table. For our operational contract we would need a causal-padding variant
  (declared, a different identity from the official one).

## 5. Decision for lane F

1. **First alternative family to run on the 5070 Ti / 4090: causal masked temporal
   autoencoder (MTAE, Ti-MAE-style point-level masking), clean-room in Keras.**
   Reasons: (a) per-step tokens → satisfies "output keeps T" without an adapter;
   (b) uses exactly the same single-window support as the identity/random/AE/DAE
   controls, so the paired contract needs no extra history; (c) cheapest; (d) no
   licence or code adoption needed. It is a control family, not a "Ti-MAE
   reproduction" and must never be labelled as such.
2. **Second: past-to-current Siamese (P2C, TimeSiam-style), clean-room in Keras**,
   per-step (no patches), causal encoder, lineage embedding by distance bucket,
   cross-attention from current latent to past latent in the training decoder
   only. It needs a past window preceding the current one, so its TRAIN support
   begins `max_distance` steps later; that support difference must be declared in
   the paired comparison. At inference only the shared encoder runs on the current
   window (plus a chosen lineage/“current” token), so the operational input is
   identical to the other families.
3. TS2Vec is the preferred **external reference reproduction** (exact command,
   univariate hourly table, MIT). TimeSiam's official ETTh1 script is the second.
   Both require running third-party code → owner request below.
4. TimeMAE code: excluded (no licence). Ti-MAE code: excluded (no official code,
   unofficial repo unlicensed).

Why clean-room instead of wrapping MIT code even where allowed: the stack is
Keras/TF and the operational contract (causal, per-step, mask/delta/calendar
inputs, target never an input) differs from all four official encoders
(bidirectional attention or two-sided padding, patch tokens). A wrapper would
not satisfy FS17 anyway. No line of third-party code is copied; the
implementations follow the papers' descriptions only.

## 6. Owner requests (only if reference reproduction is wanted)

R-F1. Authorise executing the official TS2Vec code at commit b0088e14 in an
isolated environment on one worker (PyTorch install from its requirements, CPU or
one GPU), to reproduce `ETTh1 forecast_univar` and compare against the table in §4.
R-F2. Same for TimeSiam at commit c06f05df (`PatchTST_ETTh1.sh`), including
downloading the authors' dataset bundle from Google Drive / Tsinghua Cloud.
Neither request blocks the clean-room lane F work; without them, MTAE and P2C are
reported as *modular control families* and never as reproductions of the papers.
No request is needed for TimeMAE / Ti-MAE: their code is not admissible and will
not be requested.

## 7. Implementation plan in feature-extractor

Branch `satoshi/alt-extractor-families-20261003`, cut from lane D's branch
`satoshi/univariate-temporal-extractor-20261003` once its interface
(signal/observed_mask/delta_time/calendar → latent (B,T,D)) is committed.
Both families consume the same four inputs and emit latent (B,T,D):

- MTAE: causal Conv1D stem → causal Transformer blocks (causal attention mask) →
  latent (B,T,D); pretraining replaces masked steps' signal with a learned mask
  token and sets their observed flag to 0 (the encoder never sees masked values),
  decoder (causal) reconstructs signal; loss on masked steps only.
- P2C: same encoder shared by past and current windows; lineage embedding added
  to past latent; training decoder = causal self-attention on current latent +
  cross-attention to past latent → reconstruct masked current signal.

Tests required: time preserved (B,T,D); future perturbation (changing inputs at
steps > t leaves latent[:, :t+1] unchanged); target not an input (encoder input
signature has no target, and passing one is rejected); save/load round-trip
(identical latents); encoder exportable without decoder.

## 8. Observation from reading TimeSiam's sampler (not executed)

`data_provider/data_loader.py::__getitem__` draws `r_begin = random.randint(s_begin, r_limit)`,
so the distance between "past" and "current" windows can be 0 or smaller than
`seq_len`: the current window then overlaps, or equals, the unmasked past window,
and masked current points are visible through the past branch. Our P2C sampler
forbids this by construction (lag ∈ [T, max_lag], past ends at or before the
current window begins) and a test enforces it. This is a deliberate deviation;
it is one more reason the clean-room P2C is not a TimeSiam reproduction.

## 9. Implementation status

- feature-extractor branch `satoshi/alt-extractor-families-20261003`, tip `1b74f93` (module commit
  `981fdc3`), cut from lane D's branch at `2b9b82b` (before lane D's interface
  existed). Files: `app/alt_extractor_families.py`,
  `tests/test_alt_extractor_families.py`.
- Encoder (shared): causal Conv1D stem → 2 causal self-attention blocks with
  per-step FFN → Dense(D). Inputs exactly signal / observed_mask / delta_time /
  calendar; output (B,T,D). Training: fresh masks every epoch, fixed validation
  masks, early stopping, best weights restored, update count recorded.
- Tests: 21 passed on CPU under a 3 GiB memory cap (27 s): time preserved (both
  families), future perturbation on each of the four inputs at t ∈ {0, T/2, T−2}
  (both families), target refused, save/load round trip of the encoder alone with
  no decoder layers, pooled-output guard, shape/finite guards, masked values never
  reach the encoder, best-val restore reproduces the recorded best, P2C past never
  overlaps current. Mutation check: switching encoder attention to non-causal
  makes 8/8 future-perturbation cases fail.
- Integration pending lane D: lane D's red tests (`b5f63eb`) reserve
  `U.FAMILIES["masked_temporal_ae"]` and `U.FAMILIES["past_to_current_siamese"]`
  with status `SLOT_RESERVED_FOR_LANE_F_PREP` and assert they raise
  `SlotNotImplemented`. Once lane D's `app/univariate_temporal.py` lands, these
  two slots are wired to this module through lane D's `make_extractor` interface,
  and that slot test is updated together with lane D.
- First GPU run when lane F starts: MTAE, hourly window 168, D=8, one seed, on the
  5070 Ti or 4090, under the paired contract with identity/random/AE/DAE.
- Interface compatibility: lane D pushed `e59b91c` ("wip" implementation).
  Its `INPUT_NAMES` equals ours, and its `make_windows` → `TemporalBatch.as_inputs()`
  feeds both encoders unchanged (missing data included). Verified 22/22 passing
  against `e59b91c` in a detached scratch worktree; on the lane F branch the
  compatibility test is skipped until lane D's module is merged in. I did not
  edit lane D's WIP file; wiring the two reserved slots is a joint change once
  lane D marks its interface final.
