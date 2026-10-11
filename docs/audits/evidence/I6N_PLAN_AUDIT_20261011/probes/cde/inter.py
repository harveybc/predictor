P = "P(d4341e72):"
C = "@269c2d13"
cols = ["group","parameters","why_dependent","searched_jointly","stage_split","counterexample_if_separate","required_contrast"]
R = []
def r(*a):
    assert len(a) == 7, a[0]; R.append(a)
r("IA-01 context x capacity x reduction","L_hours; core width (filters/d_model); depth (blocks, reducer stages); temporal_reduction_mode",
  "n=1+L/delta_x sets how many positions the reducer/attention must summarise; reduction feasibility depends on n (n=5 allows at most 2 factor-2 stages, n=433 allows 8); optimal width grows with sequence content",
  "partial","N1 fixes L=2h,n=5 and screens capacity; N2 searches L jointly but seeded by N1 survivors (<=3); reduction gene meaningless at n=5 under the >=2-positions rule",
  "N1 can prune a large d_model that only pays off at L=6h (n up to 433); with 24 N2 evaluations seeded around N1 winners the large-capacity x long-L corner is likely never visited (" + P + "105-108,113)",
  "Before N2 ranking: on both representatives a sealed 2x2x2 slice {L=2h,6h} x {smallest, largest capacity} x {no reduction, progressive} on the same 4 folds, reported with seed-noise band")
r("IA-02 resolution x receptive field x position","delta_x; branch/entry/reducer kernel (fixed 3); dilation (fixed 1); positional_encoding_formula",
  "kernel is in samples, so physical receptive field = 2*delta_x hours (2h at delta_x=1, 144h at delta_x=72); sinusoid uses hours from window start so origin code depends on L; at delta_x>=7h the highest band (period 2*pi h) aliases",
  "N","kernel/dilation fixed across N1-N4; PE formula never a gene nor specified; delta_x searched in N2",
  "h=72: delta_x=1 vs delta_x=36 differ in branch receptive field by 36x, so a delta_x ranking also ranks receptive field; t's PE code is 144h at L=144 and 432h at L=432 (ClockView positions from window start, tools/i6e_masked_model.py:43), so a recipe transferred across L presents a new origin code",
  "Origin-anchored PE (lag hours, t=0) declared; contrast kernel fixed-in-samples vs fixed-in-hours (kernel=ceil(K_h/delta_x)) at delta_x in {1,8,36} for h=72")
r("IA-03 output resolution x loss x capacity","delta_y (trajectory); loss (Huber fixed); huber_delta=1 MAD; task weights; core capacity",
  "trajectory adds auxiliary targets with different scales/noise; Huber delta in units of the t+h MAD clips the h/2 output at a different quantile; extra supervision changes the capacity optimum",
  "N","capacity fixed in N2 under scalar output; trajectory tested in N3 on <=2 alternatives",
  "If trajectory supervision regularises a large model, N2 (scalar) will have chosen a smaller core and N3 cannot recover it; with unspecified per-output weights the h/2 term can dominate gradients",
  "N3 trajectory arm evaluated on at least one larger-capacity config than the N2 winner; per-output TRAIN scales and weights declared; Huber delta per output from TRAIN residual quantile (FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md:85-88); MAE arm")
r("IA-04 reduction x final adapter x readout","temporal_reduction_mode; stride factor; stage count; adapter type (none/Conv1D/LSTM); LSTM units; head_readout",
  "adapter consumes the reduced sequence; LSTM over 2 positions ~ Dense; head reads last position so t-coverage of the reduced grid decides whether t reaches the head",
  "partial","both in N2 but no compatibility table, padding or parity rule",
  "n=55 (h=72,L=432,delta_x=8): causal stride-2 stage1 55->28 covers t, stage2 28->14 does NOT (probe n55_causal_k2_s2 False); valid padding loses t at stage1 (27); existing _residual_temporal_stage raises on odd n (components.py:294-295). LSTM return_sequences=True + last == return_sequences=False (probe max|diff|=0)",
  "Origin-anchored reduction rule (left-pad masked to multiple of factor, valid kernel=factor) + perturbation test on every grid n and stage count; declare readout gene {last, masked mean, attention pool} or drop return_sequences claim")
r("IA-05 multi-output x per-h scales x gradients x sharing","N4-M family heads; per_h_loss_scale; task_loss_weights; label masks; shared window/core",
  "eligible rows per head shrink with h (~96-h hourly origins per week), so masked mean losses give short-family heads and h=18 more gradient than h=72; scale choice sets relative weighting",
  "N","N4-M separate contrast after single-output N2/N3; weights 'declarar' without values",
  "Long family: h=18 has ~78 eligible origins/week vs ~24 for h=72 (BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md:115-133, derived); unweighted masked mean gives h=18 ~3x the gradient mass, so shared core may underfit h=72 and be judged worse than independent model",
  "Declare loss = sum_h w_h * masked_mean_h / s_h with w_h and s_h fixed from TRAIN; report per-head gradient norm share and per-head MAE vs independent models at the SAME window")
r("IA-06 input transform (N3) x window/capacity (N2)","input_transform RAW vs origin-relative; L; delta_x; capacity",
  "RAW-level inputs scaled by 4-year mean/sd carry slowly drifting levels; origin-relative inputs remove level so longer L may become useful or capacity needs change",
  "N","N2 fixes window/capacity under RAW; N3 swaps transform on <=2 alternatives without re-search",
  "Under RAW the best L at h=72 may be 2h because longer windows add level drift; under origin-relative inputs L=6h could win; N3 then reports 'transform helps a little' on the wrong window",
  "N3 must include each transform at >=2 windows (N2 winner L and the longest admissible) or make transform a gene in N2; declare which is concluded")
r("IA-07 output supervision (N3) x window/capacity (N2)","output_mode scalar/trajectory; L; capacity",
  "auxiliary horizon targets change effective sample information and optimal model size",
  "N","sequential N2 then N3","see IA-03","Trajectory arm on N2 winner and on one larger core; report interaction term")
r("IA-08 optimisation x capacity","learning_rate 1e-3; weight_decay 1e-4; AdamW; batch 256; post-LN without warmup; max_seconds 3600",
  "a fixed lr is not equally adequate for d_model 8..64, blocks 0..2 and n 2..433; post-LN Transformers are sensitive to lr without warmup; max_seconds truncates large models",
  "N","fit settings fixed for all stages (inherited, not in proposal)",
  "A d_model=64,2-block model at n=433 that stops by max_seconds or plateaus at lr=1e-3 is ranked worse than d_model=16: the plan would read 'capacity does not help' when it is a fit failure (FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md:8,83-84 requires LR/decay tuning)",
  "lr in {3e-4,1e-3,3e-3} on smallest and largest N1/N2 configs (TRAIN-inner only), or lr as gene; classify each fit by stop_class and final TRAIN loss; budget-stopped fits typed RESOURCE not scored as inferior")
r("IA-09 early-stopping monitor x capacity","monitor=0.5*(train+inner); patience 5; min_delta 0; max_epochs 50",
  "the TRAIN term decreases monotonically for high-capacity models, delaying patience stops and choosing later (more overfit) epochs",
  "N","fixed","High-capacity configs run to max_epochs/max_seconds with a selected epoch dominated by TRAIN loss; low-capacity configs stop earlier; the comparison mixes capacity and stopping",
  "Report selected_epoch, stop_class and inner-only loss at selection for each config; sensitivity with monitor=validation_loss on 2 configs")
r("IA-10 k integrality x h x recipe transfer","delta_x=h/k; k set {1,2,3,4,8,9}; L={1,2,4,6}h; N4 transfer of <=2 recipes to 12 h",
  "the admissible k set depends on divisibility of h; relative recipes (m,k) and absolute recipes (L hours, delta_x hours) produce different models on other h",
  "N","search only on h=4,72; transfer rule unspecified",
  "h=4 admits only k in {1,2,4} (delta_x 4,2,1); h=5 only {5,1}; h=3 only {3,1}; h=2 only {2,1}; the N1 anchor delta_x=h/2 is non-integer for h=3,5; a k=9 recipe from h=72 (delta_x=8) exists only for h in {18,36,72}; an absolute recipe L=432 is outside the declared grid for every h<72 (probe grid)",
  "Define recipe = (rule on L, rule on delta_x) with explicit fallback (nearest admissible k, ties -> finer) per h, list the 12 resulting (L,delta_x,n) before N4, and a per-family budget (>=1 local check per non-representative h)")
r("IA-11 absolute context x h-tied grid","L={1,2,4,6}h; master 24h minimum physical window",
  "market information (daily cycle 24h, weekly 120/168h) is absolute in hours, not proportional to h",
  "N","absolute contexts excluded by construction","24h reachable only for h in {4,6,12,24}; 168h for no h; h=2,3 can never reach the master default minimum 24h (MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md:315); contradicts 'No inferir que 2*h es una ley' (audit request :250)",
  "Add absolute L candidates {24,72,168} (when support allows) to the N2 gene domain for every representative and report both parametrisations")
r("IA-12 seed x folds x ranking","seed 6102026 single; 4 screening weeks; fitness ranking of 24 configs",
  "ranking differences must exceed training-seed variance and fold sampling noise",
  "N","one seed, no calibration","h=72 has ~24 eligible hourly origins per screening week with 72h-overlapping targets (~1 independent target/week); 4 folds give ~4 effective observations; two configs can swap rank by seed alone",
  "Seed-noise calibration before N2: 3 seeds x 3 configs (small, N1 best, large) x 4 folds; tolerance >= 2 SD of paired seed differences; documented as calibration, not repetition of identical units")
r("IA-13 family x inactive genes x identity","core_entry_family; blocks; d_model; heads; conv filters; lstm_units; adapter filters; reducer filters_i",
  "genes are conditional on family/stage count; equivalent genotypes must map to one effective model",
  "N","requirement only; not implemented","Existing identity = digest of full arm dict (tools/i6m_cell.py:58-63) and DEAP individual is flat list (optimizer_plugins/default_optimizer.py:185-201): two genotypes differing only in d_model under Conv1D-only would be two fits",
  "Canonicalizer (genotype -> effective config with inactive genes removed) + tests: each active gene perturbation changes config digest and model; inactive gene perturbation leaves digest/cache key identical")
r("IA-14 ARIMA resolution x convergence failure x selection","ARIMA delta; order (1,1,0); failure typing; 'no annual MAE' rule",
  "resolution chosen on TRAIN among candidates that may fail; failures remove arms from comparison",
  "N","N4 only","46 FAILED of 128 ARIMA at pause (CURRENT_EXECUTION.md:123): if failures concentrate at some delta/h, selection 'in TRAIN' silently picks among survivors and the h may end with no ARIMA control while neural rows are complete",
  "Ex-ante rule: failure in any TRAIN fold disqualifies a resolution; per-h coverage table; h without complete ARIMA reported NOT_COMPARABLE; order selection rule (or fixed order) declared")
r("IA-15 masks/gaps x delta_x x kernels/attention","input_gap_mask; delta_x phase; conv kernel; attention mask",
  "fraction of masked (weekend) slots in a window varies with delta_x and origin phase; conv kernels mix zero placeholders",
  "N","inherited behaviour, not stated in I6-N","delta_x=24 aligned to a weekend phase loses 2 of n slots, delta_x=36 a variable count; models with different delta_x see different effective information independent of resolution per se",
  "Report masked-slot fraction distribution per (h,L,delta_x) in N0; test that gating yields placeholder invariance for all families incl. LSTM adapter")
r("IA-16 batch/updates x h support x fit budget","batch 256; max_epochs 50; eligible fit rows per h after market masks",
  "updates per epoch scale with eligible rows, which shrink with h; same epoch budget = fewer updates for long h",
  "N","fixed","h=72 has roughly a quarter of the per-week eligible origins of h=4 (derived from 96-h week), so 50 epochs give ~4x fewer updates; capacity conclusions at h=72 vs h=4 are confounded with optimisation budget",
  "Report observed_updates per fit; fix budget in updates or verify convergence classification per h")
r("IA-17 representatives x families x N4-M shared window","representatives h=4 (short), h=72 (long); family-level shared window/core",
  "family model must serve 6 horizons; window chosen for 4/72 may not suit 2/12 or 18/60",
  "partial","N2 per representative; N4-M one recipe per family",
  "h=12 or h=18 boundary horizons are not represented; a short-family window tuned at h=4 may be too short for h=12",
  "Per-family check on the family extremes (2,12 and 18,72) before freezing family windows")
import csv, sys
with open(sys.argv[1], "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols); w.writerows(R)
print(len(R))
