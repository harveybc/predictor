# Data phase for hourly EURUSD direction: what the literature says (2026-10-21)

Context: selections v5/v6 found no direction signal (only weekday/session clocks pass a univariate, lag-0 screen;
every model is worse than the constant prior). Owner order: get the DATA right first. This note condenses a review
of the most-cited work; each item lists the finding and what it implies here. Citations are given so that every
claim can be audited.

## 1. Where hourly FX predictability is documented

- Evans & Lyons 2002 (JPE 110(1):170–180): order flow explains ~60 % of daily DM/$ changes, mostly CONTEMPORANEOUS;
  out-of-sample forecasting from lagged flow is weak (Evans & Lyons 2005, AER P&P) except with interbank flow data
  (Rime, Sarno & Sojli 2010, JIE 80(1)). We have no interbank flow; weekly CFTC is not a substitute.
- Andersen, Bollerslev, Diebold & Vega 2003 (AER 93(1):38–62): announcement SURPRISES move FX within minutes; the
  VOLATILITY response lasts hours. Without point-in-time consensus the surprise cannot be signed, but the release
  TIMES are known in advance (causal).
- Andersen & Bollerslev 1998 (JF 53(1)); Ito & Hashimoto 2006 (JJIE 20(4)): strong time-of-day / weekday seasonality
  in volatility and activity, much weaker in returns — consistent with only calendar clocks passing our screen.
- Ranaldo 2009 (JBF 33(12)); Breedon & Ranaldo 2013 (JMCB 45(5)): currencies tend to depreciate in their own
  domestic hours. Krohn, Mueller & Whelan 2024 (JF 79(1):541–578): the USD rises into the Tokyo and London fixes and
  reverses after them — the best-documented DIRECTIONAL hourly effect for EURUSD; a few basis points.
- Mueller, Tahbaz-Salehi & Vedolin 2017 (JF 72(3)): predictable USD excess returns on FOMC days, concentrated in
  high-uncertainty regimes — usable with scheduled dates only.
- Andersen, Bollerslev, Diebold & Vega 2007 (JIE 73(2)): cross-asset price discovery is mostly contemporaneous;
  intraday lead-lag decays within minutes → daily cross-asset series lagged 2 days are unlikely to predict hourly
  direction (consistent with v6).
- Neely & Weller 2003 (JIMF 22(2)); Hsu, Taylor & Wang 2016 (JIE 102); Neely, Weller & Ulrich 2009 (JFQA 44(2)):
  intraday technical rules in majors earn nothing after costs; profits are episodic and regime-dependent.
- Rossi 2013 (JEL 51(4)): the random walk is hard to beat; predictability exists only in specific windows.
  A constant prior beating unconditional models is the EXPECTED baseline.

## 2. Mixed-frequency fusion without adding noise

- Durbin & Koopman 2012 (OUP); Harvey 1989 (CUP): the Kalman FILTERED state uses data <= t (causal); the smoothed
  state leaks the future. State space handles missing values and irregular releases.
- Stock & Watson 2002 (JASA 97); Giannone, Reichlin & Small 2008 (JME 55(4)); Aruoba, Diebold & Scotti 2009 (JBES 27(4)):
  a few filtered common factors summarise many noisy predictors and update at each release ("ragged edge").
- Ghysels, Santa-Clara & Valkanov 2006 (J. Econometrics 131); Andreou, Ghysels & Kourtellos 2013 (JBES 31(2)):
  MIDAS — parsimonious lag weights across frequencies; daily series should enter as release-time INNOVATIONS with
  lags in DAYS, not as raw levels at hourly lags 0–2 (step functions carrying no news inside a day).

## 3. Selection that survives noise and non-stationarity

- Guyon & Elisseeff 2003 (JMLR 3): univariate filters miss joint and state-conditional relevance.
- Meinshausen & Bühlmann 2010 (JRSS-B 72(4)); Shah & Samworth 2013 (JRSS-B 75(1)): stability selection with an
  error bound; subsample in contiguous blocks for time series.
- Barber & Candès 2015 (AoS 43(5)); Candès et al. 2018 (JRSS-B 80(3)); Chi, Fan, Ing & Lv 2025 (JASA): knockoffs for
  conditional relevance, time-series variant; low power with weak, collinear signals.
- López de Prado 2018 (AFML, Wiley): permutation importance under purged, embargoed CV (our labels overlap 72 bars).
  Strobl et al. 2008; Hooker, Mentch & Zhou 2021: permute GROUPS of correlated features.
- Harvey, Liu & Zhu 2016 (RFS 29(1)); White 2000 (Econometrica 68(5)); Romano & Wolf 2005: multiplicity is
  features x lags x regimes x models — raise hurdles; reality-check / SPA tests on final strategies.

## 4. Lags and causal discovery

- Runge et al. 2019 (Sci. Adv. 5(11)); Runge 2020 (PCMCI+, UAI): tau_max from the decay of the lagged dependence, not
  a default; with tau_max = 2 at hourly resolution daily/weekly mechanisms are invisible → resample to the
  mechanism's timescale or engineer the lag into the feature. Treat finance PCMCI output as hypotheses.
- Granger 1969; Diks & Panchenko 2006; Schreiber 2000; Barnett, Barrett & Seth 2009: transfer entropy = Granger for
  Gaussian data; needs large samples, noisy at low SNR.

## 5. Decompositions without look-ahead

- Two-sided transforms (DWT/MODWT boundaries, EMD, HP filter, STL) leak the future: Quilty & Adamowski 2018
  (J. Hydrol. 563); Du, Zhao & Lei 2017 (J. Hydrol. 552); Furlaneto et al. 2017 (ESWA 82): reported gains vanish when
  the decomposition is causal. Hamilton 2018 (REStat 100(5)) on HP. Causal alternatives: à trous with a one-sided
  Haar filter (Renaud, Starck & Murtagh 2005), expanding-window SSA, Kalman local level / local trend FILTER.

## 6. LLMs and time-series foundation models

- Lopez-Lira & Tang 2023 (headline sentiment → next-day stock returns); Glasserman & Lin 2023 and Sarkar & Vafa 2024:
  look-ahead / memorisation bias inflates such results and prompting does not remove it; He, Lv, Manela & Wu 2025:
  chronologically consistent LLMs.
- Chronos (Ansari et al. 2024), TimesFM (Das et al. 2024), Moirai (Woo et al. 2024): univariate value forecasters;
  Rahimikia, Ni & Wang 2025: poor on returns off the shelf. No credible evidence of beating naive for hourly FX
  direction; any 2020–2025 evaluation sits inside their pretraining window (contaminated). Only a FORWARD (live,
  paper) evaluation would be clean.

## Actions adopted for this project (in order)

1. DST-correct event and session clock: hours to/since the Tokyo and London fixes, FOMC / ECB days, NFP / CPI release
   hours from official SCHEDULES (no consensus needed).
2. Daily / weekly data re-indexed to release time as innovations with lags in days (1, 2, 5) — MIDAS-style; CFTC at
   the Friday release.
3. Kalman-filtered dynamic factors (USD, EUR, risk-on/off) estimated on TRAIN only, filtered-only out of sample.
4. Conditional screening on a small pre-registered strata grid (volatility tercile x session x event window,
   <= ~20 strata) with FDR over the whole grid; grouped permutation importance under purged/embargoed CV; block
   stability selection.
5. Split the label: barrier hit vs timeout (volatility, where seasonality and events are strong) and TP vs SL given a
   hit (direction).
6. Causal layer-1 decompositions only (one-sided à trous, expanding SSA, Kalman local trend), each with the
   truncation bit-identity test.
7. Models last; any strategy gated by a reality-check / SPA test net of costs. LLM / TSFM work only with a forward
   (post-cutoff) evaluation.
