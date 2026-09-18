# Literature-based sample/context design: mandatory evidence before a new choice

Owner clarification: 2000 samples was an experiential example, not an asserted exact law.
The requested deliverable is a rigorous development of the choice, not another arbitrary
number or a debate about the example. P=16 in the previous handoff is NOT selected for a
new learning campaign. All candidate periods remain unselected until the requirements and
evidence below are reviewed. The pause on inherited-period factorial expansion remains.

## Primary literature and applicability

1. Zhang, Zohren and Roberts (2019), **DeepLOB**, IEEE TSP 67(11), DOI
   10.1109/TSP.2019.2907260. [Author-institute PDF](https://www.oxford-man.ox.ac.uk/wp-content/uploads/2020/03/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books.pdf).
   Section V.B, printed p.3007 (PDF p.7), reports poorer early forward-fold performance
   with little training and better results with more data. Section III.B, p.3003, describes
   over 134 million observations and chronological 6/3/3-month train/validation/test splits.
   Section IV.B, p.3005, distinguishes a 100-update input window from dataset volume.
   Use: financial precedent for measuring training sufficiency and temporal coverage.
   Limit: LOB data and composite CNN/LSTM are not our scalar sinusoid or weekly strategy;
   no transfer of a universal minimum sample size. Its trading example is gross, before fees.
2. Hoiem et al. (2021), **Learning Curves for Analysis of Deep Networks**, ICML,
   PMLR 139:4287-4296. [Paper and PDF](https://proceedings.mlr.press/v139/hoiem21a.html).
   Sections 2-4 develop curve estimation and analysis of model/design choices as training
   size changes. Use: retain training-size curves rather than one arbitrary dataset size.
   Limit: classifier experiments, not validation of our temporal error bars or market data;
   chronological sampling and dependent evaluation units require separate justification.
3. Bai, Kolter and Koltun (2018), **An Empirical Evaluation of Generic Convolutional
   and Recurrent Networks for Sequence Modeling**. [Primary paper](https://arxiv.org/abs/1803.01271).
   Section 3 describes causal/dilated temporal convolutions and residual architecture;
   experiments compare convolutional and recurrent learners. Use: evaluate actual temporal
   receptive field and training architecture. Limit: this does not establish a financial
   period or guarantee CNN superiority for our task. Our plain stack is not the paper's
   complete residual TCN merely because it has dilated convolutions.

## Required derivation, with no hidden constants

Record distinct quantities: raw observations N, valid supervised windows M, window W,
horizon h, stride s, period P in samples, physical cadence, independent series/episodes,
training updates, and evaluation units. For an uninterrupted scalar sequence with one
label at t+h and W samples ending at t:

    M = max(0, floor((N - W - h) / s) + 1)

Apply this per permitted partition, after support separation and availability rules.
It counts examples, NOT independent information. Also report (W-1)/P and h/P, periods
covered by the training time span, and the actual model support. Never infer data sufficiency
by dividing 2000 by P or counting all overlapping windows as independent cycles.

Separate three decisions and the evidence that can justify each:

* **Temporal scale:** derive business horizon/cadence from real configs and characterize
  TRAIN-only temporal structure. If there is no stable cycle, do not force a sine-period
  interpretation. A synthetic test scale must exercise a named requirement: distinguish
  phase ambiguity, estimate frequency, extrapolate or detect a nonlinear interaction.
* **Context/model:** compare short/adequate/long context relative to that requirement, with
  matched decision rows, fixed model where testing context, and explicit architecture controls.
  State what is known to the predictor. A known-frequency clean sinusoid differs fundamentally
  from unknown/drifting frequency or nonlinear dynamics. Do not claim nonlinear adequacy from
  a sine that a linear recurrence solves.
* **Training size:** estimate a learning curve from nested development training histories
  or independent synthetic realizations, with enough epochs/updates to distinguish optimization
  failure from lack of data. Choose a final size only under a predeclared error/precision target
  and cost limit; failure to meet them yields insufficient evidence, not a smaller convenient N.

Each proposed number needs: source or equation; measured input; assumption; uncertainty;
failure test; sensitivity range; cost consequence. Distinguish literature evidence, mathematical
identity, observed project measurements and design choices. Papers motivate a procedure;
they do not magically determine P for a different dataset.

## Coverage and acceptance requirements

For regression inspect coverage of phase, amplitude, slopes, turning points and target ranges;
for classification measure actual class counts AFTER labels, thresholds, windows and purging.
Positive/negative CONTROL cases mean presence/absence of the relation under test, not merely
positive/negative sine amplitudes. Phase variants and overlapping windows are not automatically
independent statistical replicates. Noise, regime changes and out-of-family cases need explicit
roles. Generator truth is diagnostic only, never a hidden feature.

Predeclare tolerable prediction error or excess risk, uncertainty target, independent evaluation
unit, multiplicity/stopping rule, model initialization variability, temporal split, and the
minimum improvement relevant to the intended decision. Assess train/validation curves and
paired baselines. Final test remains inaccessible to cost pilots and design selection.
Report if a curve is still improving, an interval too wide, or a learner undertrained.
No repeated seeds/sample-size increases until a favorable result appears.

## Assignment and next deliverable

Satoshi continues useful T1/T2 correctness tests and the requirements-first discovery, without
starting the paused factorial. Deliver a decision table combining the three decisions above,
with a bounded design to acquire any missing evidence. Musashi independently checks mathematical
assumptions and ML applicability, not just software tests. Do not ask the owner for approval
of every step; no new authorization is needed for literature review, design or code tests.

This document is literature/design work, not an executed sample-size study. No period, minimum
training size, adequacy result, or trading/RL validation is claimed here. All later scientific
measurements remain governed; the old diagnostic evidence stays intact.
