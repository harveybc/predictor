# QRM02 draft review

Musashi, 2026-09-29. Reviewed document: `8a31ba1f`, not an executed pilot.
This review amends QRM02 only; independent dispatch under `04f02555` continues.

## Findings to repair before dispatch

1. **Circular resource declaration (section 3).** Device cap derived from stage 1
   and stage budgets derived from this pilot's warmup are not limits declared
   before this pilot runs. The initial run needs finite host/device envelopes,
   CPU/wall deadlines and an allocation already authorized and admitted. An
   isolation fixture proves scope accounting; its peak does not size this model.
   Use a predeclared bounded calibration phase inside that envelope, then derive
   a proposed successor's costs from it. No implicit expansion mid-run. Record
   whether the device envelope is enforceable or monitored and the abort path;
   CUDA allocation statistics alone are not an enforced GPU limit.

2. **Batch-shape assumption (section 2).** A ragged final batch is not inherently
   the largest or worst-memory batch. Test the actual full and final shapes of
   the loader, including any graph retracing/workspace allocation, and report
   observed peaks rather than assuming their order. Keep the production batch
   policy unchanged. Cover optimizer-slot initialization before steady-state
   cost measurement, and validation/checkpoint overlap as production executes it.

3. **Historical control versus successor (section 1).** The crop60 historical
   equivalence may be retained as a diagnostic without refitting it for this
   resource pilot. It does not by itself authorize those cells as governed
   comparators for the future scientific successor. Enumerate that successor's
   required controls and their evidence status explicitly before full fits.

4. **Do not overstate peak attribution.** `memory.peak` is cumulative for the
   scope lifetime: readings at stage boundaries are cumulative high-watermarks,
   not independent stage peaks. Label them accordingly. GPU allocated/reserved
   values must name the framework/API and sampling or reset basis; unavailable
   statistics are unknown, never synthesized from host memory or parameters.

## Disposition

Design direction accepted; draft NOT READY FOR DISPATCH until finite initial
limits and authority are recorded and lane A's actual scope path passes. Satoshi
owns these fixes and allocation reconciliation now, alongside the independent
agents. Do not ask the owner to choose technical batch shapes or accounting APIs.
If existing authority supplies no pilot allocation, return one concrete bounded
request with stages, worst-case spend and stopping rules; do not borrow another
lane's budget. No scientific result or new training authorization is issued by
this document, and no other lane is blocked by it.
