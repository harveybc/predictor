# E3-M Capacity Recovery

The original12-arm batch closed at2026-10-22 15:47:27 UTC, after2.0969 hours:
10 authenticated results,2 failed,0 pending. All10 terminal manifests and the
consumer export were copied to the external second disk and hash-verified.
The original CLOSE, failed attempts and successful results remain unchanged.

The failed consolidated arms were cons_shortlist and cons_shortlist_strict_enc.
Although their wrapper return code was124, this was NOT the12-hour wall limit:
retained admission incidents prove PRESSURE_STOP_SUSTAINED_ABOVE_RESPOND.
Observed whole-tree peaks were4030111744 and4036136960 bytes under4GiB caps.
Those stopped-run peaks are lower bounds, not final measured memory demand.

## Separate Bounded Repair

Only the two failed jobs are selected for the new root
`/var/tmp/expert-guided-musashi-recovery-20261022`.
The original code/archive/data/lists are linked read-only-by-policy, byte-pinned
and reverified; science flags, seeds, GPU, splits and business remain unchanged.
The resource request is ceil(1.25*max(retained_tree_peak)/MiB)=4812MiB. This
increases rather than lowers the failed request. The incident/lease evidence
and hashes are retained in the recovery package. No pressure monitor, reserve,
kernel setting, swap or another owner's process was changed or stopped.

The new operations worker supports an explicit validated job subset; it cannot
dispatch the ten successful arms. Two PRE subset tests failed before support;
all65 focused regressions passed afterwards. The old installed worker/config
was not modified. Recovery config identity:
7a3fef865d059799dd7688bb64e4021a9f8aba4e0a53b3758ade4d1367cb3fb0.

Worker/timer stem: musashi-e3m-recovery-20261022. The existing independent mirror
collector now targets a separate recovery directory on the same external disk;
the original completed mirror remains untouched.

## Observed Status At Handoff

WAIT_ADMISSION_HOST_HEADROOM. The existing atomic admission launcher has queued
the first4812MiB request. No native scientific child or GPU compute PID was yet
observed. The wrapper's RUNNING field here means its admission subprocess is
alive, NOT that model training has begun. Report that distinction explicitly.
Last host available memory was about7.97GB; request plus existing3GiB reserve
needs about8.27GB before other commitments, so headroom was insufficient.

The admission launcher rechecks automatically, with the existing timer providing
restart/reboot recovery. No human result approval is needed. Once admitted,
an initial35-50min estimate for both arms is an extrapolation from successful
12-14min consolidated arms and larger input counts, not a measured completion
time for this new resource envelope. There is no reliable overall ETA while
resource admission is waiting, and no claim that increasing the cap guarantees
these larger arms finish. Further failures retain their true terminal causes.

Subsequent start check: the first recovery arm was admitted and its native
scientific Python child was observed. The initial headroom wait ended without
changing the request/reserve or stopping another process. No recovery arm has
completed yet; the35-50min estimate remains provisional from this start.

This remains the assigned ZigZag-guided development package, not native SAC,
not TEST and not scientific acceptance of the shared feature release.
