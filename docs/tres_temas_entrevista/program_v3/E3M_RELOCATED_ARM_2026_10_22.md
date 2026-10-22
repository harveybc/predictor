# Remaining E3-M Arm: Alternate Worker

The owner authorized moving the unfinished arm to an alternate worker with
more host RAM. Original and first recovery batches remain closed and untouched:
eleven unique completed arms, with only `cons_shortlist` incomplete.

## Execution Contract

Only `cons_shortlist` is admitted. The original archive, extracted science code,
data, feature lists and queue script were checked against 121 original pinned
files. All five runtime package versions match. Scientific CLI flags, three
seeds, 173-column list, architecture, split, scaling and training budget are
unchanged. The assigned experiment still includes its declared agent-state
observations; this is not the future no-feedback SAC baseline.

The alternate worker uses an RTX 4070 Laptop GPU and a 12 GiB host-memory cap.
It had approximately 23 GB available before admission. Another candidate's
headroom fell below 9 GB during preparation, so it was not used. These are
resource-envelope and hardware changes, not a bitwise reproduction claim.
The new configuration is
`fb8a4065ba5d997747b711221e080d1fcb1f8cda64ca1656f6bd453a0d7ba97e`.
The native scientific CUDA child was observed after admission at 22:39 UTC.
No complete result exists yet. No measured completion ETA is available.

GPU exclusivity normally refuses any compute PID. This desktop GPU also lists
the existing desktop-integration process at 6 MiB. The new private configuration
names only that already-observed process by PID, start ticks and command digest,
with a 32 MiB ceiling. PID reuse, a different command, larger memory use or any
additional compute process refuses a new launch. No existing process was stopped.
Atomic memory admission, pressure monitor, host reserve and swap policy remain
enabled. No global resource setting was changed.

## Unattended Operation And Custody

The new worker timer resumes only this arm after interruption. It verifies
inputs and complete results using the existing wrapper; successful negative
metrics are still completed execution, not rejection. It self-disables at
terminal closure. Original workers and their accepted receipts are unchanged.

An independent five-minute collector copies artifacts to the writable external
storage volume and pushes that mirror to a second machine. Both copies verify
sealed file hashes and exact configuration/closure identity. Its first run
verified both destinations with zero completed results, correctly not a scientific
success. It self-disables only after terminal closure and replica read-back.
This is physical custody, not warehouse delivery or feature-release approval.

## Checks

- PRE: explicit alternate GPU selection was unsupported; two new relocation
  tests failed before their helper existed; desktop-process guard tests failed
  before the exact exception was implemented.
- POST: 72 focused operations/selection tests passed.
- Real checks: matching archive and 121 source pins; matching package versions;
  native CUDA PID, live worker heartbeat, external-volume mirror and remote
  replica read-back. No science code was edited.
- Primary worker health: no failed system services, low load, GPU temperatures
  29/38 C; recurring firmware ACPI errors were observed. No reboot was needed
  for this relocation, and this observation is not a firmware diagnosis.

Next action: consume the automatically sealed remaining result when available;
complete the source-to-consumer review before the independent SAC comparison.
