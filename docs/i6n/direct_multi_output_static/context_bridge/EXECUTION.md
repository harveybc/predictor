# Execution Receipt

Campaign `i6dmo-e5b3dd7cd27e` was deployed from immutable runtime `ab22099f`
after 80 focused tests, independent acceptance and real population checks.
The first actual GPU workload was verified on the 5090 with epoch14/best13,
then its fit completed. That start check is not a validation measurement.
Two fresh matched R0 fits use the same 4,882-parameter model and seed6102026;
only the input recipe differs (13 slots / 144 open hours versus 13 / 24).
The six long heads are 18/24/36/48/60/72 physical hours. Per-head DEVELOPMENT
support is 3747/3453/2868/2296/1722/1142 on shared masks and FIT statistics.
The frozen release covers 97 files. Tests, publication or scientific completion
are never inferred from a timer merely being enabled.

The config-bound preparation/admission timer hands execution to the existing
campaign workers, queue and publication timers. They advance the five units,
read back governed OLAP metrics, verify two mirrors and close automatically;
only their own timers stop. No agent polls logs. See `AUTOMATION.md` for one-shot
status at campaign, unit and publication level. No TEST, strategy, R2 or DOIN.

The coordinator's first 4 GiB population-preparation requests were refused
before execution by the aggregate slice budget. Completed test fixtures in
tmpfs were the actionable source of pressure: 1,392,606,602 bytes across 1,867
readable files from this work's broad pytest run were moved to ext4, hashed
before/after and removed from tmpfs only after agreement; unreadable fixture
files were left unchanged. No scientific artifact, other active process,
slice ceiling or safety margin was altered. Preparation then ran at the same
4 GiB request. The broad suite timed out at240s; it is not reported as green.

The source and public launch receipt are on the reporting branch, while workers
execute detached, byte-attested runtime worktrees. Later documentation commits
do not change a running model's scientific identity. Status and ETA come from
the generated campaign state; this Markdown is the retained launch account.
