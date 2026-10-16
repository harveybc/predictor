# Context Bridge Operation

The private launch spec contains exactly schema, work_root, source_deployment
and input_dir. Paths are absolute; source files remain outside work_root.
Schema: `i6dmo.context_bridge_launch.v1`. Never put credentials in this file.
The existing warehouse EnvironmentFile is consumed only by publication units.

After focused acceptance, real support verification and independent review,
the operator runs this one bounded command from the reviewed checkout:

```bash
python -B -m tools.i6dmo.context_bridge_campaign install --spec <private-launch-spec>
```

The config-bound timer handles preparation/admission and stops at technical
failure or verified campaign closure. The existing campaign timers own the
five units: two fits, their two reports, then paired comparison. Publication,
readback, physical mirrors and CLOSE are mandatory parts of that machinery.
No agent must poll logs, ask to advance a stage or run the next command.
Resource refusal is retried at the SAME cap; no margin is trimmed. Completed
fits are adopted on restart, never repeated to deliver or copy results.

Bounded reads at any time:

```bash
python -B -m tools.i6dmo.context_bridge_campaign status --root <private-work-root>
python -B -m tools.i6dmo status --root <private-work-root>/campaign --level unit
python -B -m tools.i6dmo status --root <private-work-root>/campaign --level publication
```

Status counts five declared units, separates training/reporting/delivery, and
labels unknown ETA instead of extrapolating a different recipe's runtime.
TEST cannot be enumerated. One negative complete result still publishes and
closes; a failed/incomplete arm never selects a winner. No next experiment is
implicitly launched by this approval.

Historical controls are retained unchanged. Both new fits are necessary
because this contrast has new shared support/scalers/code identities; prior
model shapes alone do not justify adopting an old fit as the matched control.
That reason is recorded before fitting, not after seeing a result.

SQLite, WAL, locks, private spec and campaign root stay on ext4. Population
arrays and authenticated large artifacts use EXPERIMENT_STORAGE_ROOT; both
physical mirrors require digest verification. Git contains only code, small
configs/manifests, summaries and plots, not weights, raw data or databases.
