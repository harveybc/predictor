# Post-reboot recovery inspection

Musashi, 2026-09-26. Read-only operational inspection after the owner restarted the
coordinator. This is NOT a fresh acceptance of the new scientific results or DR04.
No training, GPU inference, service restart, broker call or lease reclamation was
performed during this inspection. Times below are America/Bogota (UTC-05:00).

## Finding: the last verification did not reach a demonstrated completion

The previous boot journal records systemd-oomd killing the Firefox scope at
**16:27:10** and the VSCode scope at **16:27:49**. User-session memory pressure
was respectively 69.86% and 70.31%, above 50% for over 20 seconds. Their current
memory usages at selection were 7.4G and 4.9G. These are victims, NOT a complete
attribution of the pressure to those applications.

Two DR01 follow-on verification scopes ended at **16:27:49**, the same second
as VSCode. The journal reports 1,227.957 and 769.094 CPU seconds, with 2.9G and
2.8G memory peaks. The second was the combined pytest invocation over admission,
D2 replay, E1 closure, MOD-E0 closure and SOTA reproduction tests. No terminal
test result for these two invocations was established by this inspection.
Their disappearance is not evidence of a successful suite.

The admission ledger independently records the second scope queued for headroom
and pressure, then admitted at **16:20:21** with PSI some/avg10 **24.5**, just
below its threshold of 25.0. It had observed 52.18 thirty seconds earlier.
This does not prove the scope caused the later kill. It DOES mean deployment of
atomic admission cannot be reported as proof that the desktop-pressure problem
is solved. Admission-time checks, post-admission protection, bypasses and load
growth are separate properties.

The coordinator rebooted at **17:40:56**. At inspection it had approximately
24 GiB available RAM and effectively empty swap. Two lease JSON files remained
on disk; they were not reclaimed or edited. Current installed lease records
carry PID/starttime but no boot identity. RR02 requires reboot-aware validation
before these records can be used as current reservations.

## What survived

The following tips were checked against the remote. These are published
deliveries, not assertions that every claim or test in them is accepted:

| Repository / branch | Tip | Recovery disposition |
|---|---|---|
| predictor / satoshi/dr01-atomic-admission-20260926 | ee30935a | Atomic admission implementation retained |
| predictor / satoshi/dr01-followon-deploy-20260926 | f66ee25d | Deployment and four CLI guards retained; combined verification interrupted or unproven |
| predictor / satoshi/dr02-dispatch-index-20260926 | b790cd12 | Eleven-lane index retained; several rows predate later deliveries |
| predictor / satoshi/dr05-confirmatory-candidate-20260926 | 86a19936 | Candidate retained, NOT external approval |
| predictor / satoshi/dr05-dr06-corrections-20260926 | c2d4388b | Lag correction and prefix output materialization delivered; not work to repeat blindly |
| predictor / satoshi/core-pretrain-resolution-20260926 | 4168ebdc | Resolution report and matched contrast retained; scientific review still due |
| predictor / satoshi/q2-context-deep-arms-20260926 | 7b0248f8 | Twelve historical cells; six full-window cells absent; successor v2 sealed but not executed |
| predictor / satoshi/huber-design-recovery-20260926 | 76650ce8 | Recovered design published; its temporary checkout did not survive reboot |
| agent-multi / satoshi/dr04-m4-verifiability-20260926 | 4b009c35 | Repairs and author's POST delivered; external M4 review still due |
| lts / satoshi/mt5-corrections-20260926 | d631f4a | Corrections published; not evidence of deployment or resolved unknown orders |

Eight inspected persistent predictor worktrees (DR01, follow-on, DR02, DR05/06,
Q2-deep, resolution, candidate and the previous auditor tree) were clean. The
primary checkout still contains the owner's untracked presentation/evidence
files. Nothing there was cleaned, staged or reverted. Git lists several missing
temporary worktrees; recover committed files from Git, not from assumed /tmp
paths. Do not prune missing worktrees before their recovery inventory is made.

Satoshi's retained session's last textual update said only deployment to the
workers and runner coverage remained. A resumed Claude process is present after
the reboot, but process presence alone does not establish active work or a final
handoff. The follow-on report itself explicitly leaves four seal-pinned runners,
two agent-multi launch paths and the M5PHET worker admission outside its new CLI
guard coverage. Therefore **the whole assignment is not established complete**.

## Live resource and service snapshot

No crispdm compute scope was active on the three hosts inspected. Worker GPUs
had no significant memory allocation: the preferred external RTX 5090 was
38 C and 0% utilization; the other worker's RTX 4090 was 32 C. Host RAM available
was only about **6.6 GiB** on the external-GPU worker and **6.4 GiB** on the
other worker; the latter retained about 5 GiB used swap. VRAM is not host RAM.
These measurements expire and must be repeated before dispatch.

Governance services were active after reboot. The M5PHET chat and Alpaca model
runner also started at 17:41:04. The LTS checkout remains on
`satoshi/mt5-unknown-outcome-20260926`, not the new corrections branch.
The reboot invalidates the previous explanation that Alpaca was a resident
pre-checkout-change process. This inspection establishes service start time and
working directory, not broker mutations or numerical behavior. Preserve the
read-only/no-new-order restriction and verify runtime identity before adoption.

## Scientific state, without inventing a new result

- A/B and the nine-cell ECL R0/R1/R2 contrast already finished. No fit is queued
  by this recovery. Their prior scores are not newly measured here.
- Q2's full-context question remains unanswered: six of eighteen cells never
  fitted. The twelve retained fits are historical, ungoverned and budget-censored.
  Successor v2 is `27ed712d88e0b1a84a3a7d48d5a8dcd899274389f8cc7341df196705dbb0300d`.
  Never combine its seal with v1 results by relabelling them.
- DR05/06 has already materialized a prefix output. The index row telling an
  agent to begin that work is stale; inspect the actual remaining delivery gaps.
- M4 repairs and a confirmatory candidate are now available for review. Neither
  is an external approval, and neither opens the reserve.
- M5PHET remains a parallel product programme, not a replacement for these
  experiments. Its chat service is available, not evidence that five useful
  models or the financial experiment have been validated.

## Reproducible inspection sources

Local, read-only commands used included:

```bash
journalctl -b -1 -u systemd-oomd --since '2026-09-26 16:20:00' --until '2026-09-26 16:35:00' --no-pager
journalctl --user -b -1 --since '2026-09-26 16:20:00' --until '2026-09-26 16:29:00' --grep 'crispdm-|Memory|Consumed|Killed|oom' --no-pager
git worktree list --porcelain
git ls-remote --heads origin 'satoshi/dr*20260926'
systemctl --user list-units --all 'lts-*' 'm5phet*' 'crispdm-*.scope' --no-pager
```

Admission evidence was read from `~/.local/state/crispdm/admission/ledger.jsonl`
and the installed `~/.local/libexec/crispdm/crispdm_admission.py`. Private raw
journals and the agent transcript are not copied into the public repository.
Continuation: [RR01-RR07](../../handoffs/SATOSHI_POST_REBOOT_ORDERS_2026_09_26.md).
