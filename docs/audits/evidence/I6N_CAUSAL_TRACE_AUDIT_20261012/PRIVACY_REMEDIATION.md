# Privacy remediation of the I6N causal trace audit branches (2026-10-12)

Owner-ordered. Rewritten: `retsu/i6n-causal-trace-audit-20261012` and
`retsu/i6n-causal-trace-audit-addendum-20261012`, range `f5fb48aa..tip` only. The base
`f5fb48aa` and all `codex/*` branches were not touched.

## What was removed

| Category | Where | Replacement |
|---|---|---|
| Real island host label | `tests/test_i6n_causal_trace_audit.py` redaction fixture; `REVIEW.md` (first version) | `host-a-0000` (the test still maps it to `ISLAND_0` and asserts it is gone) |
| Absolute data-root path prefix | `TRACE_GRAPH.json` (4 edges), `FINDINGS.csv` (E9), `REVIEW.md` | `<data-root>` |
| Sample IP in the redaction fixture | test file, `REVIEW.md` | `203.0.113.7` (RFC 5737 documentation range) |

No other real host labels, private IPs, GPU UUIDs or home paths appeared in the
four commits or their messages. The synthetic fixtures `/var/tmp/x` and `/home/someone`
stay.

## Commit mapping (messages, authors and dates preserved)

| Old | New |
|---|---|
| 5a9858c1 | 0d5baa94 |
| 45a741e9 | ad63d949 |
| 19e9da7c | ec7b7d5d |
| d0642119 | 6599ef15 |

Digests changed only because of the redaction. `MANIFEST.sha256` was recomputed in every
rewritten commit, and only three entries moved: `TRACE_GRAPH.json`, `FINDINGS.csv` and
`tests/test_i6n_causal_trace_audit.py`. No other document cites those digests. Audit
verdicts, counts and every other file are byte-identical. The `REVIEW.md` privacy section
describes the pre-redaction state, and both of its items are closed by this rewrite.
`tests/test_i6n_causal_trace_audit.py` passes 16/16 on the rewritten tip.

The pre-redaction history is kept in a private local bundle (mode 600, outside the
repository), sha256 `cf630d30c361104997fb24e26005b3520ea4b59b84987d37a0dd2d7468496608`.

## Residual risks

1. **Inherited host label (owner decision pending).** The same host label is still
   reachable from the remote through base commit `b1dd1f66`, which lives on
   `codex/i6n-doin-sequence-20261012`, `codex/i6n-orphan-recovery-20261012` and
   `codex/retsu-i6n-causal-audit-20261012`. That makes it reachable from these rewritten
   branches through their base as well. Those branches are not ours to rewrite.
2. **Dangling SHAs at the provider.** The hosting provider may still serve the old
   commits (table above) by direct SHA URL until it garbage-collects them. Git cannot verify
   this, and only a provider-side purge request removes them.
