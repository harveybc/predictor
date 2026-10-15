# Pre-fit Deployment Incidents

2026-10-15. These are engineering failures, not scientific measurements.
All original private roots and receipts are preserved. No scientific fit was
started or repeated by either failure; no closed short result was modified.

1. The first frozen-source preparation reached replication. The inherited
   deployment workdir was a file-only runtime, but provisioning requires a Git
   source. The SSH Git operation refused it. The successor private deployment
   explicitly names existing source repositories; no repository was reset,
   and the source/runtime byte attestation remains mandatory.
2. Campaign `i6dmo-6672a4eb53ae` reached worker admission. Both workers then
   raised `TypeError` while constructing the child environment: role hardening
   and explicit worker defaults repeated `PYTHONDONTWRITEBYTECODE` as keyword
   arguments. This occurs before `executor.start`. Only this campaign and its
   launcher timers/services were stopped. Its retained queue claims do not
   prove that training ran. A successor must use a new corrected frozen commit,
   new population-generation pin and new campaign identity, not a hot patch.

The correction must retain environment precedence (process < extra < role <
mandatory worker device/defaults), accepting overlapping keys without weakening
the mandatory CUDA isolation. An executable regression must drive the actual
worker through executor startup, followed by independent review of that delta.
Keep original failures visible; do not relabel them as scientific negatives.
