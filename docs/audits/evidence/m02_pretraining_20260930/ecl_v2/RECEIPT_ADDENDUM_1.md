# DONORS_FOR_R1_R2 ECL v2 receipt addendum 1

Applies to RECEIPT.json sha256 946c252fe41166dcc853808b6b7d0657ea04e85f35df5d3851706da4c2c5f3d4. The receipt itself is not modified.
Reconstruction error is not forecasting skill.

## Next donor run, declared requirements

- **Memory cap of 8G or more.** This run's cumulative whole-cgroup peak was 6,766,669,824 B, about 90 % of the 7G cap (7,516,192,768 B). The peak was reached in the fusion and core stages and likely includes page cache from the 9.03 GB fused memmap.
- **Child-process mode for MEM-02** (`branch_isolation="process"`, ba2c634f, certified 82/82). This run used the in-process path, and RSS after each branch rose from 1,374,432 to 3,380,520 KiB across the 321 branches.

## Identities a migration must reproduce (from PRETRAIN.json)

| Identity | sha256 prefix |
|---|---|
| train input (AE train, 14,672 windows) | b430df32 |
| internal validation input (purged TRAIN tail, 3,645 windows) | 981108e1 |
| input manifest, the governed ECL train resource | eca31ec1 |
| admissible-input declaration | ca1098ed |

M02 countersigns M01's DONOR_INDEX amendment 1 only if every derived field equals these records. It refuses otherwise.

Satoshi, successor technical lead, 2026-10-01
