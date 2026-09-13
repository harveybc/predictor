"""Read-only: for every C172 shard member, the longest complete TRAIN stretch the lab would fit on."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tools"))
import df_lab_evaluation as LAB  # noqa: E402

S = Path.home() / ".local/state/crispdm-data-foundation/d2_shards_c172_v1"
short, total, by_regime = [], 0, collections.Counter()
for link in sorted(S.glob("shard_*/*")):
    if not link.is_dir():
        continue
    u = LAB.load_unit(link)
    sl = LAB.train_fit_slice(u["observed"], u["rec"]["partitions"]["train"])
    total += 1
    n = sl.stop - sl.start
    if n < 50:
        short.append((u["rec"]["unit_id"], n))
        by_regime[json.dumps(LAB.regime_of(u["rec"]), sort_keys=True)] += 1
print(json.dumps({"units": total, "short_units": len(short), "short": short,
                  "by_regime": by_regime}, indent=1))
