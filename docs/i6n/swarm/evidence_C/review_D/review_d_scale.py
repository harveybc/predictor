"""R5: ETL cost vs outbox size (disposable instances)."""
import json, sys, tempfile, time
from pathlib import Path
ROOT = Path(sys.argv[1]); N = [int(x) for x in sys.argv[2].split(",")]
sys.path[:0] = [str(ROOT), str(ROOT / "olap/store/src")]
import tests.test_i6n_archive as T
res = {}
for n in N:
    with tempfile.TemporaryDirectory() as d:
        pub, archive, etl, wh, _ = T._rig(Path(d))
        chain = T.LocalChain()
        for i in range(n):
            c = T.candidate(i); c.pop("won", None)
            pub.record_candidate(c, [T.receipt(c, fold=f) for f in ("2023Q1", "2023Q2", "2023Q3", "2023Q4")])
        pub.pump(archive, chain.submit); chain.mint(empty_after=6)
        t0 = time.perf_counter(); pub.outbox.items(); t_items = time.perf_counter() - t0
        t0 = time.perf_counter(); s = etl.run(chain.blocks); t_etl = time.perf_counter() - t0
        t0 = time.perf_counter(); s2 = etl.run(chain.blocks); t_etl2 = time.perf_counter() - t0
        body = len(pub.outbox.bytes_of(pub.outbox.items()[0]["item_id"])[0])
        res[n] = {"body_bytes": body, "items_call_s": round(t_items, 4), "etl_first_s": round(t_etl, 2),
                  "etl_replay_s": round(t_etl2, 2), "errors": len(s["errors"]), "green": pub.status()["green"]}
print(json.dumps(res, indent=1))
