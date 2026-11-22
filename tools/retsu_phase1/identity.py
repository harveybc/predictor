import json
import numpy as np
import torch
from tools.specialist_exposure403 import ROOT, PARENT, assemble, environment, file_sha
from app.structural_refit import load_bundle

B = ROOT / "model/61/bundle"
assert file_sha(B/"model.pt") == "5b14e9f64f2fd296df2d0df6c6fdeffe0e16cb2e6e9715233caf400d44447c25"
assert file_sha(B/"RECIPE.json") == "96a5fb87d843d49f80274aff90e68de9de955101d99823e8f2b5b227f01d3021"
joined, orig = assemble(PARENT, B, ROOT/"history397-61-r2")
net, _ = load_bundle(B, "cpu")
risk = json.loads((PARENT/"RISK.json").read_text())
for m, lo, hi, expected in [(joined, "2016", "2024", (38782,28991)),
                            (orig, "2020", "2024", (19197,14360)),
                            (orig, "2024", "2025", (4912,3627))]:
    env = environment(m, risk, 6, [lo+"-01-01", hi+"-01-01"], .99)
    e = env.engine
    e.reset()
    bars = np.array([b for b in range(e.first,e.last) if env.valid[b] and e.entry_gate(b)[0]])
    rows, seg = env.rows[bars], e.seg_of[bars]
    slots = m.inputs.slot_ts_ns[rows]
    strict = ((slots >= e.op[seg,None]) & (slots < e.ff[seg,None])).all(axis=1)
    ids = rows[strict]
    assert (len(rows), len(ids)) == expected, (lo, hi, len(rows), len(ids), expected)
    with net.loss.actor_network_params.to_module(net.actor), torch.inference_mode():
        enc = net.actor_net.encoder.eval()
        w = torch.from_numpy(m.inputs.windows[ids[:4]].copy())
        z = enc({"features":w})
        assert w.shape == (4,13,61) and z.shape == (4,208) and torch.isfinite(z).all()
    print(lo, hi, len(rows), len(ids), tuple(z.shape), flush=True)
print("IDENTITY_OK", flush=True)
