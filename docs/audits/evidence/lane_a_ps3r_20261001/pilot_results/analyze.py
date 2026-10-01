import json, glob, collections, statistics as st
O="/home/harveybc/.local/state/scratch/m01/ps3r_pilot"
t=json.load(open(O+"/CONTRAST.json"))
rows=t["rows"]
recs=[json.load(open(f)) for f in glob.glob(O+"/records/*.json")]
cpu=sum((r.get("cost") or {}).get("seconds_total") or 0 for r in recs)
fit=collections.defaultdict(float); trace=0.0; stops=collections.Counter()
for r in recs:
    fit[r["arm"]]+=r["fit"].get("fit_seconds") or 0
    trace+=r["fit"].get("tracing_seconds_estimate") or 0
    stops[(r["arm"], r["fit"].get("stop_reason"))]+=1
print("records",len(recs),"cpu_seconds_total",round(cpu,1),"fit_seconds",{k:round(v,1) for k,v in fit.items()},"tracing_est",round(trace,1))
print("stops",dict(stops))
# strict-minimum per cell (input,target,horizon,fold,seed): which arm lowest
arms=("loss_raw","loss_random","loss_ae","loss_contrastive")
win=collections.Counter(); beats_naive=collections.Counter()
by_th=collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    vals={a:r[a] for a in arms}
    w=min(vals,key=vals.get); win[w]+=1
    for a in arms:
        beats_naive[a]+= vals[a] < r["naive"]
        by_th[(r["target"],r["horizon"])][a].append(vals[a]-r["naive"])
    by_th[(r["target"],r["horizon"])]["contrast"].append(r["contrast_ae_minus_contrastive"])
print("cells",len(rows),"strict-min wins",dict(win))
print("cells beating naive",dict(beats_naive))
for k in sorted(by_th):
    d=by_th[k]
    print(k, {a: f"{st.mean(d[a]):+.3e}" for a in arms}, "contrast_mean", f"{st.mean(d["contrast"]):+.3e}", "ae<cl", sum(1 for c in d["contrast"] if c<0), "of", len(d["contrast"]))
delta_ae=[r["delta_probe_ae"] for r in rows]; delta_cl=[r["delta_probe_contrastive"] for r in rows]
print("delta_probe>0  ae",sum(d>0 for d in delta_ae),"cl",sum(d>0 for d in delta_cl),"of",len(rows))
print("cards", len(t.get("cards",[])), collections.Counter(c["admissibility"]["verdict"] for c in t.get("cards",[])), collections.Counter(c["temporal_contract_state"] for c in t.get("cards",[])))
