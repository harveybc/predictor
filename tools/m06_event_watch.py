import json,sys,time,re
p=sys.argv[1]; out=open(sys.argv[2],'a',buffering=1)
seen_inc=None; jobs=None; warned=set(); camp={}; seen_coord=set()
D=re.compile(r"(^m04-|^d-(?!unit|profile)|pilot|^m02-ecl-v2-donors)",re.I)
def ts(): return time.strftime('%H:%M:%SZ',time.gmtime())
while True:
    try: d=json.load(open(p))
    except Exception: time.sleep(10); continue
    inc={i['lease_id'] for i in d.get('terminal_incidents_24h',[])}
    if seen_inc is None: seen_inc=set(inc)
    for i in d.get('terminal_incidents_24h',[]):
        if i['lease_id'] not in seen_inc: out.write(f"{ts()} STOP {i['host_alias']} {i['name']} {i['exit_cause']} incident={i['lease_id']} peak={i.get('tree_peak_bytes')} at={i['at']}\n")
    seen_inc|=inc
    for j in d['jobs']:
        if j.get('host_alias')=='coordinator' and re.match(r'^d\d*-runner-',j['id']) and (j.get('cgroup_peak_bytes') or 0) > 64*2**20 and ('runner64'+j['id'] not in seen_coord):
            out.write(f"{ts()} ALERT exempt runner {j['id']} peak {j.get('cgroup_peak_bytes')} B > 64 MiB (exemption breached)\n"); seen_coord.add('runner64'+j['id'])
        if j.get('host_alias')=='coordinator' and j['state'] in ('running','queued') and not j['id'].startswith('m06-') and not re.match(r'^d\d*-runner-',j['id']) and not re.match(r'^laneB-.*-bounded-read',j['id']) and (j['id'] not in seen_coord):
            out.write(f"{ts()} ALERT coordinator batch job {j['id']} {j['state']}/{j.get('phase')} (rule: zero batch jobs on the coordinator)\n"); seen_coord.add(j['id'])
    for u in d.get('unparsed_processes',[]):
        if u.get('host_alias')=='coordinator' and str(u.get('classification','')).startswith('UNLEASED_SCOPE'):
            sc=u.get('scope','')
            if not re.match(r'^crispdm-(m06-|d\d*-runner-|laneB-.*-bounded-read)',sc) and sc not in seen_coord:
                out.write(f"{ts()} ALERT coordinator unleased batch scope {sc} memory={u.get('memory_current')} (zero-batch rule)\n"); seen_coord.add(sc)
    cur={j['id']:(j['state'],j.get('phase'),j.get('host_alias'),j.get('cgroup_peak_bytes'),(j.get('progress') or {}).get('completed')) for j in d['jobs']}
    if jobs is not None:
        for k,v in cur.items():
            if D.search(k) and (k not in jobs or jobs[k][:2]!=v[:2]): out.write(f"{ts()} PHASE {k} {v[2]} -> {v[0]}/{v[1]} peak={v[3]} progress={v[4]}\n")
        for k in jobs:
            if D.search(k) and k not in cur: out.write(f"{ts()} ENDED {k} (last {jobs[k][0]}/{jobs[k][1]}, peak={jobs[k][3]}, progress={jobs[k][4]})\n")
    jobs=cur
    for c in d.get('campaigns',[]):
        sig=json.dumps([c.get('status_counts'),(c.get('incumbent') or {}).get('config_id')],sort_keys=True)
        if c['campaign'] in camp and camp[c['campaign']]!=sig: out.write(f"{ts()} CAMPAIGN {c['campaign']} counts={c.get('status_counts')} done={c.get('cells_done')}/{c.get('cells_planned')} incumbent={c.get('incumbent')} eta={c['eta'].get('earliest')}..{c['eta'].get('latest')}\n")
        camp[c['campaign']]=sig
    # GPU idle alarms come from m06-gpu-sampler (15 s sampling, 120 s threshold); none here
    for role,q in d.get('quotas_measured',{}).items():
        for key,lim in (('disk_home_free_bytes',20<<30),('mem_available_bytes',3<<30),('swap_free_bytes',2<<30)):
            v=q.get(key)
            if key=='swap_free_bytes' and not q.get('swap_total_bytes'): continue
            if v is not None and v<lim and (role,key) not in warned: out.write(f"{ts()} LOW {role} {key}={v>>20}MiB\n"); warned.add((role,key))
            if v is not None and v>=lim*1.5: warned.discard((role,key))
        if role=='coordinator':
            pa=q.get('psi_some_avg60'); ma=q.get('mem_available_bytes')
            if pa is not None and pa>10 and (role,'psi') not in warned: out.write(f"{ts()} ALERT coordinator PSI some avg60={pa} > 10\n"); warned.add((role,'psi'))
            if pa is not None and pa<5: warned.discard((role,'psi'))
            if ma is not None and ma<(12*10**9) and (role,'mem12') not in warned: out.write(f"{ts()} ALERT coordinator MemAvailable={ma/1e9:.2f} GB < 12 GB\n"); warned.add((role,'mem12'))
            if ma is not None and ma>(14*10**9): warned.discard((role,'mem12'))
        if role=='worker_b':
            pb=q.get('psi_some_avg10')
            if pb is not None and pb>37.5 and (role,'psi_respond') not in warned: out.write(f"{ts()} ALERT worker_b PSI some avg10={pb} > 37.5 (respond threshold)\n"); warned.add((role,'psi_respond'))
            if pb is not None and pb<10: warned.discard((role,'psi_respond'))
        v=q.get('unreclaimable_slab_bytes')
        if v is not None and v>(4<<30) and (role,'slab') not in warned: out.write(f"{ts()} LOW {role} unreclaimable_slab_bytes={v>>20}MiB above 4096MiB\n"); warned.add((role,'slab'))
        if v is not None and v<(3<<30): warned.discard((role,'slab'))
    time.sleep(10)
