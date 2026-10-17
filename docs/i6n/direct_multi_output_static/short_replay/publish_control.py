"""Publish the inference-only diagnostic through the existing governed outbox."""
import argparse
import json
from pathlib import Path

from tools.i6dmo.outbox import Outbox, HttpWarehouse
from tools.i6n_swarm.common import atomic_json, sha256_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spec',required=True)
    p.add_argument('--root',required=True)
    args=p.parse_args()
    root=Path(args.root)
    spec=json.loads(Path(args.spec).read_text())
    evidence=json.loads((root/'INFERENCE.json').read_text())
    assert evidence['fit_executed'] is False and evidence['test_read'] is False
    metrics=[]
    for r in evidence['metrics']:
        for name,key,unit in [('MAE','mae','USD_per_EUR'),('Naive_MAE','naive_mae','USD_per_EUR'),
                              ('Relative_Error_Vs_Naive','relative_error','ratio'),('N_Paired','n','count')]:
            metrics.append(dict(metric=name,value=r[key],unit=unit,horizon=r['horizon'],split='validation'))
    metrics.append(dict(metric='Max_MAE_Delta_Vs_Historical',value=evidence['max_mae_delta'],
                        unit='USD_per_EUR',horizon=None,split='validation'))
    report=dict(experiment_key='i6dmo:short-replay-20261016:historical-control-inference',
                experiment_set_key='i6dmo:short-replay-20261016',actor='predictor',lake='LOCAL_RETAINED',
                lineage='UNVERIFIED',config_sha256=sha256_json(dict(weights=evidence['weights_sha256'],
                    population=spec['prepared_sha256'])),code_commit=spec['code_commit'],project='predictor',
                phase='I6DMO_SAVED_CONTROL_INFERENCE',metrics=metrics,
                tags=dict(fit_executed='false',test_read='false',evaluation='DEVELOPMENT_TUNING',
                          weights_sha256=evidence['weights_sha256'],evidence_sha256=sha256_json(evidence)))
    box=Outbox(root/'outbox.sqlite3')
    try:
        box.enqueue(report,source_unit='historical-control-inference')
        pub=box.pump(HttpWarehouse(spec['warehouse']['url'],spec['warehouse']['token_env']))
        atomic_json(root/'PUBLICATION.json',pub,mode=0o600)
        atomic_json(root/'READBACK.json',[dict(report_sha256=x['report_sha256'],state=x['state'],
                                             readback_sha256=x['readback_sha256']) for x in box.items()],mode=0o600)
        if pub['pending'] or pub['mismatch'] or pub['delivered']!=1:
            raise RuntimeError('CONTROL_PUBLICATION_INCOMPLETE')
        print(json.dumps(pub,sort_keys=True))
    finally:
        box.close()


if __name__=='__main__':
    main()
