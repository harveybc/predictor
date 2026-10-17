"""Compare a reproduction with historical evidence without fitting or tuning."""
import argparse
import json
from pathlib import Path

from tools.i6n_swarm.common import atomic_json, sha256_json


def compare_training(reference, candidate):
    fields = ('epoch', 'fit_loss', 'inner_loss', 'monitored', 'stale',
              'best_epoch', 'updates', 'weights_sha256')
    pairs = list(zip(reference['history'], candidate['history']))
    first = next((a['epoch'] for a, b in pairs
                  if any(a[k] != b[k] for k in fields)), None)
    checks = {k: reference[k] == candidate[k] for k in
              ('settings', 'initial_weights_sha256', 'epochs_completed', 'best_epoch',
               'stop_reason', 'restored_weights_sha256')}
    exact = all(checks.values()) and first is None and len(reference['history']) == len(candidate['history'])
    return dict(checks=checks, first_different_epoch=first,
                compared_epochs=len(pairs), exact_training_reproduction=exact,
                historical_best_epoch=reference['best_epoch'],
                candidate_best_epoch=candidate['best_epoch'])


def compare_receipts(reference, candidate):
    fields = ('bundle_manifest_sha256', 'code_files_sha256', 'design_sha256',
              'effective_config_sha256', 'feature_manifest_sha256', 'scales_sha256')
    return {k: reference['identity'][k] == candidate['identity'][k] for k in fields}


def publish(result, spec, root):
    from tools.i6dmo.outbox import Outbox, HttpWarehouse
    checks = dict(result['checks'], **result['identity_checks'])
    checks['exact_training_reproduction'] = result['exact_training_reproduction']
    metrics = [dict(metric='Reproduction_'+k, value=int(v), unit='bool', horizon=None,
                    split='train') for k, v in checks.items()]
    report = dict(experiment_key='i6dmo:'+spec['batch_name']+':trajectory',
                  experiment_set_key='i6dmo:'+spec['batch_name'], actor='predictor',
                  lake='LOCAL_RETAINED', lineage='UNVERIFIED', config_sha256=spec['prepared_sha256'],
                  code_commit=spec['code_commit'], project='predictor', phase='I6DMO_REPRODUCTION',
                  metrics=metrics, tags=dict(evaluation='DEVELOPMENT_TUNING', test_read='false',
                                            evidence_sha256=sha256_json(result)))
    box = Outbox(root/'outbox.sqlite3')
    try:
        box.enqueue(report, source_unit='reference-device-trajectory')
        pub = box.pump(HttpWarehouse(spec['warehouse']['url'], spec['warehouse']['token_env']))
        atomic_json(root/'PUBLICATION.json', pub, mode=0o600)
        atomic_json(root/'READBACK.json', [dict(report_sha256=r['report_sha256'], state=r['state'],
                                               readback_sha256=r['readback_sha256']) for r in box.items()], mode=0o600)
        if pub['pending'] or pub['mismatch'] or pub['delivered'] != 1:
            raise RuntimeError('REPRODUCTION_PUBLICATION_INCOMPLETE')
    finally:
        box.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--publish-spec')
    args = p.parse_args()
    def load(root, name):
        return json.loads((Path(root)/name).read_text())
    result = compare_training(load(args.reference, 'TRAINING.json'), load(args.candidate, 'TRAINING.json'))
    result['identity_checks'] = compare_receipts(load(args.reference, 'RECEIPT.json'),
                                               load(args.candidate, 'RECEIPT.json'))
    result['test_read'] = False
    atomic_json(Path(args.out), result, mode=0o600)
    if args.publish_spec:
        publish(result, json.loads(Path(args.publish_spec).read_text()), Path(args.out).parent)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
