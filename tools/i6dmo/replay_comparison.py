"""Compare a reproduction with historical evidence without fitting or tuning."""
import argparse
import json
from pathlib import Path

from tools.i6n_swarm.common import atomic_json


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    def load(root, name):
        return json.loads((Path(root)/name).read_text())
    result = compare_training(load(args.reference, 'TRAINING.json'), load(args.candidate, 'TRAINING.json'))
    result['identity_checks'] = compare_receipts(load(args.reference, 'RECEIPT.json'),
                                               load(args.candidate, 'RECEIPT.json'))
    result['test_read'] = False
    atomic_json(Path(args.out), result, mode=0o600)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
