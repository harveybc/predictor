"""Disposable consumer-boundary probes; no services, GPU, or market data."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--core', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [args.core, args.node]
    from doin_core.archive.body import build_archive, verify_envelope
    from doin_node.archive.warehouse import DisposableWarehouse, project_from_reference

    row = dict(record_id='r1', candidate_id='c1', attempt=1, won=False,
               domain_id='test', peer_id='test-peer', performance=0.1,
               parameters={}, metrics={'MAE': 0.1})
    first = build_archive(block=None, candidates=[row])
    second = build_archive(block=None, candidates=[{**row, 'performance': 0.2}])
    verified = verify_envelope(second)

    class Source:
        def load_verified(self, requested):
            return verified

    def project(digest):
        db = DisposableWarehouse(':memory:')
        try:
            db.create_experiment(domain_id='test', node_id='n', experiment_id='e')
            count = project_from_reference(db, Source(), manifest_digest=digest,
                                           experiment_id='e')
            return count, db.get_rounds('e')[0]['performance']
        finally:
            db.close()

    mismatch = project(first.manifest_digest)
    assert first.manifest_digest != second.manifest_digest
    assert mismatch == (1, 0.2)
    verified.records[0]['performance'] = 999.0
    mutated = project(second.manifest_digest)
    assert mutated == (1, 999.0)
    result = dict(scope='Disposable source-boundary counterexamples, not deployed corruption',
                  wrong_reference_accepted=list(mismatch),
                  records_mutated_after_verification_accepted=list(mutated),
                  core='a50a33e', node='6c64646')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
