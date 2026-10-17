"""Inference-only check of the historical learned weights; never trains."""
import argparse
import json
from pathlib import Path

from tools.i6dmo import train as T, model as M
from tools.i6dmo.all_horizons import metric_rows
from tools.i6n_swarm.common import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spec',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    import tensorflow as tf
    if not tf.config.list_physical_devices('GPU'):
        raise ValueError('GPU_REQUIRED')
    spec=json.loads(Path(args.spec).read_text())
    root=Path(spec['gpu_data'])
    config=json.loads((root/'CONFIG.json').read_text())
    effective=T.effective_from_design(root/'SOURCE_DESIGN.json')
    data=T.load_data_bundle(root/'source_bundle',effective)
    bundle,metadata=M.load_dmo(args.model)
    x,_,_=T.materialize_monitor_inputs(data['validation'],config['scales'])
    prediction=T.to_price(M.predict_matrix(bundle.forecast_model,x),data['validation'],config['scales'])
    actual=metric_rows(data['validation'],prediction)
    reference=config['source_validation']
    result=dict(fit_executed=False,test_read=False,weights_sha256=metadata['weights_sha256'],
                metrics=actual,relative_error_delta=[a['relative_error']-b['relative_error']
                                                   for a,b in zip(actual,reference)],
                max_mae_delta=max(abs(a['mae']-b['mae']) for a,b in zip(actual,reference)))
    atomic_json(Path(args.out),result,mode=0o600)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
