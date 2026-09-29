"""Exercise the deployed provider on a disposable cube, never production."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sqlalchemy import text
from predictor_duckdb_store.provider import PredictorDuckdbStore

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT/'docs/contracts/tsl_literature_metrics.v1.json').read_text())


def digest(body):
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def fixture(dataset, horizon, seed):
    """Explicit NON_GOVERNING fabricated values test transport, not model skill."""
    d = CONTRACT['datasets'][dataset]
    tags = {k: 'fixture' for k in CONTRACT['required_context_tags']}
    tags.update(metric_contract=CONTRACT['schema'], dataset=dataset,
        resource=d['resource'], dataset_sha256=d['sha256'], model='NON_MODEL_FIXTURE',
        comparison_class='TRANSPORT_TEST_NOT_SCIENCE', horizon_steps=str(horizon),
        horizon_seconds=str(horizon*d['step_seconds']), seed=str(seed),
        target_channels=str(d['channels']), windows='2', elements=str(2*horizon*d['channels']),
        metric_scale='z_train', metric_reduction=CONTRACT['reduction'], metric_dtype='float32')
    specs = {**CONTRACT['primary_metrics'], **CONTRACT['paired_baselines']}
    metrics = [{'metric':name,'value':(i+1)/8,'split':'test','horizon':horizon,
        'unit':spec['unit'],'std_dev':None,'min_value':None,'max_value':None}
        for i,(name,spec) in enumerate(specs.items())]
    body = dict(schema='governed_terminal.v1',campaign_sha256='c'*64,
        campaign_key='tsl-warehouse-disposable-fixture',unit_id=f'{dataset}-{horizon}-{seed}',
        generation=1,actor='fixture',project='predictor',classification='NON_GOVERNING',
        status='COMPLETED',reason=None,started_at='2026-09-28T00:00:00Z',
        finished_at='2026-09-28T00:00:01Z',terminal_lake='olap_cube',config_sha256=digest(CONTRACT),
        code_identity={'kind':'git_commit','value':'d'*40},costs={'wall_seconds':1.0},
        tags=tags,synthetic_spec_sha256=None,deliveries=['0'*32],metrics=metrics,
        verified_datasets=[dict(delivery_id='0'*32,lake_id=CONTRACT['lake'],resource_id=d['resource'],
            role='benchmark',sha256=d['sha256'],bytes=1,source_sha256=None,range_from=None,
            range_to=None,delivery_kind='AS_IS',time_column='date',availability_contract_sha256='e'*64,
            state='VERIFIED_TRANSFER')],artifacts=[])
    body['terminal_sha256']=digest(body)
    return body


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='tsl-metrics-disposable-')
        self.store=PredictorDuckdbStore()
        self.store.set_params(duckdb_path=str(Path(self.temp.name)/'cube.duckdb'),schema='main',
            memory_limit='256MB',threads=1,min_free_bytes=1)
        self.store.engine()

    def tearDown(self):
        self.store.engine().dispose()
        self.temp.cleanup()

    def test_all_datasets_horizons_and_seeds_roundtrip(self):
        count=0
        for dataset in CONTRACT['datasets']:
            for horizon in CONTRACT['horizons_steps']:
                for seed in [1,2]:
                    body=fixture(dataset,horizon,seed)
                    self.assertTrue(self.store.write_terminal(body)['stored'])
                    self.store.write_terminal(body)
                    with self.store.engine().connect() as con:
                        tags=con.execute(text('SELECT tags_json FROM gov_terminal WHERE terminal_sha256=:d'),{'d':body['terminal_sha256']}).scalar()
                        self.assertEqual(json.loads(tags),body['tags'])
                        rows=con.execute(text('SELECT metric,value,unit,split,horizon FROM gov_terminal_metric WHERE terminal_sha256=:d ORDER BY metric'),{'d':body['terminal_sha256']}).fetchall()
                        self.assertEqual([tuple(r) for r in rows],sorted((m['metric'],m['value'],m['unit'],m['split'],m['horizon']) for m in body['metrics']))
                        resource=con.execute(text('SELECT resource_id FROM gov_terminal_dataset WHERE terminal_sha256=:d'),{'d':body['terminal_sha256']}).scalar()
                        self.assertEqual(resource,CONTRACT['datasets'][dataset]['resource'])
                    count+=1
        self.store.engine().dispose()
        with self.store.engine().connect() as con:
            self.assertEqual(con.execute(text('SELECT count(*) FROM gov_terminal')).scalar(),count)
            self.assertEqual(con.execute(text('SELECT count(*) FROM gov_terminal_metric')).scalar(),count*4)

    def test_nonfinite_metrics_are_refused(self):
        for value in [float('nan'),float('inf'),-float('inf')]:
            body=fixture('weather',96,1)
            body['metrics'][0]['value']=value
            del body['terminal_sha256']
            body['terminal_sha256']=digest(body)
            with self.assertRaises(Exception):
                self.store.write_terminal(body)
        with self.store.engine().connect() as con:
            self.assertEqual(con.execute(text('SELECT count(*) FROM gov_terminal')).scalar(),0)


if __name__=='__main__':
    unittest.main(verbosity=2)
