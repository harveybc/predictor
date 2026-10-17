"""Independent arithmetic audit; no C1 evaluator, model, fit or TEST import."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np


def audit(spec):
    root = Path(spec['data_root'])
    config = json.loads((root/'CONFIG.json').read_text())
    cal = json.loads((Path(spec['input_dir'])/'CALENDAR.json').read_text())
    stamp = lambda v: int(datetime.fromisoformat(v).timestamp())
    weeks = cal['market_calendar']['weeks']
    opens = np.array([stamp(w['market_open']) for w in weeks])
    closes = np.array([stamp(w['market_close']) for w in weeks])
    grid = np.concatenate([np.arange(a+3600,b+1,3600) for a,b in zip(opens,closes)])
    source = json.loads(Path(spec['source_design']).read_text())
    train = source['splits']['train']
    inner = datetime.fromisoformat(train['end'].replace('Z','+00:00'))-timedelta(weeks=8)
    bounds = {'fit':(stamp(train['start']),int(inner.timestamp())),
              'inner':(int(inner.timestamp()),stamp(train['end'])),
              'validation':tuple(stamp(source['splits']['validation'][k]) for k in ('start','end'))}
    horizons = np.array(config['model']['horizons_hours'])
    summaries = {}
    with np.load(Path(spec['input_dir'])/'INPUT.npz',allow_pickle=False) as z:
        times, prices, features = z['ts'], z['price'], z['x']
    assert times.max() < stamp(source['splits']['test']['start']), 'TEST_SOURCE_READ'
    for split, (start, end) in bounds.items():
        with np.load(root/(split+'.npz'),allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        t, mask = a['row_id'], a['mask']
        w = np.searchsorted(opens,t,side='right')-1
        assert (w>=0).all()
        A = opens[w]+cal['entry_delay_after_open_hours']*3600
        B = closes[w]-cal['force_flat_before_close_hours']*3600
        target = t[:,None]+horizons[None,:]*3600
        ix = np.searchsorted(times,target)
        observed = (ix<len(times)) & (times[np.minimum(ix,len(times)-1)]==target)
        expected = ((t>=A)&(t<B)&(t>=start)&(t<end))[:,None] & (target<B[:,None]) & (target<end) & observed
        np.testing.assert_array_equal(mask,expected,err_msg='BUSINESS_MASK_NOT_EXACT:'+split)
        for j in range(len(horizons)):
            np.testing.assert_array_equal(a['target_price'][mask[:,j],j],prices[ix[mask[:,j],j]])
        slots = t[:,None]+(a['offsets_hours']*3600).astype(np.int64)
        assert (slots[:,-1]==t).all() and (slots<=t[:,None]).all()
        rank = np.searchsorted(grid,slots)
        np.testing.assert_array_equal(grid[rank],slots)
        assert (np.diff(rank,axis=1)==2).all(), 'NOT_MG2_INPUT_WINDOW'
        source_ix = np.searchsorted(times,slots)
        np.testing.assert_array_equal(times[source_ix],slots)
        np.testing.assert_array_equal(features[source_ix],a['x'])
        np.testing.assert_array_equal(prices[np.searchsorted(times,t)],a['origin_price'])
        summaries[split] = dict(rows=len(t),valid_pairs=int(mask.sum()),
                                per_horizon=mask.sum(axis=0).tolist())
    return dict(state='PASS', business_rules=1, horizons=horizons.tolist(), splits=summaries,
                independent_of_C1=True,current_t_included=True,future_inputs=0,test_read=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec',required=True)
    args = p.parse_args()
    print(json.dumps(audit(json.loads(Path(args.spec).read_text())),sort_keys=True))
