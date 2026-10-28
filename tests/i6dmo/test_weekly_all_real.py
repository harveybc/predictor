"""Optional fit-free retained-data acceptance; private locations come only from env."""
import copy
import os
from pathlib import Path

import numpy as np
import pytest

from tools.i6dmo import weekly_all as A, weekly18 as W


@pytest.mark.skipif(not os.environ.get('WEEKLY_ALL_PARENT_SPEC'),reason='retained private data not configured')
def test_real_completed_parent_extension_preserves_seven_heads():
    from tools.i6dmo import continuation_data as D
    parent=W.read(os.environ['WEEKLY_ALL_PARENT_SPEC'])
    A.source_identity(parent)
    p=W.prepared_document(parent,parent['data_root']);W.authenticate(parent['data_root'],p['files'])
    cell=next(c for c in p['cells'] if (Path(parent['artifacts'])/'results'/c['id']/'RESULT.json').exists()
              and W.read(Path(parent['artifacts'])/'results'/c['id']/'RESULT.json')['state']=='COMPLETE')
    result=W.terminal(parent['artifacts'],cell,parent)
    old_record=W.read(Path(parent['data_root'])/cell['id']/'PERIOD.json')
    old={s:D._read(Path(parent['data_root'])/cell['id']/(s+'.npz')) for s in ('FIT','VALIDATION')}
    master=os.environ['WEEKLY_ALL_PARENT_MASTER'];inputs=os.environ['WEEKLY_ALL_INPUT_DIR']
    manifest=D._verify(master,inputs);source=D._read(Path(master)/'SOURCE.npz')
    new,record=A.extend_period(old,old_record,result,source,manifest)
    for split in old:
        A.assert_projection(old[split],new[split])
        assert new[split]['mask'].shape[1]==12
        for j,h in enumerate(A.HEADS):
            mask=new[split]['mask'][:,j]
            targets=new[split]['row_ids'][mask]+h*3600
            assert np.array_equal(new[split]['target_price'][mask,j],source['price'][np.searchsorted(source['ts'],targets)])
    assert record['scales']['naive_mae'][:7]==old_record['scales']['naive_mae']
    assert record['scales']['input_mean']==old_record['scales']['input_mean']
    assert record['scales']['input_std']==old_record['scales']['input_std']
    assert record['training']==parent['training']
    expected=copy.deepcopy(record['model']);expected['horizons_hours']=A.PARENT_HEADS
    assert expected==old_record['model']
