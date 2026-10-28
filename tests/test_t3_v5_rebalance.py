import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from rebalance_expert_t3_v5 import partition_pending
from prepare_expert_package_t3_v5 import selected_jobs


def sample():
    names = [f't3_unit_{i:02d}' for i in range(7, 45, 2)]
    states = {n: {'state': 'DONE'} for n in names[:8]}
    states[names[8]] = {'state': 'RUNNING'}
    return names, states, names[8]


def test_disjoint_complete_preserves_current():
    names, states, current = sample()
    p = partition_pending(names, states, current, 4)
    assert p['slow'] == names[-4:]
    assert p['fast'] == names[9:-4]
    assert p['retained'] == names[:9]
    assert sorted(sum(p.values(), [])) == sorted(names)
    assert len(set(sum(p.values(), []))) == len(names)


@pytest.mark.parametrize('state', ['FAILED', 'RETRY_PENDING', 'RUNNING'])
def test_entered_tail_rejected(state):
    names, states, current = sample()
    states[names[-1]] = {'state': state, 'entered': 1}
    with pytest.raises(ValueError):
        partition_pending(names, states, current, 4)


def test_cursor_and_duplicate_rejected():
    names, states, current = sample()
    for bad_names, bad_current in [(names + [names[0]], current), (names, names[0])]:
        with pytest.raises(ValueError):
            partition_pending(bad_names, states, bad_current, 4)


def test_shard_admission():
    assert selected_jobs('slow', ['t3_unit_43']) == ['t3_unit_43']
    for bad in ([], ['t3_unit_44'], ['t3_unit_01'] * 2):
        with pytest.raises(ValueError):
            selected_jobs('slow', bad)


@pytest.mark.parametrize('live,complete', [(True, True), (False, False), (False, True)])
def test_native_boundary_and_restart(tmp_path, monkeypatch, live, complete):
    import rebalance_expert_t3_v5 as h
    source, target, control = (tmp_path / n for n in ('source', 'target', 'control'))
    cfg = dict(source=str(source), control=str(control), fast_root=str(target),
               source_service='old.service', fast_service='new.service', timer='handoff.timer',
               partition=dict(retained=['x'], fast=['y'], slow=['z']))
    original = dict(root=str(source), pins={}, t3_manifest={'id': 'same'})
    h.w.atomic(source / 'ops/CONFIG.private.json', original)
    h.w.atomic(source / 'ops/JOBS.json', {'x': {'state': 'RUNNING'}})
    h.w.atomic(target / 'ops/CONFIG.private.json', dict(only_jobs=['y'], pins={},
               t3_manifest=original['t3_manifest'], timer_unit='new.timer'))
    h.w.atomic(control / 'FROZEN.json', dict(source_config_sha256=h.w.identity(original),
               manager_pid=99999999, current='x'))
    monkeypatch.setattr(h.w, 'live_children', lambda root: [123] if live else [])
    monkeypatch.setattr(h, 'launcher_live', lambda c: False)
    monkeypatch.setattr(h.w, 'planned_jobs', lambda c: [{'name': 'x'}])
    calls = []
    def seal(*a):
        calls.append('seal')
        if not complete:
            raise ValueError('incomplete')
    monkeypatch.setattr(h.w, 'seal_result', seal)
    monkeypatch.setattr(h.w, 'publish_status', lambda *a, **k: None)
    monkeypatch.setattr(h.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(h.os, 'kill', lambda *a: pytest.fail('unexpected signal'))
    if not live and not complete:
        with pytest.raises(ValueError, match='incomplete'):
            h.tick(cfg)
        assert calls == ['seal']
    else:
        h.tick(cfg)
        if live:
            assert calls == []
        else:
            assert calls[0] == 'seal'
            assert (control / 'HANDOFF.json').exists()
            h.tick(cfg)
            assert calls.count('seal') == 1


def test_index_rejects_duplicates_and_missing_assignments(tmp_path):
    from index_expert_t3_shards import collect
    for units in (['t3_unit_01'] * 22, ['t3_unit_01']):
        with pytest.raises(ValueError, match='COVERAGE'):
            collect({'shards': [{'units': units}]})


def test_index_empty_mirrors_never_claim_complete(tmp_path):
    from index_expert_t3_shards import collect
    result = collect({'shards': [{'units': [f't3_unit_{i:02d}' for i in range(1, 45, 2)],
                                'mirror': str(tmp_path)}]})
    assert result['done'] == 0 and result['pending'] == 22
