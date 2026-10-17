import inspect

import pytest

from tools.i6dmo import all_horizons as A


def test_distinct_replay_keeps_distinct_identity():
    assert A.campaign_name({}) == A.NAME
    assert A.campaign_name({'batch_name': 'short-reference-device'}) == 'short-reference-device'
    assert 'campaign_name(spec)' in inspect.getsource(A.terminal)
    assert 'campaign_name(spec)' in inspect.getsource(A.report)


def test_gpu_mismatch_rejected_before_training(monkeypatch):
    monkeypatch.setattr(A, 'command', lambda argv: 'NVIDIA GeForce RTX 4090\n')
    with pytest.raises(ValueError, match='REFERENCE_GPU_MISMATCH'):
        A.verify_reference_device({'expected_gpu_name': 'NVIDIA GeForce RTX 5090',
                                   'gpu_uuid': 'test-device'})
    source = inspect.getsource(A.run)
    assert source.index('verify_reference_device(spec)') < source.index('from tools.i6dmo import train')


def test_reference_device_exact_name_and_no_silent_fallback(monkeypatch):
    monkeypatch.setattr(A, 'command', lambda argv: 'NVIDIA GeForce RTX 5090\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'test-device')
    proof = A.verify_reference_device({'expected_gpu_name': 'NVIDIA GeForce RTX 5090',
                                       'gpu_uuid': 'test-device'})
    assert proof['matches_reference'] is True
    assert proof['gpu_name'] == 'NVIDIA GeForce RTX 5090'
    assert A.verify_reference_device({}) is None


def test_visibility_must_match_pinned_device(monkeypatch):
    monkeypatch.setattr(A, 'command', lambda argv: 'NVIDIA GeForce RTX 5090\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'other-device')
    with pytest.raises(ValueError, match='REFERENCE_GPU_VISIBILITY_MISMATCH'):
        A.verify_reference_device({'expected_gpu_name': 'NVIDIA GeForce RTX 5090',
                                   'gpu_uuid': 'test-device'})


@pytest.mark.parametrize('name', ['bad/name', '', '../other', 'has spaces'])
def test_invalid_campaign_name_rejected(name):
    with pytest.raises(ValueError, match='INVALID_REPLAY_NAME'):
        A.campaign_name({'batch_name': name})


def test_launch_selects_requested_role_not_always_secondary():
    source = inspect.getsource(A.launch)
    assert "spec.get('gpu_role','secondary')" in source
    assert "spec['gpu_uuid']" in source
