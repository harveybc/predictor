"""Restart identity must separate old search state while keeping its contract."""

from pathlib import Path

import pytest

from tools.i6n_poo_restart_config import repaired_plan, restart_domain
from tools import i6n_island_plan as plan


@pytest.mark.parametrize('horizon', [4, 72])
def test_fresh_plan_preserves_search_and_excludes_leaked_column(horizon):
    actual = repaired_plan(horizon)
    original = plan.build_plan_v3() if horizon == 4 else plan.build_plan_v4_ds_in()
    assert actual['search'] == original['search']
    assert actual['domain'] == original['domain']
    assert actual['horizons_hours'] == [horizon]
    assert actual['plan_sha256'] != original['plan_sha256']
    assert len(actual['feature_manifest']['features']) == 19
    assert 'fred.stress.stlfsi4.level' not in actual['feature_manifest']['features']


def test_restart_refuses_changed_evolution_settings():
    opt = dict(population_size=8, n_generations=100, optimization_patience=10,
               min_relative_improvement=1e-8, ga_seed=6102026, seed=6102026,
               shared_population=False)
    with pytest.raises(ValueError, match='population_size'):
        restart_domain(dict(domain_id='old', optimization_config=opt), 4, {}, Path('/runtime'))


def test_fresh_paths_cannot_load_old_population():
    opt = dict(population_size=20, n_generations=100, optimization_patience=10,
               min_relative_improvement=1e-8, ga_seed=6102026, seed=6102026,
               shared_population=False, optimization_resume_file='/old/checkpoint.json')
    actual = restart_domain(dict(domain_id='old', optimization_config=opt), 4, {}, Path('/new'))
    assert actual['optimization_config']['optimization_resume_file'].startswith('/new/state/h4/')
    assert actual['domain_id'] != 'old'
    assert opt['optimization_resume_file'] == '/old/checkpoint.json'
