"""Seal fresh H4/H72 identities after a coordinated DOIN consensus repair.

Reuses the governed business masks and existing 19-feature input artifacts.
The old chain, populations, candidate directories and checkpoints are retained.
Private endpoints and paths are read from runtime configs and written outside
the repository. This command prepares data and configs; it performs no fits.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from tools import i6n_historical_cell as cell
from tools import i6n_historical_evaluator as evaluator
from tools import i6n_island_plan as island_plan
from tools import i6n_search_plan as search_plan
from tools.i6n_contracts import feature_availability as availability
from tools.i6n_doin_sequence_config import canonical_sha256, network_id
from tools.i6n_h72_no_stlfsi4_prepare import convert_cell


def repaired_plan(horizon: int) -> dict:
    """Retain the approved search, binding its new scope and manifest."""
    if horizon not in (4, 72):
        raise ValueError('restart scope is H4 then H72')
    source = island_plan.build_plan_v3() if horizon == 4 else island_plan.build_plan_v4_ds_in()
    plan = deepcopy(source)
    plan.update(plan_version='5.0.0-poo-restart',
                experiment=f'i6n-poo-restart-v2-h{horizon}',
                sequence_scope='H4 then H72-DS-IN; separate repaired networks',
                horizons_hours=[horizon])
    plan['supersedes'].append(dict(plan_version=source['plan_version'],
                                  plan_sha256=source['plan_sha256'],
                                  reason='owner-authorized clean restart after consensus/migration repair',
                                  fits_under_superseded_plan=None))
    digest = search_plan._sha(search_plan._body(plan))
    plan['plan_sha256'] = digest
    plan['seal']['sha256'] = digest
    return search_plan.verify_plan(plan)


def restart_domain(template: dict, horizon: int, folds: dict, runtime: Path) -> dict:
    """Bind fresh state paths and exactly the approved evolutionary settings."""
    domain = deepcopy(template)
    plan = repaired_plan(horizon)
    tag = f'h{horizon}'
    opt = domain['optimization_config']
    opt.update(i6n_horizon=horizon, i6n_plan=plan,
               i6n_feature_manifest=deepcopy(plan['feature_manifest']), i6n_folds=folds,
               predictor_root=str(runtime / 'code/predictor'),
               i6n_evaluator_command=[str(runtime / 'bin/python'), '-m',
                                      'tools.i6n_historical_evaluator', 'evaluate'],
               optimization_resume_file=str(runtime / 'state' / tag / '{island_id}/checkpoint.json'),
               optimization_parameters_file=str(runtime / 'state' / tag / '{island_id}/champion.json'))
    if horizon == 4:
        opt.pop('i6n_genotype_constraints', None)
    else:
        opt['i6n_genotype_constraints'] = deepcopy(plan['search_constraints'])
    for key, expected in [('population_size', 20), ('n_generations', 100),
                          ('optimization_patience', 10), ('min_relative_improvement', 1e-8),
                          ('ga_seed', 6102026), ('seed', 6102026), ('shared_population', False)]:
        if opt[key] != expected:
            raise ValueError(f'approved setting changed: {key}')
    domain['domain_id'] = f'i6n-poo-restart-v2-h{horizon}'
    return domain


def prepare(args) -> dict:
    root, output, runtime = args.data_root.resolve(), args.output.resolve(), args.runtime_root
    if output.exists():
        raise FileExistsError('never overwrite a published restart configuration')
    old = json.loads(args.old_h4_config.read_text())
    clean = json.loads(args.h72_config.read_text())
    nodes = json.loads(args.nodes.read_text())
    manifest = availability.load_contract()
    target_features = list(availability.MANIFEST_NO_STLFSI4)
    availability.gate(target_features, manifest)
    results = dict(schema='i6n.poo_restart.manifest.v1', seed=6102026, nodes=[], horizons={})
    for horizon, template in [(4, old), (72, clean)]:
        folds = {}
        for fold, item in template['domains'][0]['optimization_config']['i6n_folds'].items():
            source = root / item['i6n_evaluation_context']['base_config']['relpath']
            if cell.file_hash(source) != item['i6n_evaluation_context']['base_config']['sha256']:
                raise ValueError(f'source cell identity changed: {fold}')
            raw = json.loads(source.read_text())
            if horizon == 4:
                legacy = raw['input']['path']
                name = Path(legacy).parent.name
                derived = root / 'inputs' / (name + '-no-stlfsi4')
                pins = {legacy: dict(input_path=f'../../../inputs/{name}-no-stlfsi4/INPUT.npz',
                                    input_sha256=cell.file_hash(derived / 'INPUT.npz'),
                                    clock_path=f'../../../inputs/{name}-no-stlfsi4/CLOCK.json',
                                    clock_sha256=cell.file_hash(derived / 'CLOCK.json'))}
                raw = convert_cell(raw, pins, manifest['contract_sha256'])
            if raw['features'] != target_features:
                raise ValueError('restart requires the exact gated 19-feature manifest')
            target = root / 'c/poo-restart-v2/cells' / f'{fold}_h{horizon}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            target.write_text(json.dumps(raw, sort_keys=True, indent=1) + '\n')
            bound = evaluator.freeze(target, root / f'bindings/poo-restart-v2-h{horizon}' / fold,
                                     f'i6n-poo-restart-v2-h{horizon}', fold, data_root=root)
            cfg = cell.read_config(target)
            data = cell.prepare(cfg)
            origins = {split: evaluator.origins_sha256(evaluator.observations(data, cfg, split)[0])
                       for split in ('train', 'inner')}
            # Excluding one column must preserve the physical scoring population.
            if origins != item['i6n_origins_sha256']:
                raise ValueError(f'origin population changed: {horizon}/{fold}')
            for key in ('counts', 'naive_sums'):
                if bound['i6n_support'][key] != item['i6n_support'][key]:
                    raise ValueError(f'paired baseline population changed: {horizon}/{fold}/{key}')
            folds[fold] = {key: bound[key] for key in ('i6n_evaluation_context', 'i6n_support')}
            folds[fold]['i6n_origins_sha256'] = origins
        domain = restart_domain(template['domains'][0], horizon, folds, runtime)
        digest = canonical_sha256(domain['optimization_config'])
        net = network_id('4e19257e8941caec2ec6a4d581a981a533ab5728c11e58c3d13040049e3cd6d5',
                         [domain['domain_id']])
        results['horizons'][str(horizon)] = dict(config_sha256=digest, network_id=net,
                                               domain_id=domain['domain_id'])
        for node in nodes:
            ident = node['island_id']
            conf = deepcopy(template)
            conf.update(port=int(node['endpoint'].rsplit(':', 1)[1]),
                        bootstrap_peers=[n['endpoint'] for n in nodes if n['island_id'] != ident],
                        domains=[deepcopy(domain)],
                        data_dir=str(runtime / f'doin/h{horizon}' / ident),
                        candidate_archive_root=str(runtime / f'archive/h{horizon}' / ident),
                        archive_reference_spool=str(runtime / f'spool/h{horizon}' / ident),
                        candidate_archive_store_url=node['candidate_archive_store_url'])
            path = output / f'h{horizon}' / ident / 'doin_node.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(conf, sort_keys=True, indent=2) + '\n')
            results['nodes'].append(dict(horizon=horizon, island_id=ident, config=str(path),
                                        config_file_sha256=cell.file_hash(path)))
    (output / 'MANIFEST.json').write_text(json.dumps(results, sort_keys=True, indent=2) + '\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('old-h4-config', 'h72-config', 'nodes', 'data-root', 'output', 'runtime-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), sort_keys=True))


if __name__ == '__main__':
    main()
