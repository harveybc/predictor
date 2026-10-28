"""A market-time campaign whose report, pair and strategy stages are the REAL code paths.

- designs: the REAL sealed MG2/MG12 designs (``docs/i6n/direct_multi_output_static``);
- data: REAL market bundles built by the data lane (``market_grid.build_market_bundle``) from the
  synthetic input fixture (no real market data, no TEST row);
- fit: :mod:`tests.i6dmo.ops_prediction_fixture` writes predictions and trains nothing;
- report / pair / strategy: the default ``python -m tools.i6dmo.unit_runner --kind <kind>`` entry
  points, i.e. ``reporting.run_unit`` (default loader, no arrays_loader) and
  ``strategy_adapter.run_unit`` with the pinned policy;
- every unit goes through ``worker.worker_once``: admission, queue claim, governed subprocess,
  stored-terminal verification and outbox enqueue.

Nothing here imports a trainer, opens a network socket or reads a credential.
"""
import hashlib
import json
import sys
from pathlib import Path

from tests.i6dmo import _synth
from tests.i6dmo.ops_fakes import INSTRUMENT, facts_provider, private_spec, sealed_market_time_designs

FIXTURE_FIT = [sys.executable, "-m", "tests.i6dmo.ops_prediction_fixture", "--kind", "fit"]
FAMILY_RULE = {"short": "MG2_MARKET_EXACT", "long": "MG12_MARKET_EXACT"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_market_data(data_root):
    """Synthetic input + REAL MG2/MG12 bundles under ``data_root`` (built once per test module)."""
    from tools.i6dmo import data, market_grid
    data_root = Path(data_root)
    src = _synth.make_input(data_root / "input")
    market_grid.build_market_bundle(src, market_grid.market_design(_synth.load_design()),
                                    data_root / "bundles" / "mg2", code_commit="TEST")
    market_grid.build_market_bundle(src, market_grid.market_design(data.long_data_view(_synth.load_design())),
                                    data_root / "bundles" / "mg12", code_commit="TEST")
    return data_root


def market_pins(data_root, *, strategy_pins=False):
    from tools.i6dmo import market_grid
    data_root = Path(data_root)
    pins = {}
    for family, rel in (("short", "bundles/mg2"), ("long", "bundles/mg12")):
        manifest = market_grid.load_market_manifest(data_root / rel)
        pins[f"{family}_bundle"] = dict(relpath=rel, sha256=_sha(data_root / rel / "BUNDLE.json"),
                                        bundle_sha256=manifest["bundle_sha256"], input_sampling=FAMILY_RULE[family])
        pins[f"{family}_input"] = dict(relpath="input", sha256=_sha(data_root / "input" / "INPUT.npz"))
    if strategy_pins:
        for key, name, body in (("strategy_bars", "strategy_bars.npz", b"bars"),
                                ("strategy_calendar", "strategy_calendar.json", b"{}")):
            (data_root / name).write_bytes(body)
            pins[key] = dict(relpath=name, sha256=_sha(data_root / name))
    return pins


def make_real_chain_campaign(tmp_path, data_root, *, name="real", strategy_pins=False, code_commit="c" * 40):
    """Init a MARKET_TIME_PAIR campaign on the sealed designs with the real report/pair/strategy entries."""
    from tools.i6dmo import campaign
    tmp_path = Path(tmp_path)
    spec = private_spec(tmp_path)
    for role in spec["roles"].values():
        role["data_root"] = str(data_root)
    spec_path = tmp_path / f"{name}-private_spec.json"
    spec_path.write_text(json.dumps(spec))
    entries = dict(campaign.DEFAULT_ENTRYPOINTS)
    entries["fit"] = list(FIXTURE_FIT)
    short, long_ = sealed_market_time_designs()
    root = tmp_path / name
    campaign.init_campaign(short, spec_path, root, adapter=None, code_commit=code_commit,
                           data_pins=market_pins(data_root, strategy_pins=strategy_pins), entrypoints=entries,
                           long_design=long_, strategy_execution={"instrument": json.loads(json.dumps(INSTRUMENT))},
                           mode=campaign.MODE_MARKET_TIME)
    return root


def worker(root, role, *, env=None):
    from tools.i6dmo import worker as W
    return W.worker_once(root, role, facts_provider=facts_provider(), executor=W.LocalExecutor(),
                         poll_interval=0.05, extra_env=dict(env or {}))


def drive(root, *, env=None, rounds=10):
    """Finite tick/worker rounds until nothing progresses (no sleeping, no polling)."""
    from tools.i6dmo import campaign
    results = []
    for _ in range(rounds):
        campaign.tick(root)
        progressed = False
        for role in ("preferred", "coordinator"):
            out = worker(root, role, env=env)
            results.append(out)
            if out["result"] not in ("IDLE", "WAIT_ARTIFACT", "CLOSED"):
                progressed = True
        if not progressed:
            break
    campaign.tick(root)
    return results


def states(root):
    from tools.i6dmo import campaign
    q = campaign.open_queue(root)
    try:
        return {u["unit_id"]: (u["state"], u["disposition"], u["cause"]) for u in q.units()}
    finally:
        q.close()
