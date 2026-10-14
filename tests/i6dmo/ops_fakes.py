"""CPU-only fakes for the I6-DMO ops lane tests: design adapter, host facts, warehouse, trainer.

Nothing here imports TensorFlow, opens a network socket or reads a credential.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs/i6n/direct_multi_output_static/EXPERIMENT.short.json"
FAKE_TRAINER = Path(__file__).with_name("ops_fake_trainer.py")
GIB = 1024 ** 3
MIB = 1024 ** 2


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fake_adapter():
    """Stand-in for tools.i6dmo.contracts (model lane): pure validation and an effective config."""
    def validate_design(design):
        errors = []
        if design.get("schema") != "i6dmo.design.v1":
            errors.append("schema")
        return dict(ok=not errors, errors=errors, design_sha256=_sha(design))

    def effective_config(design):
        return dict(schema="i6dmo.effective.fake.v1", design=design, design_sha256=_sha(design))

    return SimpleNamespace(validate_design=validate_design, effective_config=effective_config)


def load_design(**overrides):
    design = json.loads(DESIGN.read_text())
    for dotted, value in overrides.items():
        node = design
        keys = dotted.split("__")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return design


def write_design(tmp_path, name="design.json", **overrides):
    path = Path(tmp_path) / name
    path.write_text(json.dumps(load_design(**overrides)))
    return path


def private_spec(tmp_path, *, local_mirror=None, second_mirror=None):
    """A private deployment spec with placeholder role aliases and fake GPU identifiers (no real values)."""
    tmp_path = Path(tmp_path)
    return {
        "schema": "i6dmo.private_deployment.v1",
        "roles": {
            "preferred": {"ssh_alias": None, "host_key": "host-preferred", "python": sys.executable,
                          "workdir": str(ROOT), "run_root": str(tmp_path / "remote-preferred"),
                          "data_root": str(tmp_path), "order": 0,
                          "gpus": [{"class": "rtx5090", "uuid": "GPU-fake-preferred-0", "heavy": True},
                                   {"class": "rtx5070ti", "uuid": "GPU-fake-preferred-1", "heavy": False,
                                    "parallel_slot": False}]},
            "secondary": {"ssh_alias": None, "host_key": "host-secondary", "python": sys.executable,
                          "workdir": str(ROOT), "run_root": str(tmp_path / "remote-secondary"),
                          "data_root": str(tmp_path), "order": 1,
                          "gpus": [{"class": "rtx4090", "uuid": "GPU-fake-secondary-0", "heavy": True}]},
            "coordinator": {"ssh_alias": None, "host_key": "host-coordinator", "python": sys.executable,
                            "workdir": str(ROOT), "run_root": str(tmp_path / "remote-coordinator"),
                            "data_root": str(tmp_path), "order": 2, "gpus": [], "cpu_units": True},
        },
        "warehouse": {"url": "http://warehouse.invalid", "environment_file": str(tmp_path / "absent.env"),
                      "token_env": "I6DMO_FAKE_TOKEN"},
        "mirrors": {"local": str(local_mirror or tmp_path / "mirror-local"),
                    "second_host_role": "secondary",
                    "second_host_path": str(second_mirror or tmp_path / "mirror-second")},
        "launcher": {"crispdm_run": None, "initial_cap_bytes": 7936 * MIB, "wall_seconds": 600},
        "tick_interval_seconds": 300,
    }


def facts(*, mem_available=40 * GIB, mem_total=64 * GIB, slice_available=None, disk_free=500 * GIB,
          boot_id="boot-1", foreign_heavy=0, gpu_free=30 * GIB):
    return dict(boot_id=boot_id, mem_available_bytes=mem_available, mem_total_bytes=mem_total,
                slice_available_bytes=slice_available if slice_available is not None else mem_available,
                disk_free_bytes=disk_free, foreign_heavy_gpu_processes=foreign_heavy,
                gpu_free_bytes={"GPU-fake-preferred-0": gpu_free, "GPU-fake-preferred-1": gpu_free,
                                "GPU-fake-secondary-0": gpu_free})


def facts_provider(per_role=None, **defaults):
    per_role = per_role or {}
    return lambda role: facts(**{**defaults, **per_role.get(role, {})})


def trainer_entrypoints():
    """Every unit kind runs the fake trainer; the kind is passed so the fake can count fits."""
    base = [sys.executable, str(FAKE_TRAINER)]
    return {kind: base + ["--kind", kind] for kind in ("fit", "report", "pair", "final_refit", "final_test")}


def fit_count(counter):
    counter = Path(counter)
    if not counter.exists():
        return 0
    return sum(1 for line in counter.read_text().splitlines() if line.split(":")[0] in ("fit", "final_refit"))


class FakeWarehouse:
    """In-memory governed metrics API: POST /api/v1/metrics is idempotent by report_sha256."""

    def __init__(self, fail_post=0, fail_query=0, corrupt=False):
        self.reports, self.rows = {}, []
        self.posts = 0
        self.fail_post, self.fail_query, self.corrupt = fail_post, fail_query, corrupt

    def post_report(self, report):
        self.posts += 1
        if self.fail_post:
            self.fail_post -= 1
            raise RuntimeError("WAREHOUSE_HTTP_503: fake outage")
        digest = report["report_sha256"]
        if digest not in self.reports:
            self.reports[digest] = report
            for m in report["metrics"]:
                value = m["value"] + (1.0 if self.corrupt else 0.0)
                self.rows.append(dict(report_sha256=digest, metric=m["metric"], value=value, unit=m.get("unit"),
                                      horizon=m.get("horizon"), split=m.get("split")))
        return {"report_sha256": digest}

    def query_metrics(self, report_sha256):
        if self.fail_query:
            self.fail_query -= 1
            raise RuntimeError("WAREHOUSE_HTTP_502: fake readback outage")
        return [dict(r) for r in self.rows if r["report_sha256"] == report_sha256]


CODE = "c" * 40
DATA = {"corpus": {"sha256": "d" * 64}}


def write_long_design(tmp_path, **overrides):
    design = load_design(experiment_id="I6-DMO-LONG-DS12-STATIC-R0", horizons_hours=[18, 24, 36, 48, 60, 72],
                         input__history_span_hours=144, input__spacing_hours=12,
                         **{k: v for k, v in overrides.items() if k == "retraining_frequency"})
    path = Path(tmp_path) / "long-design.json"
    path.write_text(json.dumps(design))
    return path


def strategy_pin_files(tmp_path):
    """Strategy bars/calendar files under the coordinator role's data_root (= tmp_path in the fake spec)."""
    tmp_path = Path(tmp_path)
    out = {}
    for key, name, body in (("strategy_bars", "strategy_bars.npz", b"bars"),
                            ("strategy_calendar", "strategy_calendar.json", b"{}")):
        path = tmp_path / name
        if not path.exists():
            path.write_bytes(body)
        out[key] = {"relpath": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return out


def make_campaign(tmp_path, *, name="root", code_commit=CODE, data_pins=None, prior_roots=(), long=True,
                  strategy_pins=True, **design_overrides):
    from tools.i6dmo import campaign
    tmp_path = Path(tmp_path)
    design = write_design(tmp_path, name=f"{name}-design.json", **design_overrides)
    spec_path = tmp_path / "private_spec.json"
    if not spec_path.exists():
        spec_path.write_text(json.dumps(private_spec(tmp_path)))
    root = tmp_path / name
    pins = dict(data_pins or DATA)
    if strategy_pins:
        for key, pin in strategy_pin_files(tmp_path).items():
            pins.setdefault(key, pin)
    campaign.init_campaign(design, spec_path, root, adapter=fake_adapter(), code_commit=code_commit,
                           data_pins=pins, entrypoints=trainer_entrypoints(),
                           prior_roots=[str(p) for p in prior_roots],
                           long_design=write_long_design(tmp_path, **design_overrides) if long else None)
    return root


def run_worker(root, role, counter, *, provider=None, mode="ok", peak=None, clock=None):
    from tools.i6dmo import worker
    env = {"I6DMO_FAKE_COUNTER": str(counter), "I6DMO_FAKE_MODE": mode}
    if peak is not None:
        env["I6DMO_FAKE_PEAK"] = str(peak)
    kwargs = {} if clock is None else {"clock": clock}
    return worker.worker_once(root, role, facts_provider=provider or facts_provider(),
                              executor=worker.LocalExecutor(), poll_interval=0.02, extra_env=env, **kwargs)
