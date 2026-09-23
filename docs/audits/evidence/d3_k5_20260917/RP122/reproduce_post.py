"""RP122 POST: Musashi's RP121 scenarios against the CORRECTED paths.

His probe (docs/audits/evidence/RP121_MUSASHI_REVIEW_2026_09_23/probe.py) is preserved and run unchanged as the PRE at the
reviewed base. On the repaired head it stops at its own third scenario, because the line that exercised the bypass —
`delete_predictions(..., require_acceptance=False)` — now raises TypeError: the parameter no longer exists. That refusal IS the
result for finding 3, and this script records it explicitly together with the other two scenarios and the controls.
"""
import argparse, importlib.util, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(); sys.path.insert(0, str(a.repo))
    T = load("rp122_fixture", a.repo / "tests/test_df_sota_repro.py"); R = T.R
    import torch; torch.set_num_threads(1)
    out = {"scope": "disposable CPU fixtures; warehouse stub; no production mutation", "host_before": R.host_thermals()}
    with tempfile.TemporaryDirectory(prefix="satoshi-rp122-") as tmp:
        base = Path(tmp)

        class Factory:
            def mktemp(self, name):
                d = base / name; d.mkdir(); return d

        world = T.world.__wrapped__(Factory()); unit = world["cell"]["cell_id"]

        def case(name, producer=None):
            place = base / name; place.mkdir()
            root = T._copy(world, place); token = place / "token"; token.write_text("fixture")
            C = T._load("df_mod_e0_close")
            ctx = patch.object(R, "metrics_vault", producer) if producer else patch.object(R, "SotaRefusal", R.SotaRefusal)
            with ctx, patch.object(C, "warehouse_terminals", lambda *args: T._wh(world)(None)):
                R.close(SimpleNamespace(root=root, warehouse_token_file=token, warehouse_url="fixture://", data_path=world["data"],
                                        skip_replay=False, replay_device="cpu"), world["design"])
            return root

        root = case("positive")
        pos = R.accept_catalog(root, world["design"], unit, data_path=world["data"], independent=True)
        out["positive_catalog"] = {"pass": pos["pass"], "refusals": pos["refusals"], "families_complete": pos["independently_accepted_families"],
                                   "unchecked": pos["independent_comparison"]["unchecked"]}
        original = R.metrics_vault
        before = {}

        def plausible_wrong(*args, **kw):
            v = original(*args, **kw)
            before["acf"] = v["autocorrelation"]["channel_mean_residual_per_step"]["acf_by_lag"]
            before["quantiles"] = dict(v["residuals"]["quantiles"]); before["correlation"] = v["global"]["corr_pred_true"]
            v["autocorrelation"]["channel_mean_residual_per_step"]["acf_by_lag"] = [[0.0 if x is not None else None for x in row] for row in before["acf"]]
            v["residuals"]["quantiles"] = {k: 0.0 for k in before["quantiles"]}
            v["global"]["corr_pred_true"] = None
            return v

        bad = case("plausible_wrong", producer=plausible_wrong)
        acc = R.accept_catalog(bad, world["design"], unit, data_path=world["data"], independent=True)
        out["plausible_wrong_catalog"] = {"pass": acc["pass"], "refusals": acc["refusals"], "original": before,
                                          "injected": {"acf": "all defined values replaced by zero", "quantiles": "all zero", "correlation": None},
                                          "independent_correlation": acc["independent_estimators"]["corr_pred_true"],
                                          "fully_independent": acc["independent_comparison"]["fully_independent"]}
        # a locally relabelled diagnostic, with the warehouse payload untouched
        T._accept_evidence(world, root, "diagnostic", {"diagnostic_attachment": root / "REPORT.json"}, "unrelated-subject")
        digest = R.sha_file(root / "REPORT.json")
        receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
        out["wrong_kind_before_local_relabel"] = R.accepted_artifact(root, receipts, T._wh(world), digest, expect_kind="closure", expect_role="closure_report")
        held_before = json.dumps(world["held"], sort_keys=True)
        reg = json.loads((root / R.ACCEPTED_EVIDENCE).read_text())
        reg["entries"][digest].update(kind="closure", role="closure_report", subject="closure")
        (root / R.ACCEPTED_EVIDENCE).write_text(json.dumps(reg, indent=2))
        out["wrong_kind_after_local_relabel"] = R.accepted_artifact(root, receipts, T._wh(world), digest, expect_kind="closure", expect_role="closure_report")
        out["warehouse_unchanged_by_relabel"] = held_before == json.dumps(world["held"], sort_keys=True)
        # the deletion, with every local prerequisite valid and the accepted chain unavailable
        root = case("deletion_bypass")
        ready = T._ready(root, unit, world, base / "backup")
        ready["warehouse"] = lambda campaign: {"current": {}}
        refused = R.delete_predictions(root, [unit], **ready)
        out["delete_default_without_accepted_chain"] = refused["units"][unit]["state"]
        out["delete_default_refusals"] = refused["units"][unit]["preflight"]["refusals"]
        try:
            R.delete_predictions(root, [unit], require_acceptance=False, **ready)
            out["delete_bypass_without_accepted_chain"] = "ACCEPTED_THE_BYPASS"
        except TypeError as exc:
            out["delete_bypass_without_accepted_chain"] = f"REFUSED_AT_THE_INTERFACE: {type(exc).__name__}: {exc}"
        out["bypass_arrays_remaining"] = (root / "attempts" / unit / "arrays.npz").exists()
        import inspect
        out["public_signatures"] = {fn.__name__: ("require_acceptance" in inspect.signature(fn).parameters)
                                    for fn in (R.delete_predictions, R.deletion_preflight)}
    out["host_after"] = R.host_thermals()
    a.output.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("host_before", "host_after")}, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
