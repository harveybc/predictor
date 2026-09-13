#!/usr/bin/env python3
"""Governed predictor run (data-gov docs/04_FLOW_V2.md, section 8).

Every distinct input file of a config is downloaded through data-gov under
the experiment key (hash verified, cached as <cache>/<lake>/<sha256><ext>),
predictor runs on CPU from a config whose outputs all land under --out-dir,
and the results CSV is reported as metrics together with the dataset hashes,
the canonical config hash and the code commit. The receipt is
<out-dir>/GOVERNED_RUN.json. Any failure exits 1 with a one-line reason.

    tools/governed_run.py --load_config <cfg> --experiment-key K
        [--experiment-set-key S] --gov-url http://127.0.0.1:5055
        --api-key-file <file> --lake predictor_examples
        --lake-root examples/data_downsampled --metrics-lake olap_cube
        --out-dir <dir> [--cache-dir <dir>] [--from YYYY-MM-DD --to YYYY-MM-DD]
        -- <extra --flags for app/main.py>

The HTTP client is kept in this file: importing data-gov's `app.client`
would shadow predictor's own `app` package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_KEYS = (
    "x_train_file", "y_train_file",
    "x_validation_file", "y_validation_file",
    "x_test_file", "y_test_file",
)
# Outputs always redirected under --out-dir, with the config's basename when
# it names one, else predictor's default basename, so a config that relies
# on the defaults cannot write into the repository root either.
OUTPUT_DEFAULTS = {
    "results_file": "results.csv",
    "output_file": "prediction.csv",
    "uncertainties_file": "test_uncertainties.csv",
    "save_model": "predictor_model.keras",
    "loss_plot_file": "loss_plot.png",
    "model_plot_file": "model_plot.png",
    "predictions_plot_file": "predictions_plot.png",
}
PRIVATE_DEFAULTS = {"save_config": "config_out.json", "save_log": "debug_out.json"}
DEFAULT_CACHE = "~/.cache/data-gov"
KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
METRIC_ROW = re.compile(r"^\s*(Train|Validation|Test)\s+(.+?)(?:\s+H(\d+))?\s*$", re.I)
CHUNK = 1 << 20


class GovernedRunError(Exception):
    pass


# ---------------------------------------------------------------- pure parts


def resolve_inputs(config: dict, base_dir) -> dict:
    """key -> resolved path for each of the six input keys the config sets."""
    out = {}
    for key in INPUT_KEYS:
        value = config.get(key)
        if value:
            path = Path(str(value)).expanduser()
            out[key] = (path if path.is_absolute() else Path(base_dir) / path).resolve()
    return out


def distinct_paths(inputs: dict) -> list:
    seen = []
    for path in inputs.values():
        if path not in seen:
            seen.append(path)
    return seen


def resource_for(path, lake_root) -> str:
    root = Path(lake_root).resolve()
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        raise GovernedRunError(f"input {path} is outside the lake root {root}") from None


def redirect_outputs(config: dict, out_dir) -> dict:
    out = dict(config)
    out_dir = Path(out_dir)
    for key, default in {**OUTPUT_DEFAULTS, **PRIVATE_DEFAULTS}.items():
        out[key] = str(out_dir / Path(str(config.get(key) or default)).name)
    for key, value in config.items():
        if key.endswith("_plot_file") and key not in OUTPUT_DEFAULTS and value:
            out[key] = str(out_dir / Path(str(value)).name)
    return out


def governed_config(config: dict, cached: dict, out_dir) -> dict:
    out = redirect_outputs(config, out_dir)
    for key, path in cached.items():
        out[key] = str(path)
    return out


def _num(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_results_rows(rows) -> list:
    """Results rows (Metric, Average, Std Dev, Min, Max) -> metric rows.
    `Train MAE H24` -> metric MAE, split train, horizon 24; a label without
    the split or the horizon keeps them null."""
    metrics = []
    for row in rows:
        label = (row.get("Metric") or "").strip()
        if not label:
            continue
        match = METRIC_ROW.match(label)
        if match:
            split = match.group(1).lower()
            metric = match.group(2).strip()
            horizon = int(match.group(3)) if match.group(3) else None
        else:
            split, metric, horizon = None, label, None
        metrics.append({
            "metric": metric,
            "value": _num(row.get("Average")),
            "split": split,
            "horizon": horizon,
            "std_dev": _num(row.get("Std Dev")),
            "min_value": _num(row.get("Min")),
            "max_value": _num(row.get("Max")),
            "unit": None,
        })
    return metrics


def parse_results_csv(path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return parse_results_rows(csv.DictReader(handle))


def canonical_config(effective: dict, identities: dict) -> str:
    """Section 8.4: the six input keys become gov:<lake>/<resource>@<sha256>,
    output paths their basenames, save_config/save_log are dropped, then
    compact sorted JSON. Invariant to the cache and output directories."""
    # load_config names the governed config under the output directory: dropped like save_config/save_log
    out = {k: v for k, v in effective.items() if k not in PRIVATE_DEFAULTS and k != "load_config"}
    for key in INPUT_KEYS:
        if key in identities:
            out[key] = identities[key]
    for key in list(out):
        if (key in OUTPUT_DEFAULTS or key.endswith("_plot_file")) and out[key]:
            out[key] = Path(str(out[key])).name
    return json.dumps(out, sort_keys=True, separators=(",", ":"))


def refuse_governed_overrides(extra) -> None:
    """Extra flags may tune the run (--epochs ...), never replace a governed input, an output or the config."""
    governed = set(INPUT_KEYS) | set(OUTPUT_DEFAULTS) | set(PRIVATE_DEFAULTS) | {"load_config"}
    for token in extra:
        if not str(token).startswith("--"):
            continue
        name = str(token)[2:].split("=", 1)[0]
        if name in governed or name.endswith("_plot_file"):
            raise GovernedRunError(f"--{name} would override a governed input or output; refused")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def code_commit(repo_root) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    return head + ("-dirty" if dirty else "")


def build_report(metrics_lake, metrics, datasets, *, experiment_set_key=None,
                 config_sha256=None, code_commit=None, project=None, phase=None,
                 tags=None) -> dict:
    return {
        "lake": metrics_lake,
        "experiment_set_key": experiment_set_key,
        "config_sha256": config_sha256,
        "code_commit": code_commit,
        "project": project,
        "phase": phase,
        "tags": tags or {},
        "datasets": datasets,
        "metrics": metrics,
    }


# ------------------------------------------------------------------- HTTP


def _error_text(raw: bytes) -> str:
    try:
        return str(json.loads(raw.decode()).get("error") or "")
    except (ValueError, AttributeError):
        return raw[:200].decode(errors="replace").strip()


class GovHttp:
    """Minimal data-gov client over http.client: JSON POST and a streamed,
    hash-verified download with a connect timeout only."""

    def __init__(self, base_url: str, api_key: str, experiment_key: str):
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise GovernedRunError(f"bad --gov-url {base_url!r}")
        self._parts = parts
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Experiment-Key": experiment_key,
        }

    def _connection(self, connect_timeout=30):
        cls = http.client.HTTPSConnection if self._parts.scheme == "https" else http.client.HTTPConnection
        try:
            conn = cls(self._parts.hostname, self._parts.port, timeout=connect_timeout)
            conn.connect()
        except OSError as exc:
            raise GovernedRunError(f"data-gov unreachable at {self._parts.netloc}: {exc}") from None
        conn.sock.settimeout(None)
        return conn

    def _path(self, path: str, params=None) -> str:
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
        return self._parts.path.rstrip("/") + path + (f"?{query}" if query else "")

    def post_json(self, path: str, body: dict):
        data = json.dumps(body, allow_nan=False).encode()
        conn = self._connection()
        try:
            conn.request("POST", self._path(path), body=data,
                         headers={**self._headers, "Content-Type": "application/json"})
            response = conn.getresponse()
            status, raw = response.status, response.read()
        finally:
            conn.close()
        try:
            payload = json.loads(raw.decode() or "{}")
        except ValueError:
            payload = {"error": raw[:200].decode(errors="replace")}
        return status, payload

    def download(self, lake: str, resource: str, cache_dir, start=None, end=None,
                 attempts=20) -> dict:
        params = {"lake": lake, "resource": resource, "from": start, "to": end}
        target_dir = Path(cache_dir) / lake
        target_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(resource).suffix
        for _ in range(attempts):
            conn = self._connection()
            try:
                conn.request("GET", self._path("/api/v1/download", params), headers=self._headers)
                response = conn.getresponse()
                if response.status == 503 and response.getheader("Retry-After"):
                    response.read()
                    wait = _num(response.getheader("Retry-After")) or 30
                    time.sleep(min(wait, 300))
                    continue
                if response.status != 200:
                    raise GovernedRunError(
                        f"download {lake}/{resource}: http {response.status} "
                        f"{_error_text(response.read())}"
                    )
                expected = (response.getheader("X-Content-SHA256") or "").lower()
                if not re.fullmatch(r"[0-9a-f]{64}", expected):
                    raise GovernedRunError(f"download {lake}/{resource}: no X-Content-SHA256")
                info = {
                    "resource": resource,
                    "filename": response.getheader("Content-Disposition") or "",
                    "source_sha256": response.getheader("X-Source-SHA256"),
                    "delivery": response.getheader("X-Delivery"),
                    "time_column": response.getheader("X-Time-Column") or None,
                }
                # one writer, one part file: concurrent runs of the same bytes never share a partial file
                part = target_dir / f"{expected}{ext}.{os.getpid()}.{uuid.uuid4().hex}.part"
                digest = hashlib.sha256()
                size = 0
                with open(part, "wb") as handle:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        digest.update(chunk)
                        handle.write(chunk)
                        size += len(chunk)
            finally:
                conn.close()
            actual = digest.hexdigest()
            if actual != expected:
                part.unlink(missing_ok=True)
                raise GovernedRunError(
                    f"download {lake}/{resource}: sha256 mismatch "
                    f"(lake said {expected[:12]}, got {actual[:12]})"
                )
            final = target_dir / f"{expected}{ext}"
            os.replace(part, final)
            info.update({"path": str(final), "sha256": actual, "bytes": size,
                         "cached": f"{lake}/{expected}{ext}"})
            return info
        raise GovernedRunError(f"download {lake}/{resource}: busy after {attempts} attempts")


# -------------------------------------------------------------------- run


def load_api_key(path) -> str:
    if path:
        key = Path(path).expanduser().read_text(encoding="utf-8").strip()
    else:
        key = (os.getenv("DATA_GOV_API_KEY") or "").strip()
    if not key:
        raise GovernedRunError("no API key: pass --api-key-file or set DATA_GOV_API_KEY")
    return key


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run predictor on data-gov governed inputs and report the metrics.",
        epilog="Arguments after `--` are passed to app/main.py (long flags only).",
    )
    parser.add_argument("--load_config", required=True, help="predictor config JSON")
    parser.add_argument("--experiment-key", required=True)
    parser.add_argument("--experiment-set-key")
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", help="file with the service key (else DATA_GOV_API_KEY)")
    parser.add_argument("--lake", default="predictor_examples", help="data-gov lake of the inputs")
    parser.add_argument("--lake-root", default="examples/data_downsampled",
                        help="directory the lake serves; inputs map to paths relative to it")
    parser.add_argument("--metrics-lake", default="olap_cube")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--from", dest="range_from", metavar="YYYY-MM-DD")
    parser.add_argument("--to", dest="range_to", metavar="YYYY-MM-DD")
    parser.add_argument("--project", default="predictor")
    parser.add_argument("--phase", help="default: the config's parent directory name")
    return parser.parse_args(argv)


def _under_repo(path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def run(args, extra) -> dict:
    key = args.experiment_key
    if not KEY_RE.match(key):
        raise GovernedRunError(f"invalid --experiment-key {key!r}")
    if args.experiment_set_key and not KEY_RE.match(args.experiment_set_key):
        raise GovernedRunError(f"invalid --experiment-set-key {args.experiment_set_key!r}")
    api_key = load_api_key(args.api_key_file)
    config_path = _under_repo(args.load_config)
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.path.expanduser(args.cache_dir)).resolve()
    lake_root = _under_repo(args.lake_root)
    state = {
        "status": "running",
        "experiment_key": key,
        "experiment_set_key": args.experiment_set_key,
        "gov_url": args.gov_url,
        "lake": args.lake,
        "metrics_lake": args.metrics_lake,
        "config": str(config_path),
        "cache_dir": args.cache_dir,
        "out_dir": str(out_dir),
        "range": {"from": args.range_from, "to": args.range_to},
    }
    receipt_path = out_dir / "GOVERNED_RUN.json"

    def checkpoint():
        receipt_path.write_text(json.dumps(state, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    try:
        inputs = resolve_inputs(config, REPO_ROOT)
        if not inputs:
            raise GovernedRunError("the config names none of the six input keys")
        gov = GovHttp(args.gov_url, api_key, key)
        downloads = {}
        for path in distinct_paths(inputs):
            resource = resource_for(path, lake_root)
            downloads[path] = gov.download(
                args.lake, resource, cache_dir, args.range_from, args.range_to
            )
        state["inputs"] = [
            {"role": role, "path": str(path),
             **{k: downloads[path][k] for k in (
                 "resource", "sha256", "bytes", "cached", "source_sha256", "delivery", "time_column")}}
            for role, path in inputs.items()
        ]
        cached = {role: downloads[path]["path"] for role, path in inputs.items()}
        gcfg = governed_config(config, cached, out_dir)
        gcfg_path = out_dir / "governed_config.json"
        gcfg_path.write_text(json.dumps(gcfg, indent=4) + "\n", encoding="utf-8")
        state["governed_config"] = str(gcfg_path)

        refuse_governed_overrides(extra)
        cmd = [sys.executable, "app/main.py", "--load_config", str(gcfg_path), *extra]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO_ROOT))
        state["command"] = cmd
        checkpoint()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        state["exit_code"] = proc.returncode
        if proc.returncode != 0:
            raise GovernedRunError(f"predictor exited {proc.returncode}")

        effective_path = Path(gcfg["save_config"])
        if not effective_path.is_file():
            raise GovernedRunError(f"predictor wrote no effective config at {effective_path}")
        with open(effective_path, encoding="utf-8") as handle:
            effective = json.load(handle)
        for role, cached_path in cached.items():
            used = effective.get(role)
            if used is None or Path(str(used)).resolve() != Path(cached_path).resolve():
                raise GovernedRunError(f"the run did not use the governed input for {role}: {used!r}")
        identities = {
            role: f"gov:{args.lake}/{downloads[path]['resource']}@{downloads[path]['sha256']}"
            for role, path in inputs.items()
        }
        canonical = canonical_config(effective, identities)
        state["config_canonical"] = canonical
        state["config_sha256"] = sha256_text(canonical)
        state["code_commit"] = code_commit(REPO_ROOT)

        results_path = Path(gcfg["results_file"])
        if not results_path.is_file():
            raise GovernedRunError(f"no results CSV at {results_path}")
        metrics = parse_results_csv(results_path)
        if not metrics:
            raise GovernedRunError(f"no metric rows in {results_path}")
        datasets = [
            {"lake": args.lake, "resource": downloads[path]["resource"],
             "sha256": downloads[path]["sha256"], "role": role}
            for role, path in inputs.items()
        ]
        plugin = config.get("predictor_plugin") or config.get("plugin")
        report = build_report(
            args.metrics_lake, metrics, datasets,
            experiment_set_key=args.experiment_set_key,
            config_sha256=state["config_sha256"], code_commit=state["code_commit"],
            project=args.project, phase=args.phase or config_path.parent.name,
            tags={"plugin": str(plugin)} if plugin else {},
        )
        state["report"] = report
        status, receipt = gov.post_json(f"/api/v1/experiments/{key}/metrics", report)
        state["report_status"] = status
        state["receipt"] = receipt
        if status not in (200, 201):
            raise GovernedRunError(f"report refused: http {status} {receipt.get('error', '')}".strip())
        state["status"] = "ok"
        checkpoint()
        return state
    except BaseException as exc:
        state["status"] = "failed"
        state["reason"] = f"{type(exc).__name__}: {exc}" if not isinstance(exc, GovernedRunError) else str(exc)
        checkpoint()
        raise


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    args = parse_args(argv)
    try:
        state = run(args, extra)
    except GovernedRunError as exc:
        print(f"governed_run: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"governed_run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    receipt = state["receipt"]
    print(
        f"governed_run: {state['experiment_key']} report_sha256={receipt.get('report_sha256')} "
        f"lineage={receipt.get('lineage')} http={state['report_status']} "
        f"receipt={Path(state['out_dir']) / 'GOVERNED_RUN.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
