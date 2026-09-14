#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.config_handler import load_config
from app.config_merger import merge_config, process_unknown_args
from app.lake_auth import load_token
from app.plugin_loader import get_plugin_params, load_plugin

GROUPS = {
    "pipeline_plugin": "olaplake.pipeline",
    "web_plugin": "olaplake.web",
    "query_plugin": "olaplake.query",
}


def _instantiate(group, name, config):
    cls, _ = load_plugin(group, name)
    plugin = cls()
    plugin.set_params(**config)
    return plugin


def assemble(config: dict[str, Any]):
    return {
        "pipeline": _instantiate("olaplake.pipeline", config["pipeline_plugin"], config),
        "web": _instantiate("olaplake.web", config["web_plugin"], config),
        "query": _instantiate("olaplake.query", config["query_plugin"], config),
    }


def main(argv=None):
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    args, unknown = parse_args()
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    file_config = {}
    load_path = cli_args.get("load_config")
    if load_path:
        path = Path(load_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        file_config = load_config(path)
    names = dict(DEFAULT_VALUES)
    names.update(file_config)
    names.update(cli_args)
    params = [get_plugin_params(g, names[k]) for k, g in GROUPS.items()]
    config = merge_config(
        DEFAULT_VALUES, params, file_config, cli_args, process_unknown_args(unknown)
    )
    token = load_token()
    if token:
        config["lake_service_token"] = token
    plugins = assemble(config)
    return plugins["pipeline"].run({"config": config, "plugins": plugins})


if __name__ == "__main__":
    raise SystemExit(main())
