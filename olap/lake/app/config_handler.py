import json
from pathlib import Path


def load_config(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(config, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return path
