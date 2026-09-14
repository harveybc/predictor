import sys


def process_unknown_args(unknown_args):
    out = {}
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token.lstrip("-")
        if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
            out[key] = unknown_args[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def convert_type(value):
    if isinstance(value, bool) or value is None:
        return value
    text = str(value)
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return value


def merge_config(defaults, plugin_param_dicts, file_config, cli_args, unknown_args):
    merged = {}
    for params in plugin_param_dicts:
        if params:
            merged.update(params)
    merged.update(defaults or {})
    merged.update(file_config or {})
    cli_keys = [arg.lstrip("-") for arg in sys.argv if arg.startswith("--")]
    for key in cli_keys:
        if key in cli_args and cli_args[key] is not None:
            merged[key] = cli_args[key]
        elif key in unknown_args:
            merged[key] = convert_type(unknown_args[key])
    return merged
