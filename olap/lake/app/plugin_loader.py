from importlib.metadata import entry_points


def load_plugin(plugin_group: str, plugin_name: str):
    selected = entry_points(group=plugin_group)
    match = next((ep for ep in selected if ep.name == plugin_name), None)
    if match is None:
        raise ImportError(f"Plugin {plugin_name!r} not in {plugin_group!r}")
    plugin_class = match.load()
    return plugin_class, list(getattr(plugin_class, "plugin_params", {}))


def get_plugin_params(plugin_group: str, plugin_name: str):
    plugin_class, _ = load_plugin(plugin_group, plugin_name)
    return dict(getattr(plugin_class, "plugin_params", {}))
