class Plugin:
    plugin_params = {"pipeline_plugin": "default_pipeline"}

    def __init__(self):
        self.params = dict(self.plugin_params)

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def run(self, context):
        return context["plugins"]["web"].serve(context)
