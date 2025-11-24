# storm/plugins/loader.py
from importlib import import_module

def load_plugin(name: str):
    # expects e.g. storm.plugins.state.plugin:Plugin
    mod = import_module(f"plugins.{name}.plugin")
    return mod.Plugin()  # class named Plugin inside the module

def load_plugins(names):
    return [load_plugin(n) for n in names]