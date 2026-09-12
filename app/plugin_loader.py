#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugin_loader.py

Module for loading plugins using the importlib.metadata entry points API updated for Python 3.12.
Provides functions to load a specific plugin and retrieve its parameters.
"""

from importlib.metadata import entry_points, EntryPoint

def load_plugin(plugin_group: str, plugin_name: str):
    """
    Load a plugin class from a specified entry point group using its name.
    
    This function uses the updated importlib.metadata API for Python 3.12 by filtering 
    entry points with the select() method.

    Args:
        plugin_group (str): The entry point group from which to load the plugin.
        plugin_name (str): The name of the plugin to load.

    Returns:
        tuple: A tuple containing the plugin class and a list of required parameter keys 
               extracted from the plugin's plugin_params attribute.

    Raises:
        ImportError: If the plugin is not found in the specified group.
        Exception: For any other errors during the plugin loading process.
    """
    print(f"Attempting to load plugin: {plugin_name} from group: {plugin_group}")
    try:
        # C31: the loader and the eligibility identity must consult ONE
        # resolver, or they can disagree about which plugin runs — which
        # is exactly what happened: the executor read `predictor_plugin`
        # and the identity read the legacy `plugin`, so a CNN could train
        # while an ANN was recorded. The witness returned here is the
        # same object the identity binds, and it refuses a name two
        # distributions register differently instead of silently taking
        # whichever was installed first.
        from app.plugin_resolver import (PLUGIN_ROLES, load_from_witness,
                                         resolve)
        role = next((r for r, (_k, g, _a) in PLUGIN_ROLES.items()
                     if g == plugin_group), None)
        if role is None:
            raise ImportError(
                f"group {plugin_group!r} is not a declared plugin role")
        witness = resolve(role, plugin_name)
        plugin_class = load_from_witness(witness)
        # Extract the keys from the plugin's plugin_params attribute as required parameters.
        required_params = list(plugin_class.plugin_params.keys())
        print(f"Successfully loaded plugin: {plugin_name} with params: {plugin_class.plugin_params}")
        return plugin_class, required_params
    except StopIteration:
        print(f"Failed to find plugin {plugin_name} in group {plugin_group}")
        raise ImportError(f"Plugin {plugin_name} not found in group {plugin_group}.")
    except Exception as e:
        print(f"Failed to load plugin {plugin_name} from group {plugin_group}, Error: {e}")
        raise

def get_plugin_params(plugin_group: str, plugin_name: str):
    """
    Retrieve the plugin parameters from a specified entry point group using the plugin name.
    
    This function loads the plugin class using the updated importlib.metadata API and returns 
    its plugin_params attribute.

    Args:
        plugin_group (str): The entry point group from which to retrieve the plugin.
        plugin_name (str): The name of the plugin.

    Returns:
        dict: A dictionary representing the plugin parameters (plugin_params).

    Raises:
        ImportError: If the plugin is not found in the specified group.
        ImportError: For any errors encountered while retrieving the plugin parameters.
    """
    print(f"Getting plugin parameters for: {plugin_name} from group: {plugin_group}")
    try:
        # C31: same resolver, same witness. A second lookup here could
        # report the parameters of a plugin other than the one that runs.
        from app.plugin_resolver import (PLUGIN_ROLES,
                                         declared_plugin_params, resolve)
        role = next((r for r, (_k, g, _a) in PLUGIN_ROLES.items()
                     if g == plugin_group), None)
        if role is None:
            raise ImportError(
                f"group {plugin_group!r} is not a declared plugin role")
        params = declared_plugin_params(resolve(role, plugin_name))
        print(f"Retrieved plugin params: {params}")
        return params
    except StopIteration:
        print(f"Failed to find plugin {plugin_name} in group {plugin_group}")
        raise ImportError(f"Plugin {plugin_name} not found in group {plugin_group}.")
    except Exception as e:
        print(f"Failed to get plugin params for {plugin_name} from group {plugin_group}, Error: {e}")
        raise ImportError(f"Failed to get plugin params for {plugin_name} from group {plugin_group}, Error: {e}")
