# -*- coding: utf-8 -*-
"""
This File contains the LoadCSV class plugin. 
"""

from app.plugin_base import PluginBase
from numpy import genfromtxt
from sys import exit
from flask import current_app
from app.db import get_db
import json

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class VisSqlite(PluginBase): 
    """ input plugin for the FeatureExtractor class, after initialization, the input_ds attribute is set """

    def __init__(self, conf):
        """ Initializes PluginBase. Do NOT delete the following line whether you have initialization code or not. """
        super().__init__(conf)
        # Insert your plugin initialization code here.
        pass

        #Imported methods
        import ._dashboard 

