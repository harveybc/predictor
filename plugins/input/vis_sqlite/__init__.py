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
        
    def to_json(self,cur, one=False):
        """ Transform the result of an sql execute() into a array of dicts. """
        r = [dict((cur.description[i][0], value) for i, value in enumerate(row)) for row in cur.fetchall()]
        cur.connection.close()
        return (r[0] if r else None) if one else r
    
    #Imported methods
    from ._dashboard import load_data, get_user_id, get_max, get_count, get_column_by_pid, get_columns, get_users, get_user_by_username, get_processes, get_process_by_pid, processes_by_uid
    from ._user import user_create

