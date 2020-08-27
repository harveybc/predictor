# -*- coding: utf-8 -*-
"""
This File contains the LoadCSV class plugin. 
"""

from app.plugin_base import PluginBase
from numpy import genfromtxt
from sys import exit
from flask import current_app
from app.db import get_db

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

    def parse_cmd(self, parser):
        """ Adds command-line arguments to be parsed, overrides base class """
        parser.add_argument("--input_file", help="Input dataset file to load including path.", required=True)
        return parser
    
    def load_data(self, p_config, process_id):
        """load the data for the mse plot for the last training process, also the last validation plot and a list of validation stats."""
        p_config = current_app.config['P_CONFIG']
        db = get_db()
        self.input_ds = []
        for table in p_config['input_plugin_config']['tables']:
            c = 0
            fields = ""
            for f in table['fields']:
                if c > 0:
                    fields = fields + ","
                fields = fields + f
                c = c + 1
            query = db.execute(
                "SELECT " + fields +
                " FROM " + table['table_name'] +
                " t JOIN process p ON t.process_id = " + str(process_id) +
                " ORDER BY t.created DESC"
            ).fetchall()
            self.input_ds.append(query)
        return self.input_ds
        
    def get_user_id(self, username):
        """Search for the user_d, having the username """
        db = get_db()
        result = db.execute(
            "SELECT id FROM user WHERE username="+username
        ).fetchall()        
        return result[0]

    def row2dict(self,row):
        """ Convert a sql query result into a dict object """
        d = {}
        for column in row.__table__.columns:
            d[column.name] = str(getattr(row, column.name))
        return d

    def get_max(self, user_id, table, field ):
        """Returns the maximum of the selected field belonging to the user_id from the specified table."""
        db = get_db()
        #user_id = self.get_user_id(username)
        row = db.execute(
            "SELECT t." + field + ", p.id"
            " FROM " + table + " t, process p, user u"
            " WHERE t.process_id = p.id" +
            " AND p.user_id = " + str(user_id) + 
            " ORDER BY t." + field + " DESC LIMIT 1"
        ).fetchone()
        result = dict(row)        
        return result

    def get_count(self, table):
        """Returns the count of rows in the specified table. """
        db = get_db()
        #user_id = self.get_user_id(username)
        row = db.execute(
            "SELECT COUNT(id) FROM " + table
        ).fetchone() 
        result = dict(row)        
        return result

    def get_column_by_pid(self, table, column, process_id):
        """Returns a column from a table filtered by process_id column. """
        db = get_db()
        #user_id = self.get_user_id(username)
        rows = db.execute(
            "SELECT " + column +
            " FROM " + table + 
            " WHERE process_id = " + str(process_id)
        ).fetchall()
        #result = dict(rows)  
        #rows = dict(zip(rows.keys(), rows))      
        result = [r for r, in rows]
        return result 

# TODO: COMPLETAR 
    def processes_by_uid(self, user_id):
        """Returns a column from a table filtered by user_id column. """
        db = get_db()
        #user_id = self.get_user_id(username)

        #TODO: CAMBIAR POR: BUSCAR POR SEPARADO LOS MSE PARA CADA ID DE PROCESS QUE PRETENEZCA A USER_ID
        p = db.execute(
            "SELECT p.id" +
            " FROM process p"  +
            " WHERE p.user_id = " + str(user_id)

        ).fetchall()
        #  TODO: para cada p, busca los últimos mse y fecha
        result=[]
        result['rows'] = [(pid,tmse,vmse,created,vcreated) for (pid,tmse,vmse,created,vcreated) in rows]

        #result = dict(rows)  
        #rows = dict(zip(rows.keys(), rows))      
        
        return result
        
