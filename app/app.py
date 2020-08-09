# -*- encoding: utf-8 -*-

from flask import Flask, url_for
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from importlib import import_module
from logging import basicConfig, DEBUG, getLogger, StreamHandler
from os import path

db = SQLAlchemy()
login_manager = LoginManager()

def register_extensions(app):
    db.init_app(app)
    login_manager.init_app(app)

def register_blueprints(app):
    for module_name in ('base', 'home'):
        module = import_module('app.{}.routes'.format(module_name))
        app.register_blueprint(module.blueprint)

# If it is the first time the app is run, create the database and perform data seeding
def configure_database(app):

    #@app.before_first_request
    def initialize_database():
        print("Creating database")
        from models.user import User
        db.create_all()
        print("Seeding database")
        from models.user_seed import user_seed
        user_seed(app, db)

    @app.teardown_request
    def shutdown_session(exception=None):
        db.session.remove()
    

def create_app(config):
    app = Flask(__name__, static_folder='base/static')
    app.config.from_object(config)
    register_extensions(app)
    register_blueprints(app)
    configure_database(app)
    
    return app
