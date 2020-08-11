from flask import Blueprint
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import click
from app.app import  create_db
from   decouple import config
import os

app = Flask(__name__)
#db = SQLAlchemy()
create_db(app)

bp_init_db = Blueprint('init_db', __name__)
basedir    = os.path.abspath(os.path.dirname(__file__))

# Set up the App SECRET_KEY
os.environ["SECRET_KEY"] = config('SECRET_KEY', default='S#perS3crEt_007')

# This will create a file in <app> FOLDER
os.environ["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite3')
os.environ["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

@bp_init_db.cli.command('init_db')
def init_db():
    """
    Initialize the database.

    :param safety: confirmation of DB_URI to prevent screw up in prod
    :type safety: bool default True
    :return: None
    """
    begin_reset = False
    response = input("Warning: It will delete any existing data in the db. Do you wish to continue? [Y/N]")
    if response.upper() == "Y":
        begin_reset = True
    else:
        raise SystemExit()
    if begin_reset:
        print("Dropping database")
        db.drop_all()
        print("Creating database")
        from models.user import User
        db.create_all()
        print("Seeding database")
        from models.user_seed import user_seed
        user_seed(app, db)
        click.echo("All tables dropped and recreated")


# you MUST register the blueprint
app.register_blueprint(bp_init_db)


    