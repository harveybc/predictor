from flask import Blueprint
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import click
from app.app import  create_db
from config import config_dict

app = Flask(__name__)

bp_init_db = Blueprint('db_init', __name__)
# TODO: make DEBUG/PRODUCTION MODE  parametrizable
app_config = config_dict['Debug']
#conf = Config()
app.config.from_object(app_config)
#print ("app.config = ",app_config)
#db = SQLAlchemy()

db = create_db(app)


@bp_init_db.cli.command('init')
def init():
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
        # import all the models
        from models.user import User
        from models.process import Process
        from models.training_progress import TrainingProgress
        from models.validation_plots import ValidationPlots
        from models.validation_stats import ValidationStats
        print("Dropping database")
        db.drop_all()
        print("Creating database")
        db.create_all()
        print("Seeding database...")
        from models.seeds.user import seed
        seed(app, db)
        print("    Info:    Default user test:pass created. Delete the test user at will.")
        print("    Warning: Do not forget to delete the test user after creating a new user.")
        
        
        from models.seeds.process import seed
        print("    Info:    Two default processes named process0 and process1 were created. Delete them discretionally.")
        
        seed(app, db)
        
        click.echo("Note: If you want training and validation test data seeding, please execute the scripts\\test_data_seed.bat or scripts/test_data_seed.sh from the visualizer's root directory. ")
        click.echo("Migration Done. ")

# you MUST register the blueprint
app.register_blueprint(bp_init_db)



    