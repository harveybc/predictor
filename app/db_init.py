from flask import Blueprint

bp = Blueprint('init_db', __name__)

@bp.cli.command('init_db')
#@click.argument('name')
def init_db():
    """
    Initialize the database.

    :param safety_check: confirmation of DB_URI to prevent screw up in prod
    :type safety_check: bool default True
    :return: None
    """
    begin_reset = False
    if not safety_check:
        begin_reset = True
    else:
        click.echo("Current DB URI:")
        click.echo(app.config["CONFIG_VAR_FOR_DB_URI"])
        response = input("Do you wish to continue? [Y/N]")
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



app.register_blueprint(bp)
def init(safety_check):
    