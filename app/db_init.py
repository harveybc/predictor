import click
from app.app import create_app, db



# Create an app context for the database connection.
app = create_app()
db.app = app


@click.group()
def cli():
    """Run db related tasks."""
    pass


@click.option(
    "--safety-check/--no-safety-check",
    default=True,
    help="Confirm DB_URI before reset?",
)
def init(safety_check):
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
        db.drop_all()
        db.create_all()
        click.echo("All tables dropped and recreated")


cli.add_command(init)