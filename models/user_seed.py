from flask.cli import with_appcontext
from models.user import User

@with_appcontext
def user_seed():
    tmp =  User(username='test', password='pass', email='test@test.com').save()
