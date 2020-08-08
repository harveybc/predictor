from models.user import User

def user_seed():
    tmp =  User(username='test', password='pass', email='test@test.com').save()
