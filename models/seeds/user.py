from models.user import User

def seed(app, db):
    with app.app_context():
        tmp =  User(id=0, username='test', password='pass', email='test@test.com')
        db.session.add(tmp)
        db.session.commit()