from models.user import User

def seed(app, db):
    with app.app_context():
        tmp =  User(id=0, username='test', password='pass', admin=True, email='test@test.com')
        db.session.add(tmp)
        tmp =  User(id=1, username='test2', password='pass2', admin=False, email='test2@test.com')
        db.session.add(tmp)
        db.session.commit()