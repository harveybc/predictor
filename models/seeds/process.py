from models.process import Process

def seed(app, db):
    with app.app_context():
        tmp =  Process(id=0, name='process0', description='d0', model_link='ml0', training_data_link='tdl0', validation_data_link='vdl0',user_id=0)
        db.session.add(tmp)
        tmp =  Process(id=1, name='process1', description='d1', model_link='ml1', training_data_link='tdl1', validation_data_link='vdl1',user_id=1)
        db.session.add(tmp)
        db.session.commit()