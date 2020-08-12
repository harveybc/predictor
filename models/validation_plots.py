 
    
from flask_login import UserMixin
from sqlalchemy import Binary, Column, Integer, Float, String, DateTime, ForeignKey

#from app import db, login_manager
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

from app.base.util import hash_pass

from app.db_init import db
from app.app import login_manager
import datetime

#db = SQLAlchemy()
#login_manager = LoginManager()

class ValidationPlots(db.Model):

    __tablename__ = 'validation_plots'
    
    id = Column(Integer, primary_key=True)
    original = Column(Float)
    predicted = Column(Float)
    predicted2 = Column(Float)
    created=Column(DateTime, default=datetime.datetime.utcnow)
    process_id=Column(Integer, ForeignKey('process.id'))
      
    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0]

            if property == 'password':
                value = hash_pass( value ) # we need bytes here (not plain str)
                
            setattr(self, property, value)

    def __repr__(self):
        return str(self.original)



