echo "This Script is made to be executed from the visualizer's root directory."
set FLASK_APP=app/db_init.py
set FLASK_ENV=development
flask db_init init
