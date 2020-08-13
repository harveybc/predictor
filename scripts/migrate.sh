echo "This Script is made to be executed from the visualizer's root directory."
export FLASK_APP=app/db_init.py
export FLASK_ENV=development
flask db_init init