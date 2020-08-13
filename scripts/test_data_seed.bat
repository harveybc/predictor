echo "This Script is made to be executed from the visualizer's root directory."
echo "Also requires that the database is already created with the migrate.bat script."
echo "Warning: It will add test data to the training_progress, validation_plots and validation stats tables."
sqlite3 db.sqlite3 ". read tests/data/data.sql"
