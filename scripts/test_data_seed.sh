echo "This Script is made to be executed from the visualizer's root directory."
echo "Also requires that the database is already created with the migrate.sh script."
echo "Warning: It will add test data to the training_progress, validation_plots and validation stats tables."
cat tests/data/data.sql | sqlite3 db.sqlite3
