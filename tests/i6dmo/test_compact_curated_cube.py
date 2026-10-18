import pytest
duckdb=pytest.importorskip('duckdb')
from tools.i6dmo.compact_curated_cube import compact


def test_copy_schema_and_exact_unaffected_rows_with_constraints_and_views(tmp_path):
    path=tmp_path/'cube.duckdb';c=duckdb.connect(str(path))
    c.execute('CREATE TABLE feature_pair_metrics(id INTEGER PRIMARY KEY,population_id VARCHAR,value DOUBLE)')
    c.execute("INSERT INTO feature_pair_metrics VALUES (1,'EURUSD',1),(2,'ETH',2)")
    c.execute('CREATE VIEW example AS SELECT * FROM feature_pair_metrics')
    c.close();r=compact(path)
    assert r['removed_rows']==1 and not r['invalid_backup_created']
    c=duckdb.connect(str(path))
    try:
        assert c.execute('SELECT * FROM example').fetchall()==[(2,'ETH',2.0)]
        with pytest.raises(duckdb.ConstraintException):c.execute("INSERT INTO feature_pair_metrics VALUES (2,'ETH',3)")
    finally:c.close()


def test_profile_identity_detects_population_stored_as_digest(tmp_path):
    path=tmp_path/'cube.duckdb';c=duckdb.connect(str(path))
    c.execute('CREATE TABLE df_fact_variable_profile(run_id VARCHAR,population_id VARCHAR)')
    c.execute("INSERT INTO df_fact_variable_profile VALUES ('EURUSD:profile','opaque-hash'),('ETH:profile','other-hash')");c.close()
    compact(path);c=duckdb.connect(str(path))
    try:assert c.execute('SELECT run_id FROM df_fact_variable_profile').fetchall()==[('ETH:profile',)]
    finally:c.close()
    assert sorted(p.name for p in tmp_path.iterdir())==['cube.duckdb']


def test_final_selection_run_names_and_terminal_children_are_removed(tmp_path):
    path=tmp_path/'cube.duckdb';c=duckdb.connect(str(path))
    c.execute('CREATE TABLE df_fact_feature_selection_decision(run_id VARCHAR,population_id VARCHAR)')
    c.execute("INSERT INTO df_fact_feature_selection_decision VALUES ('phase1-eurusd-final','opaque'),('phase1-final','eth-opaque')")
    c.execute('CREATE TABLE gov_terminal(terminal_sha256 VARCHAR PRIMARY KEY,campaign_key VARCHAR)')
    c.execute('CREATE TABLE gov_terminal_metric(terminal_sha256 VARCHAR,value DOUBLE)')
    c.execute("INSERT INTO gov_terminal VALUES ('bad','i6dmo-old'),('good','unrelated')")
    c.execute("INSERT INTO gov_terminal_metric VALUES ('bad',1),('good',2)");c.close()
    compact(path);c=duckdb.connect(str(path))
    try:
        assert c.execute('SELECT run_id FROM df_fact_feature_selection_decision').fetchall()==[('phase1-final',)]
        assert c.execute('SELECT * FROM gov_terminal_metric').fetchall()==[('good',2.0)]
    finally:c.close()


def test_unrelated_governed_reports_and_metrics_survive_copy_order(tmp_path):
    path=tmp_path/'cube.duckdb';c=duckdb.connect(str(path))
    c.execute('CREATE TABLE gov_report(report_sha256 VARCHAR PRIMARY KEY,experiment_key VARCHAR,experiment_set_key VARCHAR,project VARCHAR,phase VARCHAR)')
    c.execute('CREATE TABLE gov_metric(report_sha256 VARCHAR,value DOUBLE)')
    c.execute("INSERT INTO gov_report VALUES ('bad','old','old','predictor','I6DMO'),('good','other','other','other-project','LIVE')")
    c.execute("INSERT INTO gov_metric VALUES ('bad',1),('good',2)");c.close()
    compact(path);c=duckdb.connect(str(path))
    try:
        assert c.execute('SELECT report_sha256 FROM gov_report').fetchall()==[('good',)]
        assert c.execute('SELECT * FROM gov_metric').fetchall()==[('good',2.0)]
    finally:c.close()
