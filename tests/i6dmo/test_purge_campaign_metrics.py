import importlib.util

import pytest

duckdb=pytest.importorskip('duckdb')
from tools.i6dmo.purge_campaign_metrics import purge


@pytest.mark.parametrize('execute',[False,True])
def test_scoped_deletion_preserves_unrelated_reports_and_dry_run_rolls_back(execute):
    c=duckdb.connect(':memory:')
    try:
        c.execute('CREATE TABLE gov_report(report_sha256 VARCHAR,experiment_key VARCHAR,experiment_set_key VARCHAR)')
        c.execute('CREATE TABLE gov_metric(report_sha256 VARCHAR,value DOUBLE)')
        c.execute("INSERT INTO gov_report VALUES ('bad','i6dmo-monitor:x','i6dmo-monitor'),('good','other-project','other-project')")
        c.execute("INSERT INTO gov_metric VALUES ('bad',1),('good',2)")
        r=purge(c,execute)
        assert r['counts']=={'gov_metric':1,'gov_report':1}
        assert c.execute('SELECT count(*) FROM gov_report').fetchone()[0]==(1 if execute else 2)
        assert c.execute("SELECT value FROM gov_metric WHERE report_sha256='good'").fetchone()[0]==2
    finally:c.close()


def test_extended_purge_keeps_independent_population_and_other_projects():
    c=duckdb.connect(':memory:')
    try:
        c.execute('CREATE TABLE gov_report(report_sha256 VARCHAR,experiment_key VARCHAR,experiment_set_key VARCHAR,project VARCHAR,phase VARCHAR)')
        c.execute("INSERT INTO gov_report VALUES ('old','old-key','old-set','predictor','I6DMO'),('other','other','other','other-project','I6DMO')")
        c.execute('CREATE TABLE fs_test(population_id VARCHAR,score DOUBLE)')
        c.execute("INSERT INTO fs_test VALUES ('EURUSD',1),('ETH',2)")
        c.execute('CREATE TABLE feature_pair_metrics(population_id VARCHAR,score DOUBLE)')
        c.execute("INSERT INTO feature_pair_metrics VALUES ('EURUSD',3),('ETH',4)")
        c.execute('CREATE TABLE df_fact_feature_selection_decision(run_id VARCHAR,population_id VARCHAR)')
        c.execute("INSERT INTO df_fact_feature_selection_decision VALUES ('phase1-eurusd-final','opaque'),('phase1-final','eth-opaque')")
        r=purge(c,True,True,True)
        assert r['counts']['gov_report']==1 and r['counts']['fs_test']==1
        assert r['counts']['feature_pair_metrics']==1
        assert c.execute('SELECT population_id FROM feature_pair_metrics').fetchall()==[('ETH',)]
        assert c.execute('SELECT project FROM gov_report').fetchall()==[('other-project',)]
        assert c.execute('SELECT population_id FROM fs_test').fetchall()==[('ETH',)]
        assert c.execute('SELECT run_id FROM df_fact_feature_selection_decision').fetchall()==[('phase1-final',)]
    finally:c.close()
