"""Owner-requested removal of affected I6-DMO metrics, not a whole-cube reset.

The warehouse file must be closed by its owner. Dry-run is the default. No
backup, experiment body or historical result is created by this utility.
"""
import argparse
import json

from tools.i6dmo.compact_curated_cube import keep_clause


def purge(conn,execute=False,include_predecessors=False,include_selection=False):
    tables={r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main' AND table_type='BASE TABLE'").fetchall()}
    counts={}
    def remove(table,where):
        quoted='"'+table.replace('"','""')+'"'
        expected=conn.execute(f'SELECT count(*) FROM {quoted} WHERE {where}').fetchone()[0]
        counts[table]=expected
        if not execute:return
        removed=0
        # Each batch autocommits: bounded deletion state rather than one
        # multi-million-row transaction retaining all index updates in RAM.
        while removed<expected:
            rows=conn.execute(f'DELETE FROM {quoted} WHERE rowid IN '
                              f'(SELECT rowid FROM {quoted} WHERE {where} LIMIT 5000) RETURNING 1').fetchall()
            if not rows:raise RuntimeError('PURGE_PROGRESS_MISMATCH')
            removed+=len(rows)
        if conn.execute(f'SELECT count(*) FROM {quoted} WHERE {where}').fetchone()[0]:
            raise RuntimeError('PURGE_INCOMPLETE')
        conn.execute('CHECKPOINT')
    try:
        if 'gov_report' in tables:
            cond="lower(experiment_key) LIKE 'i6dmo-%' OR lower(experiment_key) LIKE 'i6dmo.%' OR lower(experiment_set_key) LIKE 'i6dmo-%' OR lower(experiment_set_key) LIKE 'i6dmo.%'"
            if include_predecessors:
                cond += " OR (project='predictor' AND (upper(phase) LIKE 'I6DMO%' OR upper(phase) LIKE 'I6N%' OR upper(phase) LIKE 'I6E%' OR upper(phase)='I6A' OR upper(phase)='VALIDATION_REPLAY'))"
            conn.execute('CREATE TEMP TABLE doomed_reports AS SELECT report_sha256 FROM gov_report WHERE '+cond)
            for table in ['gov_metric','gov_dataset','gov_report']:
                if table in tables:
                    where='report_sha256 IN (SELECT report_sha256 FROM doomed_reports)'
                    remove(table,where)
        if include_selection:
            for table in sorted(tables):
                if not table.startswith(('df_fact_','df_dim_feature_selection_','fs_','fs4_','feature_')):
                    continue
                quoted='"'+table.replace('"','""')+'"'
                columns={v[0] for v in conn.execute('SELECT * FROM '+quoted+' LIMIT 0').description}
                keep=keep_clause(table,columns)
                if keep!='TRUE':
                    remove(table,'NOT ('+keep+')')
        if 'gov_terminal' in tables:
            conn.execute("CREATE TEMP TABLE doomed_terminals AS SELECT terminal_sha256 FROM gov_terminal WHERE lower(campaign_key) LIKE 'i6dmo-%' OR lower(campaign_key) LIKE 'i6dmo.%'")
            for table in ['gov_terminal_metric','gov_terminal_dataset','gov_terminal_artifact','gov_terminal']:
                if table in tables:
                    where='terminal_sha256 IN (SELECT terminal_sha256 FROM doomed_terminals)'
                    remove(table,where)
    finally:
        conn.execute('DROP TABLE IF EXISTS doomed_reports')
        conn.execute('DROP TABLE IF EXISTS doomed_terminals')
    if execute:
        conn.execute('CHECKPOINT')
    return dict(state='DELETED' if execute else 'DRY_RUN',counts=counts)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--duckdb',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--include-predecessors',action='store_true')
    p.add_argument('--include-selection',action='store_true')
    a=p.parse_args()
    import duckdb
    conn=duckdb.connect(a.duckdb)
    conn.execute("SET memory_limit='3GB'")
    conn.execute('SET threads=1')
    try:print(json.dumps(purge(conn,a.execute,a.include_predecessors,a.include_selection),sort_keys=True))
    finally:conn.close()


if __name__=='__main__':main()
