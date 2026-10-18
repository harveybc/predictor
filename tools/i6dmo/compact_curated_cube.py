"""Rewrite a closed DuckDB with only unaffected rows and no invalid-run backup.

DuckDB COPY DATABASE (SCHEMA) preserves schema/constraints/views; filtered
inserts avoid loading huge indexes just to delete affected rows. The original
is replaced only after independent counts and row comparisons succeed.
"""
import argparse
import json
import os
from pathlib import Path


def ident(value):return '"'+value.replace('"','""')+'"'


def quote(value):return "'"+str(value).replace("'","''")+"'"


def keep_clause(table,columns):
    tests=[]
    if 'population_id' in columns and table.startswith(('df_','feature_','fs')):
        tests.append("coalesce(upper(population_id),'')<>'EURUSD'")
    if 'run_id' in columns and table.startswith(('df_','feature_','fs')):
        tests.append("coalesce(upper(run_id),'') NOT LIKE '%EURUSD%'")
    if table=='gov_report':
        tests.append("NOT (coalesce(lower(experiment_key),'') LIKE 'i6dmo%' OR coalesce(lower(experiment_set_key),'') LIKE 'i6dmo%')")
        if {'project','phase'}.issubset(columns):
            tests.append("NOT (project='predictor' AND (upper(phase) LIKE 'I6DMO%' OR upper(phase) LIKE 'I6N%' OR upper(phase) LIKE 'I6E%' OR upper(phase)='I6A' OR upper(phase)='VALIDATION_REPLAY'))")
    if table.startswith('gov_doin') and 'experiment' in columns:
        tests.append("coalesce(lower(experiment),'') NOT LIKE 'i6n%'")
    if table=='gov_terminal' and 'campaign_key' in columns:
        tests.append("coalesce(lower(campaign_key),'') NOT LIKE 'i6dmo%' AND coalesce(lower(campaign_key),'') NOT LIKE 'i6n%'")
    if table=='gov_campaign_disposition' and 'campaign_key' in columns:
        tests.append("coalesce(lower(campaign_key),'') NOT LIKE 'i6%'")
        if 'run_id' in columns:tests.append("coalesce(lower(run_id),'') NOT LIKE 'i6%'")
        if 'declared_in' in columns:tests.append("coalesce(lower(declared_in),'') NOT LIKE '%i6n%'")
    return ' AND '.join(tests) if tests else 'TRUE'


def compact(path):
    import duckdb
    path=Path(path)
    target=path.with_name(path.name+'.cleaning')
    if not path.is_file() or target.exists():raise ValueError('SOURCE_REQUIRED_TARGET_MUST_BE_NEW')
    c=duckdb.connect(':memory:')
    counts={}
    try:
        c.execute("SET memory_limit='1GB'")
        c.execute('SET threads=1')
        c.execute('SET preserve_insertion_order=false')
        c.execute('ATTACH '+quote(path)+' AS dirty (READ_ONLY)')
        c.execute('ATTACH '+quote(target)+' AS clean')
        c.execute('COPY FROM DATABASE dirty TO clean (SCHEMA)')
        tables=c.execute("SELECT table_schema,table_name FROM information_schema.tables WHERE table_catalog='dirty' AND table_type='BASE TABLE' ORDER BY table_schema,table_name").fetchall()
        pending=list(tables)
        # Insert parents first using the engine's actual FK dependencies.
        fk=c.execute("SELECT schema_name,table_name,referenced_table FROM duckdb_constraints() WHERE database_name='dirty' AND constraint_type='FOREIGN KEY'").fetchall()
        if ('main','gov_report') in tables:
            fk += [('main',name,'gov_report') for name in ('gov_metric','gov_dataset') if ('main',name) in tables]
        if ('main','gov_doin_projection') in tables and ('main','gov_doin_projection_event') in tables:
            fk.append(('main','gov_doin_projection_event','gov_doin_projection'))
        if ('main','gov_terminal') in tables:
            fk += [('main',name,'gov_terminal') for name in ('gov_terminal_metric','gov_terminal_dataset','gov_terminal_artifact') if ('main',name) in tables]
        ordered=[]
        while pending:
            ready=[r for r in pending if all((s,parent) in ordered for s,t,parent in fk if (s,t)==r)]
            if not ready:raise ValueError('CYCLIC_OR_UNRESOLVED_FOREIGN_KEYS')
            ordered.extend(ready);pending=[r for r in pending if r not in ready]
        selections={}
        for schema,table in ordered:
            src='dirty.'+ident(schema)+'.'+ident(table)
            dst='clean.'+ident(schema)+'.'+ident(table)
            cols={r[0] for r in c.execute('SELECT * FROM '+src+' LIMIT 0').description}
            where=keep_clause(table,cols)
            if table in ('gov_metric','gov_dataset') and 'gov_report' in {t for _,t in tables}:
                where+=' AND report_sha256 IN (SELECT report_sha256 FROM clean.main.gov_report)'
            if table=='gov_doin_projection_event' and ('main','gov_doin_projection') in tables:
                where+=' AND projection_key IN (SELECT projection_key FROM clean.main.gov_doin_projection)'
            if table in ('gov_terminal_metric','gov_terminal_dataset','gov_terminal_artifact') and ('main','gov_terminal') in tables:
                where+=' AND terminal_sha256 IN (SELECT terminal_sha256 FROM clean.main.gov_terminal)'
            query='SELECT * FROM '+src+' WHERE '+where
            c.execute('INSERT INTO '+dst+' '+query)
            expected=c.execute('SELECT count(*) FROM ('+query+')').fetchone()[0]
            actual=c.execute('SELECT count(*) FROM '+dst).fetchone()[0]
            if expected!=actual:raise ValueError('KEPT_ROW_COUNT_MISMATCH')
            # Exact multisets, not rounded aggregate metrics.
            for left,right in [(query,'SELECT * FROM '+dst),('SELECT * FROM '+dst,query)]:
                if c.execute('SELECT count(*) FROM (('+left+') EXCEPT ALL ('+right+'))').fetchone()[0]:
                    raise ValueError('KEPT_ROW_CONTENT_MISMATCH')
            total=c.execute('SELECT count(*) FROM '+src).fetchone()[0]
            counts[schema+'.'+table]=dict(kept=actual,removed=total-actual)
            selections[(schema,table)]=actual
        c.execute('DETACH clean');c.execute('DETACH dirty');c.close();c=None
        check=duckdb.connect(str(target),read_only=True)
        try:
            for (schema,table),n in selections.items():
                if check.execute('SELECT count(*) FROM '+ident(schema)+'.'+ident(table)).fetchone()[0]!=n:
                    raise ValueError('REOPEN_COUNT_MISMATCH')
            for schema,name in check.execute("SELECT schema_name,view_name FROM duckdb_views() WHERE NOT internal").fetchall():
                check.execute('SELECT * FROM '+ident(schema)+'.'+ident(name)+' LIMIT 0')
        finally:check.close()
        before=path.stat().st_size;after=target.stat().st_size
        os.replace(target,path)
        return dict(state='REPLACED_WITH_UNAFFECTED_ROWS',before_bytes=before,after_bytes=after,
                    removed_rows=sum(v['removed'] for v in counts.values()),tables=counts,invalid_backup_created=False)
    finally:
        if c is not None:c.close()
        # Failed candidate is disposable; the original remains untouched.
        if target.exists():target.unlink()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--duckdb',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:raise SystemExit('Explicit --execute required; warehouse owner must be stopped')
    print(json.dumps(compact(a.duckdb),sort_keys=True))


if __name__=='__main__':main()
