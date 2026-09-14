import sqlite3

import pytest
from sqlalchemy import event

from query_plugins.sql_query import Plugin


def warehouse(tmp_path):
    db = tmp_path / 'schema.sqlite'
    with sqlite3.connect(db) as connection:
        connection.executescript('''
            CREATE TABLE experiments (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE metrics (id INTEGER PRIMARY KEY, experiment_id INTEGER REFERENCES experiments(id), value REAL);
            CREATE VIEW metric_values AS SELECT experiment_id, value FROM metrics;
        ''')
    plugin = Plugin()
    plugin.set_params(sqlite_path=str(db))
    plugin.engine()
    return plugin


def test_inventory_includes_views_without_count_scans(tmp_path):
    plugin = warehouse(tmp_path)
    sql = []
    event.listen(plugin.engine(), 'before_cursor_execute', lambda conn, cursor, statement, params, ctx, many: sql.append(statement))
    inventory = {item['resource_id']: item for item in plugin.discover()}
    assert inventory['metric_values']['kind'] == 'view'
    assert inventory['experiments']['kind'] == 'table'
    assert inventory['experiments']['rows'] is None
    assert all('COUNT(*)' not in statement.upper() for statement in sql)


def test_reflection_covers_columns_and_relationships(tmp_path):
    plugin = warehouse(tmp_path)
    details = plugin.resource_schema('experiments')
    assert details['primary_key'] == ['id']
    name = next(column for column in details['columns'] if column['name'] == 'name')
    assert name['nullable'] is False
    assert name['type'] == 'TEXT'
    foreign_key = plugin.resource_schema('metrics')['foreign_keys'][0]
    assert foreign_key['referred_table'] == 'experiments'
    assert foreign_key['constrained_columns'] == ['experiment_id']
    assert plugin.resource_schema('metric_values')['kind'] == 'view'
    with pytest.raises(ValueError, match='not in inventory'):
        plugin.resource_schema('missing')
