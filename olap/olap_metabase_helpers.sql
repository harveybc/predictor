-- olap_metabase_helpers.sql

-- Recursive JSONB flattener
CREATE OR REPLACE FUNCTION olap.jsonb_deep_each(j JSONB, prefix TEXT DEFAULT '')
RETURNS TABLE(path TEXT, value_text TEXT)
LANGUAGE sql IMMUTABLE AS $$
    WITH RECURSIVE rec(jdata, p) AS (
        SELECT j, prefix
        UNION ALL
        SELECT e.value,
               CASE
                   WHEN jsonb_typeof(e.value) IN ('object','array')
                        THEN p || '.' || e.key
                   ELSE p || '.' || e.key
               END
        FROM rec r,
             jsonb_each(r.jdata) e
        WHERE jsonb_typeof(r.jdata) = 'object'
    )
    SELECT p, value::text
    FROM rec r, jsonb_each_text(r.jdata) e
    WHERE jsonb_typeof(r.jdata) <> 'object';
$$;

-- View to expose all config key/values
CREATE OR REPLACE VIEW olap.v_experiment_config_kv AS
SELECT e.experiment_id,
       e.experiment_key,
       kv.path,
       kv.value_text
FROM olap.dim_experiment e,
     LATERAL olap.jsonb_deep_each(e.config, '') kv;
