def upsert_dim(conn, table, key_col, key_val, extra_cols=None):
    extra_cols = extra_cols or {}
    cols = [key_col] + list(extra_cols.keys())
    vals = [key_val] + list(extra_cols.values())
    placeholders = ", ".join([f":{c}" for c in cols])
    colnames = ", ".join(cols)

    if extra_cols:
        sql = f"""
            INSERT INTO olap.{table} ({colnames})
            VALUES ({placeholders})
            ON CONFLICT ({key_col}) DO UPDATE SET {", ".join([f"{c}=EXCLUDED.{c}" for c in extra_cols.keys()])}
            RETURNING *;
        """
    else:
        sql = f"""
            INSERT INTO olap.{table} ({colnames})
            VALUES ({placeholders})
            ON CONFLICT ({key_col}) DO NOTHING
            RETURNING *;
        """

    row = conn.execute(text(sql), dict(zip(cols, vals))).fetchone()
    return row
