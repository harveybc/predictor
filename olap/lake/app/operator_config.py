"""Validate and stage warehouse settings; the running engine is unchanged."""

import json
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit


def save_pending(config, form):
    proposed = dict(config)
    for field in ('title', 'schema', 'data_gov_url', 'holdout_start'):
        if field in form:
            proposed[field] = form[field].strip()
    holdout = proposed.get('holdout_start')
    if holdout and date.fromisoformat(holdout).isoformat() != holdout:
        raise ValueError('holdout_start must be YYYY-MM-DD')
    proposed['holdout_start'] = holdout or None
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', proposed.get('schema') or 'public'):
        raise ValueError('schema must be an SQL identifier')
    url = urlsplit(proposed.get('data_gov_url') or 'http://127.0.0.1:5055')
    if url.scheme not in {'http', 'https'} or not url.netloc or url.username or url.password:
        raise ValueError('data_gov_url must be HTTP(S) without credentials')
    proposed['data_gov_url'] = url.geturl()
    path = Path(config.get('operator_config_path') or Path(__file__).resolve().parents[1] / 'examples/config/local.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(proposed, indent=2, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix='.config-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path
