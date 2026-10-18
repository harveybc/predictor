"""Bounded transient recovery for the prequential weekly TEST (RECOVERY_DESIGN.md).

Classification is by TYPE and typed code, never by message text. Transients get an initial
attempt plus at most two retries with fixed cooldowns, every attempt in a durable ledger. A
permanent or unknown cause is terminal on first occurrence. Nothing here predicts, fits, sleeps
or polls: the clock is injected and only compared.
"""
import json
from pathlib import Path
import time
import traceback

from tools.i6dmo import weekly18 as W

MAX_RETRIES = 2
COOLDOWNS = (30, 60)
TRANSIENT_PROCESS = 'TRANSIENT_PROCESS'
RESOURCE_BLOCKED = 'RESOURCE_BLOCKED'
PERMANENT = 'PERMANENT'
UNKNOWN = 'UNKNOWN'
LEDGER = 'ATTEMPTS.json'
REQUIRED_ACTION = {
    TRANSIENT_PROCESS: 'Restore the GPU/runtime on the executing role, then audited recovery of this ONE cell '
                       '(the sealed prediction and the ledger are kept; no automatic restart).',
    PERMANENT: 'Audited recovery: the integrity/contract/version/causality refusal must be understood and '
               'corrected at its source; the cell is never retried automatically.',
    UNKNOWN: 'Audited recovery: classify the exception from its evidence first; unknown causes are never '
             'retried automatically.',
}
#: The module's own contract/integrity/version/causality refusals (ValueError codes). A code that
#: is NOT in this table is UNKNOWN: an unlisted code is never assumed transient.
PERMANENT_CODES = frozenset({
    'PREDICT_INPUT_KEYS', 'PREDICT_INPUT_NONFINITE_OR_FUTURE', 'DONOR_NOT_YET_AVAILABLE',
    'SEALED_FORECAST_IDENTITY', 'SEALED_ORIGINS_CHANGED', 'FORECAST_SHAPE_OR_NONFINITE',
    'OBSERVED_INPUT_CHANGED', 'NAIVE_CACHE_CHANGED', 'UPDATE_CHANGED_PREDICTION',
    'NOT_FOUR_CALENDAR_YEARS', 'TRAIN_CUTOFF_OR_LABEL_MATURITY', 'NO_OBSERVED_MONITOR_PAIRS',
    'OBSERVED_WEEK_NOT_FINISHED', 'RUNTIME_VERSION_MISMATCH', 'ARTIFACT_DIGEST', 'DONOR_PATH',
    'DONOR_WEIGHTS_CHANGED', 'WINNING_MODEL_CHANGED', 'UPDATE_DONOR_CHANGED',
    'WARM_START_WEIGHTS_CHANGED', 'SAVED_CHECKPOINT_RELOAD', 'CONTINUATION_NONFINITE_PREDICTION',
    'CONTINUATION_ZERO_NAIVE', 'PREPARED_CHANGED', 'CELLS_CHANGED', 'REPLICATION_RECIPE_CHANGED',
    'TEST_QUALIFICATION_CHANGED', 'TERMINAL_IDENTITY', 'COMPLETED_UPDATE_RECEIPT_MISSING'})
PERMANENT_TYPES = (AssertionError, KeyError, MemoryError)


class TransientProcessError(RuntimeError):
    """A typed, process-local transient (GPU not enumerated, driver tool unavailable)."""

    def __init__(self, code, detail=''):
        self.code = code
        super().__init__(code + (':' + detail if detail else ''))


class InterruptedAttempt(RuntimeError):
    """An unfinished attempt from an exited process; its cause is UNKNOWN."""


def _tf_transient_types():
    import sys
    tf = sys.modules.get('tensorflow')
    if tf is None:
        return ()
    return (tf.errors.ResourceExhaustedError, tf.errors.UnavailableError)


def classify(exc):
    """(cause_class, cause_code) from the exception TYPE and typed code; message text is ignored."""
    if isinstance(exc, TransientProcessError):
        return TRANSIENT_PROCESS, exc.code
    if isinstance(exc, (ConnectionError, TimeoutError, BlockingIOError)) or isinstance(exc, _tf_transient_types()):
        return TRANSIENT_PROCESS, type(exc).__name__
    if isinstance(exc, PERMANENT_TYPES):
        return PERMANENT, type(exc).__name__
    if isinstance(exc, ValueError):
        code = (exc.args[0] if exc.args and isinstance(exc.args[0], str) else '').split(':', 1)[0]
        return (PERMANENT, code) if code in PERMANENT_CODES else (UNKNOWN, code or type(exc).__name__)
    return UNKNOWN, type(exc).__name__


# --------------------------------------------------------------------------- durable ledger
def ledger(folder):
    path = Path(folder) / LEDGER
    return W.read(path) if path.exists() else dict(schema='i6dmo.prequential_attempts.v1', attempts=[])


def _write(folder, doc):
    W.B.put(Path(folder) / LEDGER, doc)


def next_attempt_at(doc):
    """When the next attempt may start, or None when nothing is pending."""
    attempts = doc['attempts']
    return attempts[-1].get('next_attempt_at') if attempts and attempts[-1]['outcome'] == 'TRANSIENT' else None


def in_cooldown(doc, now):
    at = next_attempt_at(doc)
    return at is not None and now < at


def begin_attempt(folder, now):
    doc = ledger(folder)
    index = len(doc['attempts']) + 1
    doc['attempts'].append(dict(index=index, started_at=now, ended_at=None, outcome='RUNNING',
                                cause_class=None, cause_code=None, exception_type=None, next_attempt_at=None))
    _write(folder, doc)
    return index


def finish_attempt(folder, index, now, outcome, **fields):
    doc = ledger(folder)
    entry = doc['attempts'][index - 1]
    if entry['index'] != index:
        raise ValueError('LEDGER_INDEX')
    entry.update(ended_at=now, outcome=outcome, **fields)
    _write(folder, doc)
    return doc


def record_failure(folder, index, exc, now):
    """Keep the evidence of a failed attempt; decide whether a retry remains. Never deletes anything."""
    folder = Path(folder)
    cause_class, code = classify(exc)
    evidence = folder / 'attempts' / f'a{index:02d}' / 'FAILURE.json'
    W.B.put(evidence, dict(attempt=index, cause_class=cause_class, cause_code=code,
                           exception_type=type(exc).__name__, message=str(exc)[:2000],
                           traceback=traceback.format_exc()[-4000:], at=now))
    transient_so_far = sum(1 for a in ledger(folder)['attempts'] if a['outcome'] == 'TRANSIENT')
    retry = cause_class == TRANSIENT_PROCESS and transient_so_far < MAX_RETRIES
    outcome = 'TRANSIENT' if retry else ('EXHAUSTED' if cause_class == TRANSIENT_PROCESS else cause_class)
    cooldown = COOLDOWNS[min(transient_so_far, len(COOLDOWNS) - 1)] if retry else None
    doc = finish_attempt(folder, index, now, outcome, cause_class=cause_class, cause_code=code,
                         exception_type=type(exc).__name__, next_attempt_at=(now + cooldown) if retry else None,
                         evidence=str(evidence.relative_to(folder)))
    return dict(retry=retry, cause_class=cause_class, cause_code=code, cooldown=cooldown,
                attempts=doc['attempts'], required_action=REQUIRED_ACTION[cause_class],
                reason=('TRANSIENT_EXHAUSTED:' if outcome == 'EXHAUSTED' else '') + type(exc).__name__ + ':' + str(exc)[:600])


def recover_failure(folder, now):
    """Called under the worker lock: never reopen an interrupted or terminal attempt."""
    doc = ledger(folder)
    if not doc['attempts']:
        return None
    last = doc['attempts'][-1]
    if last['outcome'] == 'RUNNING':
        return record_failure(folder, last['index'],
                              InterruptedAttempt('Previous process exited without a durable outcome'), now)
    if last['outcome'] not in (PERMANENT, UNKNOWN, 'EXHAUSTED'):
        return None
    evidence = W.read(Path(folder) / 'attempts' / f"a{last['index']:02d}" / 'FAILURE.json')
    return dict(retry=False, cause_class=last['cause_class'], cause_code=last['cause_code'],
                attempts=doc['attempts'], required_action=REQUIRED_ACTION[last['cause_class']],
                reason=('TRANSIENT_EXHAUSTED:' if last['outcome'] == 'EXHAUSTED' else '') +
                       evidence['exception_type'] + ':' + evidence['message'][:600])


def finalization_failure(folder, exc, now):
    """A failed replay is evidence, not another fit attempt or a rewritten completed ledger."""
    cause_class, code = classify(exc)
    reason = type(exc).__name__ + ':' + str(exc)[:600]
    W.B.put(Path(folder) / 'FINALIZATION_FAILURE.json',
            dict(cause_class=cause_class, cause_code=code, reason=reason, at=now))
    return dict(retry=False, cause_class=cause_class, cause_code=code,
                attempts=ledger(folder)['attempts'], required_action=REQUIRED_ACTION[cause_class], reason=reason)


# --------------------------------------------------------------------------- scheduling
READY_PREDECESSOR = ('COMPLETE', 'SKIPPED_NO_EVAL_ROWS')


def classify_pending(cells, results, root, now):
    """Per pending cell: eligible / cooling / blocked, with the dependency named."""
    out = {}
    for c in cells:
        if c['id'] in results:
            continue
        pred = c.get('predecessor')
        doc = ledger(Path(root) / 'results' / c['id'])
        if pred and pred not in results:
            out[c['id']] = dict(state='WAIT_DEPENDENCY', dependency=pred)
        elif pred and results[pred]['state'] not in READY_PREDECESSOR:
            out[c['id']] = dict(state='BLOCKED_DEPENDENCY', dependency=pred,
                                cause=results[pred].get('reason'), cause_class=results[pred].get('cause_class'))
        elif in_cooldown(doc, now):
            last = doc['attempts'][-1]
            out[c['id']] = dict(state='RETRY_PENDING', dependency=pred, attempt=last['index'],
                                cause_code=last['cause_code'], next_attempt_at=last['next_attempt_at'])
        else:
            out[c['id']] = dict(state='ELIGIBLE', dependency=pred, attempt=len(doc['attempts']) + 1)
    return out


def choose(cells, results, root, now):
    """The first eligible, non-cooling cell in plan order; the other family runs during a cooldown."""
    view = classify_pending(cells, results, root, now)
    cell = next((c for c in cells if view.get(c['id'], {}).get('state') == 'ELIGIBLE'), None)
    return cell, view


# --------------------------------------------------------------------------- status
def status(spec, root, results, now=None):
    """Campaign/cell states that never hide a block as RUNNING and never promise an ETA under one."""
    now = time.time() if now is None else now
    root = Path(root)
    progress = W.read(root / 'PROGRESS.json') if (root / 'PROGRESS.json').exists() else {}
    cells = {}
    for cid, r in results.items():
        if r['state'].startswith('FAILED'):
            cells[cid] = dict(state='BLOCKED_TERMINAL', cause_class=r.get('cause_class', UNKNOWN),
                              cause_code=r.get('cause_code'), cause=r.get('reason'),
                              required_action=r.get('required_action', REQUIRED_ACTION[UNKNOWN]),
                              attempts=len(r.get('attempts') or []) or None,
                              dependents=[c['id'] for c in spec['cells'] if c.get('predecessor') == cid])
        else:
            cells[cid] = dict(state=r['state'])
    pending = classify_pending(spec['cells'], results, root, now)
    for cid, view in pending.items():
        cells[cid] = view
    running = progress.get('cell') if progress.get('state') == 'RUNNING' else None
    if running and running in cells and cells[running]['state'] in ('ELIGIBLE', 'RETRY_PENDING'):
        cells[running] = dict(cells[running], state='RUNNING', stage=progress.get('stage'))
    if progress.get('state') == 'WAIT_RESOURCES' and progress.get('cell') in cells:
        cells[progress['cell']] = dict(cells[progress['cell']], state='WAIT_RESOURCES',
                                       memory_available=progress.get('memory_available'),
                                       foreign_gpu_processes=progress.get('foreign_gpu_processes'))
    states = [v['state'] for v in cells.values()]
    blocked = [cid for cid, v in cells.items() if v['state'] in ('BLOCKED_TERMINAL', 'BLOCKED_DEPENDENCY')]
    retry = [cid for cid, v in cells.items() if v['state'] == 'RETRY_PENDING']
    if blocked:
        state = 'BLOCKED'
    elif 'RUNNING' in states:
        state = 'RUNNING'
    elif 'WAIT_RESOURCES' in states:
        state = 'WAIT_RESOURCES'
    elif retry:
        state = 'RETRY_PENDING'
    elif all(s in READY_PREDECESSOR for s in states) and len(cells) == len(spec['cells']):
        state = 'COMPUTE_COMPLETE'
    else:
        state = 'IN_PROGRESS'
    estimable = not blocked and not retry and 'WAIT_RESOURCES' not in states
    return dict(state=state, cells=cells, blocked=blocked, retry_pending=retry,
                next_attempt_at=min((cells[c]['next_attempt_at'] for c in retry), default=None),
                eta_estimable=estimable,
                eta_note=None if estimable else 'NOT_ESTIMABLE: a cell is blocked or waiting for resources/a retry')
