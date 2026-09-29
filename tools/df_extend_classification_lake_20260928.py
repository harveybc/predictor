"""CB02: additive registration of the classification benchmark corpora in the EXISTING
`sota_benchmarks` lake.  No second registry, no new service, no new procedure: this rebinds the
same operator adopter the TSL extension used (inventory -> prepare -> rehearsal on a disposable
stack -> additive change -> restart -> live route post-check -> write-once receipt, rollback on
any failure), so every rule that adopter enforces applies unchanged.

What is registered and what is NOT:
  * six resources, three corpora, official splits only, distributor bytes unchanged.  Each
    resource gets its OWN directory: the shared adopter derives a governed campaign key from the
    resource's parent directory name, so two splits under one directory would collide on the
    second campaign.  That collision was found in rehearsal, on the disposable stack, before any
    production change;
  * `untimed`: none of these corpora has an observed availability axis, so delivery is
    whole-resource AS_IS with an UNDECLARED scope and every date range is refused.  That refusal
    is what makes research-only material structurally unusable as live-trading data: a
    point-in-time slice of these resources cannot be served at all;
  * two of the three corpora are NONCOMMERCIAL / RESEARCH-ONLY and say so in the declared sheet
    AND in the per-resource use class.  No commercial or trading entitlement is created here;
  * NOT registered: BANKING77's `categories.json` (the ordered official label vocabulary) is
    pinned by digest below but is not a `.csv`/`.parquet`, which is the only file type the
    deployed provider will deliver; Financial PhraseBank (a zip of latin-1 `.txt`) and MASSIVE
    (per-locale `.json.gz`) are blocked by the same rule and are NOT converted here, because a
    converted file is a derived artifact and would no longer be the distributor's bytes.

Known limitation, reported rather than hidden: the deployed resource-contract validator was
written for time series.  It requires non-empty `event_time_column`, `available_time_column`,
`timezone` and `frequency` strings even for a corpus that has no time axis at all.  On the
`untimed` path none of them is ever parsed, so this registration carries explicit
not-a-time-series sentinels instead of naming a column that does not exist.  The correction
belongs in the provider, not in a fabricated contract.

    python tools/df_extend_classification_lake_20260928.py inventory
    python tools/df_extend_classification_lake_20260928.py prepare
    python tools/df_extend_classification_lake_20260928.py rehearse
    python tools/df_extend_classification_lake_20260928.py adopt
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import df_sota_lake_adopt as S

HOME = Path.home()
HOST = HOME / '.local/state/crispdm-data-foundation/satoshi-sota-lake-adoption-20260922T024837Z/public-panels.host.json'
STATE = HOME / '.local/state/crispdm-data-foundation/classification-extension-20260928'
DOWNLOADS = HOME / 'Downloads/CLASSIFICATION_benchmarks_20260928'
SCRIPT = Path(__file__).resolve()

#: A corpus has no time axis.  These sentinels satisfy the deployed validator's "non-empty string"
#: requirement without naming a column that is not in the file.  The untimed delivery path never
#: reads them; they exist only so the contract digest is well defined.
NO_TIME = {'event_time_column': 'NOT_A_TIME_SERIES_NO_EVENT_TIME',
           'available_time_column': 'NOT_A_TIME_SERIES_NO_AVAILABLE_TIME',
           'timezone': 'NAIVE_WALL_CLOCK', 'time_unit': None,
           'frequency': 'NOT_A_TIME_SERIES_NO_SAMPLING_INTERVAL'}

#: Expected identities.  `sha256` is of the local file; `upstream_oid` is the distributor's own
#: object id for the same bytes (a Hugging Face LFS oid, which IS the sha256, or a git blob id).
#: Inventory refuses any file that does not match both, so no unverified byte is ever registered.
PLAN = [
 {'corpus': 'agnews_zhang2015', 'split': 'train', 'local': 'ag_news_train.parquet',
  'resource': 'agnews_zhang2015_train/train.parquet',
  'sha256': 'fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee',
  'bytes': 18585438, 'rows': 120000, 'oid_kind': 'HF_LFS_SHA256',
  'upstream_oid': 'fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee'},
 {'corpus': 'agnews_zhang2015', 'split': 'test', 'local': 'ag_news_test.parquet',
  'resource': 'agnews_zhang2015_test/test.parquet',
  'sha256': '71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb',
  'bytes': 1234829, 'rows': 7600, 'oid_kind': 'HF_LFS_SHA256',
  'upstream_oid': '71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb'},
 {'corpus': 'fomc_tdw_shah2023', 'split': 'train', 'local': 'fomc_train.csv',
  'resource': 'fomc_tdw_shah2023_train/train.csv',
  'sha256': '3c9ec066b7bbdedc60d553b48e74ae6ca36715b5de2f9000a82e76e909bd76b7',
  'bytes': 422592, 'rows': 1984, 'oid_kind': 'GIT_BLOB',
  'upstream_oid': '00253eaaaa017929bd0c60e61d79923b4eaaec8c'},
 {'corpus': 'fomc_tdw_shah2023', 'split': 'test', 'local': 'fomc_test.csv',
  'resource': 'fomc_tdw_shah2023_test/test.csv',
  'sha256': 'c4b6a660a3cd67f940f59b1b77fc4d2f1b99e56c94eaf54b9298a37647ecfbac',
  'bytes': 103896, 'rows': 496, 'oid_kind': 'GIT_BLOB',
  'upstream_oid': '230d4a28d6781b58a32d6a31d038e7599d623e16'},
 {'corpus': 'banking77_casanueva2020', 'split': 'train', 'local': 'banking77_train.csv',
  'resource': 'banking77_casanueva2020_train/train.csv',
  'sha256': 'b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b',
  'bytes': 839073, 'rows': 10003, 'oid_kind': 'GIT_BLOB',
  'upstream_oid': '98e2543cf482d0dca7bfb175ebe35d98efad95be'},
 {'corpus': 'banking77_casanueva2020', 'split': 'test', 'local': 'banking77_test.csv',
  'resource': 'banking77_casanueva2020_test/test.csv',
  'sha256': 'd12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d',
  'bytes': 239961, 'rows': 3080, 'oid_kind': 'GIT_BLOB',
  'upstream_oid': '799687a8367359432985b8b13d85a2baf73f92dd'},
]

CORPORA = {
 'agnews_zhang2015': {
   'dataset_id': 'ag_news',
   'original_citation': ('X. Zhang, J. Zhao and Y. LeCun, "Character-level Convolutional Networks '
                         'for Text Classification," NIPS, 2015, arXiv:1509.01626.'),
   'distributor': 'huggingface.co/datasets/fancyzhx/ag_news',
   'revision': 'eb185aade064a813bc0b7f42de02595523103ca4',
   'licence': 'UNDECLARED_BY_DISTRIBUTOR (dataset card states "unknown"); the underlying AG corpus '
              'is offered for non-commercial academic research only',
   'use_class': 'BENCHMARK/RESEARCH_ONLY_UNRESOLVED_LICENCE',
   'commercial_use': 'REFUSED', 'trading_use': 'REFUSED',
   'label_order': ['World', 'Sports', 'Business', 'Sci/Tech'],
   'columns': ['text', 'label']},
 'fomc_tdw_shah2023': {
   'dataset_id': 'fomc_communication',
   'original_citation': ('A. Shah, S. Paturi and S. Chava, "Trillion Dollar Words: A New Financial '
                         'Dataset, Task & Market Analysis," ACL, pp. 6664-6679, 2023, '
                         'doi:10.18653/v1/2023.acl-long.368.'),
   'distributor': 'huggingface.co/datasets/gtfintechlab/fomc_communication',
   'revision': '6b0283f55f0005a6d38d49f271d795c21fccc1a3',
   'licence': 'CC BY-NC 4.0',
   'use_class': 'BENCHMARK/RESEARCH_ONLY_NONCOMMERCIAL',
   'commercial_use': 'REFUSED', 'trading_use': 'REFUSED',
   'label_order': ['LABEL_0', 'LABEL_1', 'LABEL_2'],
   'columns': ['index', 'sentence', 'year', 'label', 'orig_index']},
 'banking77_casanueva2020': {
   'dataset_id': 'banking77',
   'original_citation': ('I. Casanueva, T. Temcinas, D. Gerz, M. Henderson and I. Vulic, "Efficient '
                         'Intent Detection with Dual Sentence Encoders," NLP4ConvAI, 2020, '
                         'arXiv:2003.04807.'),
   'distributor': 'github.com/PolyAI-LDN/task-specific-datasets',
   'revision': '9d081458ff52e53cf7e848f414e6e9344e4e6696',
   'licence': 'CC BY 4.0',
   'use_class': 'BENCHMARK/PUBLIC',
   'commercial_use': 'NOT_EVALUATED_HERE', 'trading_use': 'REFUSED',
   'label_order': 'see categories.json, pinned by digest in NOT_REGISTERED below',
   'columns': ['text', 'category']},
}

#: Pinned but NOT deliverable through this lake, with the reason.  Recording them here is what
#: keeps a later run from quietly inventing a substitute.
NOT_REGISTERED = [
 {'artifact': 'banking77 categories.json (ordered official 77-label vocabulary)',
  'source': 'github.com/PolyAI-LDN/task-specific-datasets@9d081458ff52e53cf7e848f414e6e9344e4e6696'
            ':banking_data/categories.json',
  'git_blob': 'cdd2a5c77a4079a455f8fb7e751d1ecee0e2a5a4',
  'sha256': '53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63',
  'reason': 'the deployed provider delivers only .csv and .parquet'},
 {'artifact': 'Financial PhraseBank v1.0 (four agreement subsets)',
  'source': 'huggingface.co/datasets/takala/financial_phrasebank@'
            '8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964:data/FinancialPhraseBank-v1.0.zip',
  'licence': 'CC BY-NC-SA 3.0',
  'reason': 'distributed as a zip of latin-1 .txt; only a derived file would be registrable, and a '
            'derived file is not the distributor bytes'},
 {'artifact': 'MASSIVE intent, en-US and es-ES',
  'source': 'huggingface.co/datasets/mteb/amazon_massive_intent@'
            '940fd47a81eaa7f2cc7b129674d945d618ac38c2 (per-locale .json.gz); original '
            'huggingface.co/datasets/AmazonScience/massive (loading script over a .tar.gz of jsonl)',
  'licence': 'CC BY 4.0',
  'reason': 'same file-type rule; second-stage corpus, deliberately not started'},
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def save(path, doc):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(doc, indent=2) + '\n')
    os.replace(temp, path)


def inventory():
    """Verify every downloaded file against BOTH the expected sha256 and the distributor's own
    object id, then write the inventory.  Nothing is copied into the lake here."""
    STATE.mkdir(parents=True, exist_ok=True)
    records = []
    for item in PLAN:
        source = DOWNLOADS / item['local']
        digest, size = sha(source), source.stat().st_size
        assert digest == item['sha256'], f"{item['local']}: sha256 {digest} != expected"
        assert size == item['bytes'], f"{item['local']}: {size} bytes != expected"
        if item['oid_kind'] == 'GIT_BLOB':
            observed = git_blob(source)
        else:
            observed = digest
        assert observed == item['upstream_oid'], f"{item['local']}: upstream identity mismatch"
        records.append({**item, 'path': str(source), 'verified_upstream_identity': True,
                        **{k: v for k, v in CORPORA[item['corpus']].items()}})
    save(STATE / 'INVENTORY.json', {'records': records, 'not_registered': NOT_REGISTERED,
                                    'script_sha256': sha(SCRIPT)})
    print(f'INVENTORY_VERIFIED {len(records)} resources', flush=True)


def extension(before, records):
    """Additive only: the reduction of the candidate back to `before` must be exactly `before`."""
    after = copy.deepcopy(before)
    settings = after['backend']['settings']
    for r in records:
        resource = r['resource']
        assert resource not in settings['include_globs'], resource
        settings['include_globs'].append(resource)
        settings['untimed'].append(resource)
        settings['resource_contracts'][resource] = dict(NO_TIME)
    reduced = copy.deepcopy(after)
    for r in records:
        for key in ['include_globs', 'untimed']:
            reduced['backend']['settings'][key].remove(r['resource'])
        del reduced['backend']['settings']['resource_contracts'][r['resource']]
    assert reduced == before, 'the candidate change is not purely additive'
    return after


def prepare():
    records = json.loads((STATE / 'INVENTORY.json').read_text())['records']
    for r in records:
        source = Path(r['path'])
        assert sha(source) == r['sha256']
        target = S.STORE_ROOT / r['resource']
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            assert sha(target) == r['sha256'], f'{target} holds different bytes'
        else:
            shutil.copy2(source, target)
        assert sha(target) == r['sha256'] and target.stat().st_size == r['bytes']
    shutil.copy2(HOST, STATE / 'host.before.json')
    candidate = extension(json.loads(HOST.read_text()), records)
    save(STATE / 'host.candidate.json', candidate)
    save(STATE / 'PLAN.json', {
        'scope': 'six additive classification resources; existing TSL resources unchanged; only the '
                 'benchmark lake service restarts',
        'availability': 'UNDECLARED; retrospective corpus; whole-resource AS_IS only; every date '
                        'range refused, which is what forbids any point-in-time or live-trading use',
        'research_only': [c for c, v in CORPORA.items() if v['commercial_use'] == 'REFUSED'],
        'acceptance': ['disposable route per resource', 'matching delivered SHA256',
                       'reconciled terminals', 'warehouse content match',
                       'date-range refusal', 'TSL regression'],
        'host_before_sha256': sha(HOST), 'candidate_sha256': sha(STATE / 'host.candidate.json'),
        'script_sha256': sha(SCRIPT)})
    print('PREPARED', flush=True)


def binding():
    return {'candidate_sha256': sha(STATE / 'host.candidate.json'), 'script_sha256': sha(SCRIPT),
            'inventory_sha256': sha(STATE / 'INVENTORY.json')}


def adopter():
    A = S.bind()
    records = json.loads((STATE / 'INVENTORY.json').read_text())['records']
    original = S.lake_entry()
    candidate = json.loads((STATE / 'host.candidate.json').read_text())
    A.RESOURCES = {r['resource']: r['corpus'] for r in records}
    built = copy.deepcopy(original)
    for r in records:
        built['declared'][r['resource']] = r
    built['entry'].update(candidate['backend']['settings'])
    A.lake_entry = lambda: built

    def host_config(*, port, state_dir, token_file=None):
        result = copy.deepcopy(candidate)
        result['web_port'] = port
        result['operator_config_path'] = str(Path(state_dir) / 'pending.json')
        return result

    A.lake_host_config = host_config
    return A, records


def rehearse():
    A, _ = adopter()
    report = A.rehearse(STATE / 'REHEARSAL.json')
    assert report['route_ok'], 'Disposable route failed; production untouched'
    save(STATE / 'REHEARSED_BINDING.json', binding())
    print('REHEARSAL_PASSED', flush=True)


def adopt():
    A, records = adopter()
    assert not (STATE / 'ADOPTION.json').exists(), 'write-once: an adoption receipt already exists'
    assert json.loads((STATE / 'REHEARSED_BINDING.json').read_text()) == binding()
    plan = json.loads((STATE / 'PLAN.json').read_text())
    assert sha(HOST) == plan['host_before_sha256'], 'the live host config moved since prepare'
    gov_sha = sha(A.RUNTIME_CONFIG)
    for r in records:
        assert sha(S.STORE_ROOT / r['resource']) == r['sha256']
    candidate = json.loads((STATE / 'host.candidate.json').read_text())
    before = json.loads((STATE / 'host.before.json').read_text())
    assert extension(before, records) == candidate
    report = {'started': A.now_iso(), 'routes': {}, 'adopted': False}
    mutated = False
    try:
        mutated = True
        save(HOST, candidate)
        subprocess.run(['systemctl', '--user', 'restart', S.LAKE_HOST_UNIT], check=True, timeout=120)
        assert A._service_healthy('http://127.0.0.1:5060')
        token = A.API_KEY_FILE.read_text().strip()
        for r in records:
            result = A.route_checks(
                A.GOV_URL, token, cache_dir=STATE / 'cache',
                run_id=f"cb02-classification-{int(time.time())}", resource=r['resource'],
                expect_sha=r['sha256'], outbox_dir=STATE / 'outbox',
                cube_url='http://127.0.0.1:5057', cube_token=A._cube_token())
            report['routes'][r['resource']] = result
            save(STATE / 'ADOPTION.progress.json', report)
            assert result['route_complete'], r['resource']
        assert sha(A.RUNTIME_CONFIG) == gov_sha
        report['data_gov_config_unchanged'] = True
        report['adopted'] = True
        report['service'] = A.service_state(S.LAKE_HOST_UNIT)
        save(S.STORE_ROOT / 'CLASSIFICATION_EXTENSION_20260928.json', {
            'resources': records, 'not_registered': NOT_REGISTERED,
            'receipt': str(STATE / 'ADOPTION.json'),
            'availability': 'UNDECLARED', 'use_class': 'BENCHMARK/PUBLIC and '
            'BENCHMARK/RESEARCH_ONLY_* per resource; see each record'})
    except BaseException as exc:
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
        if mutated:
            try:
                shutil.copy2(STATE / 'host.before.json', HOST)
                subprocess.run(['systemctl', '--user', 'restart', S.LAKE_HOST_UNIT], check=True,
                               timeout=120)
                report['rollback_verified'] = (sha(HOST) == sha(STATE / 'host.before.json')
                                               and A._service_healthy('http://127.0.0.1:5060'))
                report['rollback_content_equal'] = json.loads(HOST.read_text()) == before
            except BaseException as rollback:
                report['rollback_failure'] = str(rollback)
        raise
    finally:
        report['finished'] = A.now_iso()
        save(STATE / 'ADOPTION.json', report)
    print('ADOPTED: ' + ', '.join(sorted({r['corpus'] for r in records})), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['inventory', 'prepare', 'rehearse', 'adopt'])
    globals()[ap.parse_args().action]()
