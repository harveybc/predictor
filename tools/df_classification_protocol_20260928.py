"""CB01/CB02: pin every evaluation population, label vocabulary and subset decision BEFORE any
score can exist, from the bytes the governed route delivered.

The input is the content-addressed delivery cache of the CB02 registration, never the staging
download and never a fresh network fetch.  Every file is re-hashed IN THIS PROCESS and refused
unless its sha256 equals the digest the governed delivery published, so the identities below are
bound to the delivered bytes and not to a name.

What it resolves, and why it has to be resolved now:

  * AG News, the Laya published row.  Laya's benchmark script reads `fancyzhx/ag_news` test and
    takes `list(d)[:400]`.  That is deterministic file order, not a random sample, so the 400 rows
    ARE recoverable -- but the script pins no dataset revision, so the recovery is only meaningful
    against a named revision.  This pins the population against the registered revision and gives
    it a digest.  The 400-row population and the full 7600-row official test are two different
    populations and are kept apart here.
  * BANKING77 version compatibility.  We registered the authors' own CSVs.  MTEB's
    `Banking77Classification` reads a DIFFERENT artefact, the `mteb/banking77` jsonl mirror at its
    own pinned revision, and Laya's script reads that same mirror.  Comparing a number measured on
    one to a number published on the other is only legitimate if the rows are the same rows, so
    that is checked here as a decision, not assumed.
  * Label order.  The authors' `categories.json` order, the alphabetical order the Hugging Face
    loading script assigns, and the order an option list would be written in are three different
    orders.  A prompted classifier's answer depends on the order it is shown, so all of them are
    recorded.

    python tools/df_classification_protocol_20260928.py populations
    python tools/df_classification_protocol_20260928.py naive
"""
import argparse
import hashlib
import json
from pathlib import Path

HOME = Path.home()
CACHE = HOME / '.local/state/crispdm-data-foundation/classification-extension-20260928/cache/sota_benchmarks'
STAGING = HOME / 'Downloads/CLASSIFICATION_benchmarks_20260928'
OUT = Path(__file__).resolve().parent.parent / 'docs/contracts/classification_populations.v1.json'
NAIVE_OUT = Path(__file__).resolve().parent.parent / 'docs/contracts/classification_naive_baselines.v1.json'

#: sha256 -> (governed resource, suffix).  A delivery is identified by its digest, so the digest is
#: the key: if the bytes are not these bytes, nothing below is produced.
DELIVERED = {
 'fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee':
   ('agnews_zhang2015_train/train.parquet', '.parquet'),
 '71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb':
   ('agnews_zhang2015_test/test.parquet', '.parquet'),
 '3c9ec066b7bbdedc60d553b48e74ae6ca36715b5de2f9000a82e76e909bd76b7':
   ('fomc_tdw_shah2023_train/train.csv', '.csv'),
 'c4b6a660a3cd67f940f59b1b77fc4d2f1b99e56c94eaf54b9298a37647ecfbac':
   ('fomc_tdw_shah2023_test/test.csv', '.csv'),
 'b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b':
   ('banking77_casanueva2020_train/train.csv', '.csv'),
 'd12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d':
   ('banking77_casanueva2020_test/test.csv', '.csv'),
}

#: The mirror MTEB and the Laya script actually read.  NOT governed: it is compared here, and a
#: comparison is the whole point -- it is never substituted for the registered authors' bytes.
MIRROR = {
 'test': ('mteb_banking77_test.jsonl',
          'fb1b0043ded745b8767687084786e6dd0a5f0ce03243b6131992a1c7ae2c2595'),
 'train': ('mteb_banking77_train.jsonl',
           'd411780d8c0e18e166f5664c6cfe90dc9de399d722aa7cde282e31a771323ea7'),
}
MIRROR_REVISION = '0fd18e25b25c072e09e0d92ab615fda904d66300'

#: Verbatim from the Laya benchmark script `research/scripts/bench_apps.py`: the option keys and
#: their glosses, in the order the script writes them.  Changing either is a new variant.
LAYA_AGNEWS_CRITERIA = {'world': 'world news and international politics', 'sports': 'sports',
                        'business': 'business and economy', 'sci_tech': 'science and technology'}
LAYA_AGNEWS_INSTRUCTIONS = 'What is the topic of `article`?'
LAYA_AGNEWS_N = 400


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def delivered(digest):
    """Open a delivered file by its digest, re-hashing it here before it is read as data."""
    resource, suffix = DELIVERED[digest]
    path = CACHE / f'{digest}{suffix}'
    if not path.is_file():
        raise SystemExit(f'REFUSED: no governed delivery on disk for {resource}')
    observed = sha(path)
    if observed != digest:
        raise SystemExit(f'REFUSED: {resource} bytes are not the delivered bytes')
    return path, resource


def population_digest(rows):
    """A population identity over ordered (position, text, label) triples: canonical, separator-
    explicit JSON so no formatting choice can change the digest."""
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('ascii')).hexdigest(), len(rows)


def populations():
    import pandas as pd
    doc = {'schema': 'classification_populations.v1', 'produced_before_any_score': True,
           'source': 'the CB02 governed delivery cache, re-hashed in this process',
           'populations': {}, 'label_orders': {}, 'decisions': []}

    # ---- AG News -------------------------------------------------------------------------------
    path, resource = delivered('71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb')
    test = pd.read_parquet(path)
    names = ['World', 'Sports', 'Business', 'Sci/Tech']
    full = [[i, str(t), int(l)] for i, (t, l) in enumerate(zip(test['text'], test['label']))]
    head = full[:LAYA_AGNEWS_N]
    for key, rows, note in (
        ('agnews_test_full', full,
         'the full official Zhang et al. test split; a SEPARATE experiment, never the denominator '
         'of the published sampled row'),
        ('agnews_test_first400_laya_published', head,
         "the population of Laya's published AG News row: `list(d)[:400]`, file order, no seed, "
         'recovered here against a named revision because the author script pins none')):
        digest, n = population_digest(rows)
        counts = {}
        for _, _, label in rows:
            counts[names[label]] = counts.get(names[label], 0) + 1
        doc['populations'][key] = {'resource': resource, 'resource_sha256': sha(path), 'rows': n,
                                   'population_sha256': digest, 'class_counts': counts,
                                   'note': note}
    doc['label_orders']['agnews_huggingface_classlabel'] = names
    doc['label_orders']['agnews_laya_option_keys'] = list(LAYA_AGNEWS_CRITERIA)
    doc['populations']['agnews_test_first400_laya_published']['prompt'] = {
        'instructions': LAYA_AGNEWS_INSTRUCTIONS, 'criteria': LAYA_AGNEWS_CRITERIA,
        'option_count': 4, 'within_the_author_stated_20_option_budget': True}

    # ---- BANKING77: registered authors' bytes ---------------------------------------------------
    b77 = {}
    for split, digest in (('train', 'b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b'),
                          ('test', 'd12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d')):
        path, resource = delivered(digest)
        frame = pd.read_csv(path)
        rows = [[i, str(t), str(c)] for i, (t, c) in enumerate(zip(frame['text'], frame['category']))]
        pop, n = population_digest(rows)
        b77[split] = frame
        doc['populations'][f'banking77_{split}_authors_csv'] = {
            'resource': resource, 'resource_sha256': sha(path), 'rows': n,
            'population_sha256': pop, 'distinct_labels': int(frame['category'].nunique())}
    labels = set(b77['train']['category'])
    ascii_sorted = sorted(labels)
    case_insensitive = sorted(labels, key=str.lower)
    doc['label_orders']['banking77_ascii_sorted'] = ascii_sorted
    doc['label_orders']['banking77_case_insensitive_sorted'] = case_insensitive
    doc['label_orders']['banking77_authors_categories_json_order'] = json.loads(
        (STAGING / 'banking77_categories.json').read_text())
    # A trap worth naming: the two sorts differ.  `Refund_not_showing_up` is first under an ASCII
    # sort and 61st under a case-insensitive one, so an id map built with a bare sorted() would
    # mislabel most classes, and a prompted classifier shown the options in the wrong order is a
    # different experiment.
    doc['label_orders']['banking77_ascii_and_case_insensitive_sorts_differ'] = (
        ascii_sorted != case_insensitive)

    # ---- BANKING77: is the evaluator's mirror the same rows? ------------------------------------
    mirror_report = {'mirror': f'mteb/banking77@{MIRROR_REVISION}', 'splits': {}}
    for split, (name, expect) in MIRROR.items():
        path = STAGING / name
        observed = sha(path)
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        same_order = [str(r['text']) for r in rows] == [str(t) for t in b77[split]['text']]
        same_multiset = sorted((str(r['text']), str(r['label_text'])) for r in rows) == sorted(
            (str(t), str(c)) for t, c in zip(b77[split]['text'], b77[split]['category']))
        by_id = {}
        for r in rows:
            by_id.setdefault(int(r['label']), str(r['label_text']))
        observed_order = [by_id[i] for i in sorted(by_id)]
        mirror_report['splits'][split] = {
            'file_sha256': observed, 'matches_expected_sha256': observed == expect,
            'rows': len(rows), 'authors_rows': int(len(b77[split])),
            'same_text_sequence_in_file_order': same_order,
            'same_text_label_multiset': same_multiset,
            'mirror_label_id_order_is_ascii_sorted': observed_order == ascii_sorted,
            'mirror_label_id_order_is_case_insensitive_sorted':
                observed_order == case_insensitive,
            'mirror_label_id_order_is_the_authors_categories_json_order':
                observed_order == doc['label_orders']['banking77_authors_categories_json_order'],
            'mirror_label_id_order': observed_order}
    doc['populations']['banking77_mteb_mirror'] = mirror_report
    verdict = all(v['same_text_label_multiset'] for v in mirror_report['splits'].values())
    doc['decisions'].append({
        'question': "may a BANKING77 number measured on the registered authors' CSVs be compared "
                    'to a published MTEB `Banking77Classification` number?',
        'resolved_before_any_score': True,
        'answer': 'YES on content' if verdict else 'NO: the artefacts differ',
        'evidence': 'text/label multiset equality against the mirror revision the evaluator pins',
        'caveat': 'row ORDER may still differ; the MTEB evaluator draws its own 8-per-label '
                  'subsample under seed 42, so order-dependent recipes must re-check order'})

    # ---- FOMC: the split that exists versus the split the paper describes ------------------------
    fomc = {}
    for split, digest in (('train', '3c9ec066b7bbdedc60d553b48e74ae6ca36715b5de2f9000a82e76e909bd76b7'),
                          ('test', 'c4b6a660a3cd67f940f59b1b77fc4d2f1b99e56c94eaf54b9298a37647ecfbac')):
        path, resource = delivered(digest)
        frame = pd.read_csv(path)
        fomc[split] = frame
        rows = [[int(i), str(s), int(l)] for i, s, l in
                zip(frame['index'], frame['sentence'], frame['label'])]
        pop, n = population_digest(rows)
        doc['populations'][f'fomc_{split}_distributor_split'] = {
            'resource': resource, 'resource_sha256': sha(path), 'rows': n,
            'population_sha256': pop, 'columns': list(frame.columns),
            'label_counts': {str(k): int(v) for k, v in
                             frame['label'].value_counts().sort_index().items()},
            'year_range': [int(frame['year'].min()), int(frame['year'].max())]}
    overlap = set(fomc['train']['sentence']) & set(fomc['test']['sentence'])
    doc['populations']['fomc_split_integrity'] = {
        'train_rows': int(len(fomc['train'])), 'test_rows': int(len(fomc['test'])),
        'total_rows': int(len(fomc['train']) + len(fomc['test'])),
        'sentences_in_both_splits': len(overlap),
        'no_validation_split_is_distributed': True}
    doc['decisions'].append({
        'question': 'which FOMC split may a reproduction use?',
        'resolved_before_any_score': True,
        'answer': 'ONLY the distributor train/test files registered here, as one fixed split. The '
                  'original paper reports a mean over several seeded splits, so a single-split '
                  'number is NOT the same estimator and must not be printed in the same column.',
        'evidence': 'the distributor publishes exactly two files and no validation split; the '
                    'file names carry one split identifier, not a family of seeds'})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
    print(f'PINNED {len(doc["populations"])} populations -> {OUT}')
    for d in doc['decisions']:
        print('DECISION:', d['answer'])


def _naive(train_labels, test_labels, *, seed=42):
    """Majority and stratified baselines FIT ON TRAIN LABELS ONLY and scored on the identical test
    rows.  No model, no text: this is the paired denominator a classifier has to beat, and it is
    computed before any classifier score exists so it cannot be chosen afterwards."""
    import numpy as np
    from collections import Counter
    counts = Counter(train_labels)
    majority = counts.most_common(1)[0][0]
    classes = sorted(counts)
    prior = np.array([counts[c] for c in classes], dtype=float)
    prior /= prior.sum()
    truth = list(test_labels)
    n = len(truth)

    def scores(pred):
        correct = sum(1 for a, b in zip(pred, truth) if a == b)
        per_class = {}
        for c in classes:
            tp = sum(1 for a, b in zip(pred, truth) if a == c and b == c)
            fp = sum(1 for a, b in zip(pred, truth) if a == c and b != c)
            fn = sum(1 for a, b in zip(pred, truth) if a != c and b == c)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[str(c)] = {'precision': precision, 'recall': recall, 'f1': f1,
                                 'support': tp + fn}
        macro_f1 = sum(v['f1'] for v in per_class.values()) / len(classes)
        support = sum(v['support'] for v in per_class.values())
        weighted_f1 = (sum(v['f1'] * v['support'] for v in per_class.values()) / support
                       if support else 0.0)
        return {'accuracy': correct / n, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
                'per_class': per_class}

    rng = np.random.default_rng(seed)
    strat = scores([classes[i] for i in rng.choice(len(classes), size=n, p=prior)])
    # The stratified draw is one sample; its expectation is analytic, so both are reported.
    test_counts = Counter(truth)
    expected_accuracy = sum(prior[classes.index(c)] * test_counts[c] for c in classes) / n
    return {'test_rows': n, 'classes': [str(c) for c in classes],
            'train_class_counts': {str(c): counts[c] for c in classes},
            'test_class_counts': {str(c): test_counts[c] for c in classes},
            'majority': {'predicted_class': str(majority), **scores([majority] * n)},
            'stratified_one_draw_seed42': strat,
            'stratified_expected_accuracy_analytic': expected_accuracy}


def naive():
    import pandas as pd
    doc = {'schema': 'classification_naive_baselines.v1',
           'computed_before_any_model_score': True,
           'rule': 'fit on TRAIN labels only, scored on the identical governed test rows',
           'tasks': {}}
    train, _ = delivered('fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee')
    test, _ = delivered('71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb')
    tr, te = pd.read_parquet(train), pd.read_parquet(test)
    names = ['World', 'Sports', 'Business', 'Sci/Tech']
    doc['tasks']['agnews_test_full'] = _naive([names[i] for i in tr['label']],
                                             [names[i] for i in te['label']])
    doc['tasks']['agnews_test_first400_laya_published'] = _naive(
        [names[i] for i in tr['label']], [names[i] for i in te['label'][:LAYA_AGNEWS_N]])
    train, _ = delivered('3c9ec066b7bbdedc60d553b48e74ae6ca36715b5de2f9000a82e76e909bd76b7')
    test, _ = delivered('c4b6a660a3cd67f940f59b1b77fc4d2f1b99e56c94eaf54b9298a37647ecfbac')
    tr, te = pd.read_csv(train), pd.read_csv(test)
    doc['tasks']['fomc_combined_s_seed944601'] = _naive(list(tr['label']), list(te['label']))
    train, _ = delivered('b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b')
    test, _ = delivered('d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d')
    tr, te = pd.read_csv(train), pd.read_csv(test)
    doc['tasks']['banking77_test'] = _naive(list(tr['category']), list(te['category']))
    NAIVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    NAIVE_OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
    print(f'NAIVE BASELINES -> {NAIVE_OUT}')
    for task, v in doc['tasks'].items():
        print(f"  {task}: majority acc {v['majority']['accuracy']:.6f} "
              f"wF1 {v['majority']['weighted_f1']:.6f} macroF1 {v['majority']['macro_f1']:.6f} "
              f"| stratified expected acc {v['stratified_expected_accuracy_analytic']:.6f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['populations', 'naive'])
    globals()[ap.parse_args().action]()
