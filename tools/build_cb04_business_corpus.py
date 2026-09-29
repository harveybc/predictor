"""Build and seal the CB04 held-out business news corpus. Run once; then it is frozen.

Order of work matters here and is recorded, not asserted: this builder pins the
sha256 of `docs/contracts/classification_metrics.v1.json` and of
`app/classification_receipt.py` as they stood when the corpus was written. The
corpus was therefore assembled *after* the protocol existed, and the manifest
carries the evidence. A corpus written first and a protocol fitted to it
afterwards is a corpus that has already been tuned against.

What it tests, and why it is not a public benchmark
--------------------------------------------------
Public sentiment and intent benchmarks measure a task nobody in this programme
needs. Knowing that a headline reads positively does not say whether it bears on
a currency pair, whether it is new information or the third restatement of the
same release, or whether it matters before the next session. The three questions
here are the ones a news decision actually asks:

* `relevance`      — does this bear directly on the named asset's economics?
* `novelty`        — is this new information, a restatement, or a correction?
* `window`         — does it matter immediately, within the session, or not at all?

The material includes the cases that separate these from sentiment: items with a
clearly positive tone and no relevance, items that name the asset while being
about something else, restatement chains of one underlying release, and a
correction that reverses an earlier number.

Honesty about what this corpus is
---------------------------------
The items are written for this purpose. They are fabricated, in the register of
real macro and corporate news, and they name no real organisation, person,
publication or record: institutional actors appear only as generic roles ("the
euro-area central bank", "a national statistics office"). Nothing here may be
read as evidence about any real institution, and nothing here is market data.

The labels are the author's. That is a named limitation, not independence
achieved: a truly independent corpus needs third-party labels over licensed
feed text, which this programme does not hold an entitlement for. The refusal is
named in the manifest as
INDEPENDENT_THIRD_PARTY_LABELS_UNAVAILABLE_NO_FEED_ENTITLEMENT, and a named
refusal is not a completed family.

Held apart
----------
Items and labels are written to separate files and sealed separately.
`app.business_corpus` will hand over the items freely and the labels only against
an appended entry in the use ledger, so that a second, third and fourth scoring
against this same corpus is visible in the record instead of being quietly
described as untouched validation.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs/audits/evidence/cb04_business_corpus_20260928"

CORPUS_ID = "cb04_business_news.v1"
BUILT_ON = "2026-09-28"
BUILT_BY = "Satoshi, successor technical lead"

RELEVANCE = ("related", "unrelated", "unclear")
NOVELTY = ("new", "restatement", "correction")
WINDOW = ("immediate", "session", "none")

#: (id, asset, language, published_at, headline, body, relevance, novelty, window, why)
#: `why` is the construction note: what this item is in the corpus to separate.
ITEMS = [
    # -- squarely relevant macro, first publication ---------------------------
    ("b001", "EURUSD", "en", "2026-04-07T11:45:00Z",
     "Euro-area central bank raises its main refinancing rate by 25 basis points",
     "The euro-area central bank's governing council raised the main refinancing rate by 25 basis points, above the unchanged rate most surveyed economists had expected. The accompanying statement said further tightening would depend on services inflation.",
     "related", "new", "immediate",
     "canonical relevant macro surprise; a policy rate decision on one leg of the pair"),
    ("b002", "EURUSD", "en", "2026-04-07T12:05:00Z",
     "Rate decision: euro-area central bank moves to 25 basis point increase",
     "A second wire report of the same governing council decision repeats the 25 basis point increase and the reference to services inflation. No new figure is given.",
     "related", "restatement", "session",
     "restatement of b001 from a second wire; relevant but not new information"),
    ("b003", "EURUSD", "en", "2026-04-07T14:30:00Z",
     "Correction: euro-area rate increase was 50 basis points, not 25",
     "An earlier report of a 25 basis point increase was wrong. The governing council raised the main refinancing rate by 50 basis points. The earlier figure should be disregarded.",
     "related", "correction", "immediate",
     "a correction that reverses the magnitude of b001; a corpus that cannot see corrections cannot be used on news"),
    ("b004", "EURUSD", "en", "2026-05-12T08:30:00Z",
     "Northern reserve board holds its policy rate and drops the tightening bias",
     "The northern reserve board left its policy target unchanged and removed the sentence on further tightening that had appeared in its previous three statements.",
     "related", "new", "immediate",
     "relevant macro on the other leg; a removed sentence, not a numeric surprise"),
    ("b005", "EURUSD", "en", "2026-05-15T09:00:00Z",
     "Euro-area harmonised consumer prices rise 2.9 per cent year on year",
     "The statistics office's flash estimate put harmonised consumer price inflation at 2.9 per cent year on year, against 2.6 per cent expected. Core prices rose 3.1 per cent.",
     "related", "new", "immediate",
     "scheduled inflation release with a surprise against consensus"),
    ("b006", "EURUSD", "en", "2026-05-15T09:40:00Z",
     "Flash inflation estimate confirmed at 2.9 per cent in second release",
     "The statistics office republished its flash estimate without revision. Harmonised consumer prices rose 2.9 per cent year on year.",
     "related", "restatement", "none",
     "a confirmation of b005 with no new information and no remaining window"),
    ("b007", "EURUSD", "en", "2026-06-02T13:30:00Z",
     "Northern economy adds 310,000 jobs, well above expectations",
     "Non-farm payroll employment rose by 310,000, against 175,000 expected. The unemployment rate fell to 3.7 per cent and average hourly earnings rose 0.4 per cent on the month.",
     "related", "new", "immediate",
     "labour market surprise on the second leg"),
    ("b008", "EURUSD", "en", "2026-06-18T07:00:00Z",
     "Euro-area current account surplus narrows sharply in April",
     "The euro-area current account surplus narrowed to 12 billion from 31 billion, driven by a deterioration in the goods balance.",
     "related", "new", "session",
     "relevant flow data, slower window than a rate decision"),
    ("b009", "EURUSD", "en", "2026-06-25T16:00:00Z",
     "Euro-area finance ministers fail to agree a joint borrowing framework",
     "Finance ministers ended two days of talks without agreement on a joint borrowing framework. Two member states withheld consent; no new meeting was scheduled.",
     "related", "new", "session",
     "fiscal and political news with a genuine but slower currency channel"),
    ("b010", "EURUSD", "en", "2026-07-01T10:15:00Z",
     "Euro-area central bank announces an unscheduled governing council call",
     "The euro-area central bank said the governing council would hold an unscheduled call this afternoon. It gave no agenda.",
     "related", "new", "immediate",
     "an announcement with no content; relevance is high and the content is absent"),

    # -- clearly irrelevant, including positive-tone traps -------------------
    ("b011", "EURUSD", "en", "2026-04-09T09:00:00Z",
     "Regional airline reports its best quarterly load factor on record",
     "A regional airline said load factors reached 88 per cent, its best quarter on record, and raised its full-year guidance.",
     "unrelated", "new", "none",
     "strongly positive tone, no channel to the pair: sentiment is not relevance"),
    ("b012", "EURUSD", "en", "2026-04-11T15:20:00Z",
     "Two mid-cap software firms agree an all-share merger",
     "Two mid-cap software firms agreed an all-share merger valuing the smaller at a 22 per cent premium. Both boards recommended the deal.",
     "unrelated", "new", "none",
     "corporate news of a size that moves no currency"),
    ("b013", "EURUSD", "en", "2026-04-14T06:30:00Z",
     "Grocery chain recalls a batch of chilled desserts",
     "A grocery chain recalled one batch of chilled desserts after a labelling error. No illness was reported.",
     "unrelated", "new", "none",
     "negative tone, no economic content at all"),
    ("b014", "EURUSD", "en", "2026-04-16T12:00:00Z",
     "Euro strengthens against the dollar in quiet afternoon trade",
     "The euro traded 0.3 per cent firmer against the dollar in a quiet session with no scheduled releases. Volumes were below the twenty-day average.",
     "unrelated", "new", "none",
     "names the pair and reports its own price: a price report is not news about the pair's drivers"),
    ("b015", "EURUSD", "en", "2026-04-21T08:00:00Z",
     "Euro-area listed brewer opens a distribution centre",
     "A euro-area listed brewer opened a distribution centre employing 140 people, part of a previously announced capital plan.",
     "unrelated", "restatement", "none",
     "euro-area entity, no macro channel, and already announced"),
    ("b016", "EURUSD", "en", "2026-04-23T17:45:00Z",
     "Football club refinances its stadium debt",
     "A football club refinanced stadium debt at a lower coupon, extending maturities by seven years.",
     "unrelated", "new", "none",
     "financial vocabulary with no macro relevance; a keyword baseline should fail here"),
    ("b017", "EURUSD", "en", "2026-05-02T11:10:00Z",
     "Municipal transport authority raises single-ride fares",
     "A municipal transport authority raised single-ride fares by 10 cents, its first increase in four years.",
     "unrelated", "new", "none",
     "a price increase that is not an inflation statistic"),
    ("b018", "EURUSD", "en", "2026-05-06T13:00:00Z",
     "Weather service issues a heat advisory for the southern coast",
     "A national weather service issued a two-day heat advisory for the southern coast, with temperatures forecast above 38 degrees.",
     "unrelated", "new", "none",
     "out-of-domain item; the decision must be `unrelated`, not an abstention"),
    ("b019", "EURUSD", "en", "2026-05-08T10:20:00Z",
     "Dollar-denominated bond issue by a regional water utility is fully subscribed",
     "A regional water utility's dollar-denominated bond issue was fully subscribed. The proceeds refinance maturing debt.",
     "unrelated", "new", "none",
     "mentions the dollar in an instrument name only"),
    ("b020", "EURUSD", "en", "2026-05-11T09:30:00Z",
     "Retail chain's like-for-like sales fall 1 per cent in the quarter",
     "A retail chain reported a 1 per cent decline in like-for-like sales and held its guidance unchanged.",
     "unrelated", "new", "none",
     "single-company demand data, below any macro threshold"),

    # -- genuinely ambiguous: the abstention cases --------------------------
    ("b021", "EURUSD", "en", "2026-05-19T14:00:00Z",
     "Unnamed official says the exchange rate is being watched closely",
     "An official who asked not to be named said the exchange rate was being watched closely. The institution declined to comment.",
     "unclear", "new", "session",
     "unattributed and unquantified: the honest answer is an abstention, not a guess"),
    ("b022", "EURUSD", "en", "2026-05-22T16:30:00Z",
     "Survey of economists shows a wide split on the next policy move",
     "A survey of 41 economists found 20 expecting a further increase and 21 expecting no change. The previous survey was not directly comparable.",
     "unclear", "new", "none",
     "relevant subject, no directional content, and a non-comparable base"),
    ("b023", "EURUSD", "en", "2026-06-05T11:00:00Z",
     "Leaked draft of a growth forecast circulates ahead of publication",
     "A document described as a draft growth forecast circulated ahead of its scheduled publication. Its authenticity was not confirmed.",
     "unclear", "new", "immediate",
     "unverified provenance; relevance cannot be settled from the item"),
    ("b024", "EURUSD", "en", "2026-06-09T08:45:00Z",
     "Trade negotiations resume with no agenda published",
     "Trade negotiations between two blocs resumed. Neither side published an agenda or a timetable.",
     "unclear", "new", "none",
     "a channel exists in principle and nothing in the item establishes it"),
    ("b025", "EURUSD", "en", "2026-06-12T15:15:00Z",
     "Central bank speaker cancels a scheduled appearance",
     "A scheduled appearance by a central bank speaker was cancelled. No reason was given and no text was released.",
     "unclear", "new", "session",
     "absence of information about a relevant actor"),

    # -- relevant but slow, and relevant but already priced -----------------
    ("b026", "EURUSD", "en", "2026-06-16T09:00:00Z",
     "Euro-area bank lending survey shows a fourth quarter of tightening standards",
     "The quarterly bank lending survey showed credit standards tightening for a fourth consecutive quarter, in line with the previous reading.",
     "related", "new", "session",
     "relevant and new but continuing a known trend: window is session, not immediate"),
    ("b027", "EURUSD", "en", "2026-06-23T07:30:00Z",
     "Euro-area industrial production revised down by 0.1 percentage point",
     "Industrial production for April was revised to a 0.2 per cent monthly decline from 0.1 per cent.",
     "related", "correction", "none",
     "a revision of a second-tier series: a correction whose window has closed"),
    ("b028", "EURUSD", "en", "2026-06-30T13:00:00Z",
     "Northern reserve board publishes the minutes of its May meeting",
     "The minutes of the May meeting showed two participants favouring an increase. The decision and statement were published in May.",
     "related", "restatement", "session",
     "relevant, restates a known decision, adds dispersion detail"),
    ("b029", "EURUSD", "en", "2026-07-03T12:30:00Z",
     "Northern reserve board chair repeats that decisions will be taken meeting by meeting",
     "In prepared remarks the chair repeated that decisions would be taken meeting by meeting and that the board was not on a preset path.",
     "related", "restatement", "none",
     "relevant speaker, no new information: a correct answer must not reward the speaker's identity alone"),
    ("b030", "EURUSD", "en", "2026-07-08T09:00:00Z",
     "Euro-area unemployment rate unchanged at 6.4 per cent",
     "The euro-area unemployment rate was unchanged at 6.4 per cent, matching expectations and the previous month.",
     "related", "new", "session",
     "relevant release with no surprise; relevance does not require a surprise"),

    # -- second language, same questions ------------------------------------
    ("b031", "EURUSD", "es", "2026-07-10T09:00:00Z",
     "El banco central de la zona euro sube su tipo de referencia en 25 puntos basicos",
     "El consejo de gobierno del banco central de la zona euro subio el tipo de referencia en 25 puntos basicos. El comunicado condiciona nuevas subidas a la inflacion de servicios.",
     "related", "new", "immediate",
     "Spanish is one of the interface languages; the same decision must answer the same way"),
    ("b032", "EURUSD", "es", "2026-07-10T09:35:00Z",
     "Segunda agencia informa de la subida de 25 puntos basicos",
     "Una segunda agencia informa de la misma decision del consejo de gobierno, sin aportar cifras nuevas.",
     "related", "restatement", "session",
     "Spanish restatement of b031"),
    ("b033", "EURUSD", "es", "2026-07-14T08:00:00Z",
     "Una cadena de supermercados abre veinte tiendas nuevas",
     "Una cadena de supermercados anuncio la apertura de veinte tiendas y la contratacion de 400 personas.",
     "unrelated", "new", "none",
     "Spanish irrelevant item with positive tone"),
    ("b034", "EURUSD", "es", "2026-07-16T15:00:00Z",
     "Un funcionario no identificado dice que se vigila el tipo de cambio",
     "Un funcionario que pidio no ser identificado dijo que se vigila el tipo de cambio. La institucion no quiso comentar.",
     "unclear", "new", "session",
     "Spanish abstention case, parallel to b021"),
    ("b035", "EURUSD", "es", "2026-07-20T09:00:00Z",
     "La inflacion armonizada de la zona euro sube al 3,1 por ciento",
     "La oficina de estadistica situo la inflacion armonizada en el 3,1 por ciento anual, frente al 2,8 por ciento previsto.",
     "related", "new", "immediate",
     "Spanish relevant release with a surprise"),

    # -- near-duplicate chains and one deliberate trap ----------------------
    ("b036", "EURUSD", "en", "2026-07-23T09:00:00Z",
     "Euro-area services activity index falls to 49.8, below the expansion threshold",
     "The flash services activity index fell to 49.8 from 51.2, below the 50 threshold and below the 50.9 expected.",
     "related", "new", "immediate",
     "first publication of a survey that crosses a threshold"),
    ("b037", "EURUSD", "en", "2026-07-23T09:02:00Z",
     "Services index at 49.8 in flash reading",
     "The flash services activity index printed 49.8. The previous reading was 51.2.",
     "related", "restatement", "immediate",
     "a two-minute-later duplicate: novelty must separate what relevance cannot"),
    ("b038", "EURUSD", "en", "2026-07-23T11:00:00Z",
     "Final services index confirms 49.8 with no revision",
     "The final services activity index confirmed the flash reading of 49.8 with no revision.",
     "related", "restatement", "none",
     "third member of the same chain, window now closed"),
    ("b039", "EURUSD", "en", "2026-07-27T14:00:00Z",
     "Euro-area central bank corrects a figure in its published balance sheet table",
     "The euro-area central bank corrected a figure in a published balance sheet table, attributing the error to a reporting institution. The policy stance was unaffected.",
     "related", "correction", "none",
     "a correction by a relevant actor that carries no decision content"),
    ("b040", "EURUSD", "en", "2026-07-30T10:00:00Z",
     "Analyst note argues the euro is undervalued against the dollar",
     "An analyst note argued the euro was undervalued against the dollar on a purchasing power basis and set a twelve-month target.",
     "unrelated", "new", "none",
     "opinion about the pair, not news about the pair: the sharpest trap in the corpus"),

    # -- operational cases a live news path must handle ---------------------
    ("b041", "EURUSD", "en", "2026-08-03T09:00:00Z",
     "Statistics office postpones the release of second quarter national accounts",
     "The statistics office postponed the second quarter national accounts release by one week, citing a processing error.",
     "related", "new", "session",
     "news about the absence of a scheduled release"),
    ("b042", "EURUSD", "en", "2026-08-05T09:00:00Z",
     "Headline only: euro-area retail sales",
     "Euro-area retail sales.",
     "unclear", "new", "none",
     "a truncated item with no content; a classifier that answers confidently here is guessing"),
    ("b043", "EURUSD", "en", "2026-08-07T09:00:00Z",
     "Euro-area central bank and northern reserve board announce a standing swap line review",
     "The two central banks said they would review the terms of an existing standing swap line. The line remains in place unchanged during the review.",
     "related", "new", "session",
     "both legs, real channel, no immediate change"),
    ("b044", "EURUSD", "en", "2026-08-11T09:00:00Z",
     "Payroll figure republished after a transmission error, unchanged at 310,000",
     "A payroll figure republished after a transmission error was unchanged at 310,000. The original release stands.",
     "related", "restatement", "none",
     "looks like a correction and is a restatement: the two labels must not collapse"),
    ("b045", "EURUSD", "en", "2026-08-14T09:00:00Z",
     "Euro-area sovereign spreads widen after an unscheduled rating review is announced",
     "Sovereign spreads widened after a rating agency announced an unscheduled review of one member state. No rating action was taken.",
     "related", "new", "immediate",
     "market-observable relevance without a policy decision"),
]


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def jsonl(records) -> bytes:
    return ("".join(json.dumps(record, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True) + "\n" for record in records)).encode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="a frozen corpus is not rebuilt; this exists only to re-verify")
    arguments = parser.parse_args()
    out = Path(arguments.out)

    identifiers = [row[0] for row in ITEMS]
    assert len(set(identifiers)) == len(identifiers), "duplicate item id"
    for row in ITEMS:
        assert row[6] in RELEVANCE and row[7] in NOVELTY and row[8] in WINDOW, row[0]

    items = [{"item_id": row[0], "asset": row[1], "language": row[2],
              "published_at": row[3], "headline": row[4], "body": row[5],
              "synthetic": True,
              "questions": ["relevance", "novelty", "window"]}
             for row in ITEMS]
    labels = [{"item_id": row[0], "relevance": row[6], "novelty": row[7],
               "window": row[8], "construction_note": row[9]} for row in ITEMS]

    items_bytes, labels_bytes = jsonl(items), jsonl(labels)
    if out.exists() and not arguments.allow_overwrite:
        existing = out / "items.jsonl"
        if existing.exists():
            raise SystemExit(f"{out} already holds a frozen corpus; refusing to rebuild it")
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.jsonl").write_bytes(items_bytes)
    (out / "labels.jsonl").write_bytes(labels_bytes)

    def counts(field, vocabulary):
        return {value: sum(1 for row in labels if row[field] == value) for value in vocabulary}

    manifest = {
        "schema": "business_corpus_manifest.v1",
        "corpus_id": CORPUS_ID,
        "corpus_class": "BUSINESS_HELD_OUT",
        "built_on": BUILT_ON,
        "built_by": BUILT_BY,
        "built_after_the_protocol": {
            "claim": "the corpus was assembled after the metric contract was written, so it "
                     "could not be tuned against",
            "evidence": "the two digests below are the contract and its implementation as they "
                        "stood when this corpus was written; both precede this file in the "
                        "branch history",
            "contract_sha256": digest_file(ROOT / "docs/contracts/classification_metrics.v1.json"),
            "implementation_sha256": digest_file(ROOT / "app/classification_receipt.py"),
            "contract_commit": "02a34033 (contract and schema tests) and 0ab20c80 (implementation)",
        },
        "questions": {
            "relevance": {"vocabulary": list(RELEVANCE),
                          "asks": "does this item bear directly on the named asset's economics",
                          "counts": counts("relevance", RELEVANCE)},
            "novelty": {"vocabulary": list(NOVELTY),
                        "asks": "is this new information, a restatement of an already published "
                                "item, or a correction of one",
                        "counts": counts("novelty", NOVELTY)},
            "window": {"vocabulary": list(WINDOW),
                       "asks": "does it matter immediately, within the session, or not at all",
                       "counts": counts("window", WINDOW)},
        },
        "why_not_a_public_benchmark":
            "public sentiment and intent benchmarks measure a task this programme does not need. "
            "Tone does not decide relevance, and no public sentiment set asks whether an item is "
            "the third restatement of one release or a correction that reverses it.",
        "construction_procedure": [
            "1. The metric contract and its implementation were written and their tests passed.",
            "2. The three questions were chosen from what a news decision actually asks, not "
            "from what an available benchmark happens to label.",
            "3. Items were written by category, with the separating cases named per item in "
            "labels.jsonl -> construction_note: positive-tone irrelevance, items that name the "
            "asset while being about something else, restatement chains of one release, a "
            "correction reversing a magnitude, an item that looks like a correction and is a "
            "restatement, opinion about the pair rather than news about it, a truncated item, "
            "and out-of-domain items that must be answered `unrelated` rather than abstained.",
            "4. Items and labels were written to separate files and sealed separately.",
            "5. Both seals were recorded here. Nothing was scored against the corpus during "
            "construction: no model of any kind was run over it in this work package.",
        ],
        "rows": len(items),
        "items_file": "items.jsonl",
        "items_sha256": digest_bytes(items_bytes),
        "labels_file": "labels.jsonl",
        "labels_sha256": digest_bytes(labels_bytes),
        "use_ledger_file": "USE_LEDGER.jsonl",
        "licence": "written for this repository; no third-party text is included and no licence "
                   "is inherited",
        "what_it_is_not": [
            "not real news: every item is fabricated and marked synthetic true",
            "not about any real organisation, person, publication or record; institutional "
            "actors appear only as generic roles, and no real body is quoted or imitated",
            "not market data, and not usable as evidence about any real institution",
            "not independently labelled: the labels are the author's",
            "not large: 45 items is a screen, and a difference on 45 rows is not a ranking",
            "not a profitability claim, not a causal claim, and not deployment evidence",
        ],
        "named_refusals": {
            "INDEPENDENT_THIRD_PARTY_LABELS_UNAVAILABLE_NO_FEED_ENTITLEMENT":
                "a corpus independent in the full sense needs third-party labels over licensed "
                "feed text. This programme holds no such entitlement, so that corpus does not "
                "exist here. This one is held out and sealed, which is a different and weaker "
                "property. A named refusal is not a completed family.",
        },
        "authorises_broker_deployment": False,
        "execution_authority": "NONE",
        "holding_rule": "items may be read freely; labels are handed over only against an "
                        "appended use-ledger entry, so repeated scoring against this corpus is "
                        "visible in the record instead of being described as untouched "
                        "validation",
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    (out / "MANIFEST.json").write_bytes(manifest_bytes)
    ledger = out / "USE_LEDGER.jsonl"
    if not ledger.exists():
        ledger.write_bytes(b"")

    print(f"rows={len(items)} items_sha256={manifest['items_sha256']}")
    print(f"labels_sha256={manifest['labels_sha256']}")
    print(f"manifest_sha256={digest_bytes(manifest_bytes)}")
    for name, block in manifest["questions"].items():
        print(f"  {name}: {block['counts']}")


if __name__ == "__main__":
    main()
