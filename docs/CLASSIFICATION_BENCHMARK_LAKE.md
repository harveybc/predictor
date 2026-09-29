# Registered classification benchmarks

Verified through the live governed route on 2026-09-29 04:37 UTC
(2026-09-28 local time). No scientific experiment was run and no model score
of ours exists on any of these corpora.

Lake identifier: `sota_benchmarks` — the same lake that already holds the
Time-Series-Library benchmarks. No second registry was created.

| Resource | Rows | Columns | Bytes | Classes |
| --- | ---: | ---: | ---: | ---: |
| `agnews_zhang2015_train/train.parquet` | 120000 | 2 | 18585438 | 4 |
| `agnews_zhang2015_test/test.parquet` | 7600 | 2 | 1234829 | 4 |
| `fomc_tdw_shah2023_train/train.csv` | 1984 | 5 | 422592 | 3 |
| `fomc_tdw_shah2023_test/test.csv` | 496 | 5 | 103896 | 3 |
| `banking77_casanueva2020_train/train.csv` | 10003 | 2 | 839073 | 77 |
| `banking77_casanueva2020_test/test.csv` | 3080 | 2 | 239961 | 77 |

Each split is the distributor's official split, and the local files are
unchanged distributor bytes. Nothing was converted, re-encoded, re-split,
imputed or normalized as part of this registration. Each resource has its own
directory because the governed campaign key is derived from the resource's
parent directory name.

## Provenance, revisions and licences

| Corpus | Original citation | Distributor and pinned revision | Licence | Use class |
| --- | --- | --- | --- | --- |
| AG News | X. Zhang, J. Zhao and Y. LeCun, "Character-level Convolutional Networks for Text Classification," NIPS, 2015, arXiv:1509.01626 | `fancyzhx/ag_news` @ `eb185aade064a813bc0b7f42de02595523103ca4` | Undeclared by the distributor; the underlying AG corpus permits non-commercial research use only and forbids redistribution under a different name | `BENCHMARK/RESEARCH_ONLY_UNRESOLVED_LICENCE` |
| FOMC / Trillion Dollar Words | A. Shah, S. Paturi and S. Chava, "Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis," ACL, pp. 6664-6679, 2023, doi:10.18653/v1/2023.acl-long.368 | `gtfintechlab/fomc_communication` @ `6b0283f55f0005a6d38d49f271d795c21fccc1a3` | CC BY-NC 4.0 | `BENCHMARK/RESEARCH_ONLY_NONCOMMERCIAL` |
| BANKING77 | I. Casanueva, T. Temcinas, D. Gerz, M. Henderson and I. Vulic, "Efficient Intent Detection with Dual Sentence Encoders," NLP4ConvAI, pp. 38-45, 2020, arXiv:2003.04807 | `github.com/PolyAI-LDN/task-specific-datasets` @ `9d081458ff52e53cf7e848f414e6e9344e4e6696` | CC BY 4.0 | `BENCHMARK/PUBLIC` |

Byte identity was verified against the distributor's own object id as well as
against a retained sha256: the AG News parquets against the Hugging Face LFS
oid, which is the sha256 itself, and the four CSVs against their upstream git
blob ids.

```text
agnews train    fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee
agnews test     71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb
fomc   train    3c9ec066b7bbdedc60d553b48e74ae6ca36715b5de2f9000a82e76e909bd76b7
fomc   test     c4b6a660a3cd67f940f59b1b77fc4d2f1b99e56c94eaf54b9298a37647ecfbac
b77    train    b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b
b77    test     d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d
```

## Access, semantics and the research-only bar

Register a campaign naming the lake, the exact resource and a dataset role, then
request a whole-resource governed delivery with its campaign and unit ids.

These are retrospective corpora with no observed availability axis: delivery is
`AS_IS`, availability `UNDECLARED`, availability label `UNKNOWN`, and **every
date-ranged request is refused**. A point-in-time slice cannot be produced,
which is the structural reason research-only material here cannot become
live-trading data. They are not authorized as point-in-time or live financial
data by this registration, and no commercial or trading entitlement is created
by it. Two of the three corpora additionally carry `commercial_use: REFUSED` and
`trading_use: REFUSED` in the declared sheet.

The FOMC `year` column is data, not evidence of when a row became available. The
AG News and BANKING77 resources have no time column at all. Because the deployed
resource-contract validator still requires non-empty time fields, these
resources carry explicit `NOT_A_TIME_SERIES_*` sentinels, which the untimed
delivery path never parses. That is a provider limitation, recorded rather than
concealed behind a column name that does not exist.

## Acceptance

All six resources returned `VERIFIED_TRANSFER`; the delivered file's sha256 was
re-hashed on disk and matched the retained digest; availability stayed
`UNDECLARED`/`UNKNOWN`; both units of every campaign were closed with terminals
through the outbox, so no campaign is left open; campaign reconciliation had no
missing units; and the warehouse content matched the emitted terminals. Refusal
checks held. Each route deliberately records one completed unit and one failed
probe: these are operational tests, not scientific training results.

The change was rehearsed first on a disposable data-gov, lake host and DuckDB
warehouse. The first rehearsal **failed usefully**: two splits under one corpus
directory collided on the governed campaign key, which is derived from the
resource's parent directory name, and the second campaign was refused with 409.
Production was untouched; the layout was corrected and the receipt is preserved
as `REHEARSAL.attempt1.json`.

Only the benchmark lake service restarted. The data-gov runtime config was
byte-identical before and after; data-gov and the warehouse were not restarted.
The three Time-Series-Library resources were re-hashed after the change and all
match their receipted digests; their contracts and the original `BUILD_RECEIPT`
were not rewritten. New provenance is in the additive
`CLASSIFICATION_EXTENSION_20260928.json` beside it.

Operator receipts are retained under
`~/.local/state/crispdm-data-foundation/classification-extension-20260928/`:
`INVENTORY.json`, `PLAN.json`, `host.before.json`, `host.candidate.json`,
`REHEARSAL.json`, `REHEARSAL.attempt1.json`, `REHEARSED_BINDING.json` and
`ADOPTION.json`.

## Pinned but not registered

Three artifacts are pinned by digest and deliberately not registered, because
the deployed provider delivers only `.csv` and `.parquet` and a converted file
would no longer be the distributor's bytes: BANKING77's ordered 77-label
vocabulary `categories.json` (git blob `cdd2a5c77a4079a455f8fb7e751d1ecee0e2a5a4`),
Financial PhraseBank v1.0 (a zip of latin-1 `.txt`, CC BY-NC-SA 3.0) and MASSIVE
intent for en-US and es-ES (per-locale `.json.gz`, CC BY 4.0). Recording them
here is what keeps a later run from quietly substituting something else.
