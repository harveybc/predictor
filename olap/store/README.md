# predictor-olap-store

The predictor OLAP cube as an installable backend of a
[data-warehouse](https://github.com/harveybc/data-warehouse) host.

```bash
pip install "git+https://github.com/harveybc/predictor.git#subdirectory=olap/store"
```

It registers `predictor_olap` in the `datawarehouse.backends` entry-point group and owns
the domain: the star schema, the append-only `gov_*` tables and the reporting behaviour.
It owns no HTTP route, no console and no governance decision — those belong to the host and
to [data-gov](https://github.com/harveybc/data-gov).

Capabilities declared: `describe`, `storage`, `discover`, `query` (read-only SQL),
`write_metrics`, `write_terminal`, `terminal_digests`. There is no download capability, and
the host answers 422 for one: a warehouse delivers query results, not files.

## Provenance of `predictor_olap_store.query`

That module is the deployed query plugin **verbatim**. It is copied rather than moved so
the running service keeps loading exactly what it loads today. The package declares the
digest and the revision it came from (`SOURCE_SHA256`, `SOURCE_REVISION`, `SOURCE_PATH`),
and the parity test checks the copy against the declared digest, against the deployed file
whenever the service tree is present in the checkout, and against the blob in the declared
revision when that revision is in the clone. The copy disappears when the service tree is
integrated into the default branch.

## Tests

The suite lives in `olap/store/tests` and needs nothing but this package.
