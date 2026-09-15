# The three machines obtain data through data-gov, 2026-09-15

Musashi's instruction: check the identity contract that exists, configure the workers with it,
prove a governed delivery from each machine.

## The identity contract, as deployed

`access_plugins/default_access.py` with `principals` and `policies` in the runtime
configuration. A principal is `sha256("<salt>:<secret>")` stored as `api_key_hash`; a policy
grants `principal x lake x verbs`. Live today: people `harvey`, `musashi`; services
`predictor`, `doin`, `heuristic-strategy`. Three policies are `principal: "*"` over
`financial_files`, `predictor_examples` and `olap_cube`; one is `predictor -> governance_smoke
-> download`.

The lake refuses anything that does not come through this path: a direct
`GET /api/v2/download` against the lake answers **401 `unauthenticated`**.

## What was done

The two workers reach the live services over a reverse tunnel opened from omega (the stores
bind `127.0.0.1`, and gamma has no ssh route back to omega). The `predictor` service key was
provisioned to each worker at `~/work/gov/predictor.key`, mode 0600.

`tools/worker_delivery_probe.py` drives data-gov's **own** `DataGovClient`: campaign,
`governed_download` with the campaign and unit headers, local verification, confirmation.

| host | campaign | bytes | delivery |
|---|---|---|---|
| omega | `cee1eafcea56…` | 228,801 | transferred |
| gamma | `33c90915f559…` | 228,801 | transferred |
| gamma (second run) | `0d989742b25e…` | 228,801 | **from cache** |
| dragon | `c4020a472f95…` | 228,801 | transferred |

Every delivery verified against `X-Content-SHA256`
(`4b60c34839e2dfb1f8a92f31cbe81c9d413041c559026481b94af3884acd9027`), and the bytes on disk
re-hashed to the same value.

**One download per group of experiments** is not a plan, it is the deployed behaviour: the
cache is content-addressed (`~/.cache/data-gov/<sha256>`), so gamma's second campaign
transferred nothing and confirmed against the same digest.

## Two refusals worth keeping

Both came from data-gov while wiring this, and both are the governance working:

* an invented campaign field is refused — `invalid campaign schema`, because the schema is
  exact-keyed, so a probe cannot quietly grow into its own protocol;
* a probe run from a directory that is not a git checkout is refused — `invalid code_identity`.
  A campaign must say which commit it ran from.

## What is still service-level rather than machine-level

These deliveries authenticate as the **`predictor` service principal**, so the receipts
attribute them to that service; the machine appears in the campaign key, the unit id and the
campaign body, not in the authenticated identity.

Per-machine principals (`satoshi@gamma`, `satoshi@dragon`) cannot be created without a
restart of `:5055`: `app/operator_config.py::editable_config` exposes `web_host`, `web_port`,
`max_downloads`, `lakes` and `policies` — **not** `principals` — and the runtime configuration
is read once at start. That is the operation that fails and its cause; it is not an approval
that is missing.
