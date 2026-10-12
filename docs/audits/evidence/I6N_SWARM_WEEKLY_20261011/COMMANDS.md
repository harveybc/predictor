# I6-N swarm weekly: commands (no secrets)

Start (owner action required; idempotent; run on the coordinator from a checkout of
`satoshi/i6n-deploy-20261012` @dfa54cb1):

    python -m tools.i6n_swarm_deploy --spec /var/tmp/i6n-swarm-deploy-20261012/DEPLOY.prod.private.json

It runs preflight, installs/starts the coordinator units, re-verifies bytes and
installs/starts the secondary (4090 host) workers, installs the preferred (5090 host)
workers and the coordinator timer `i6n-swarm-i6nw1-join-preferred.timer`, which starts
the preferred only after the running S3 pilot unit is inactive and its result sha256
is verified.

Status (read-only, any time):

    python -m tools.i6n_swarm_campaign status --root /var/tmp/i6n-swarm-prod-20261012 --json
    python -m tools.i6n_swarm_deploy --spec /var/tmp/i6n-swarm-deploy-20261012/DEPLOY.prod.private.json --status

Cross-host byte check:

    python -m tools.i6n_swarm.crosshost_verify --pin 2ded0406

Archive deployment (owner action required): `docs/i6n/swarm/DEPLOY_STEPS_D.md` §0 (rebuild
the lake wheel without the stale build/ directory), §2-§6, then the smoke record.

Integrated suites: `docs/audits/evidence/I6N_SWARM_WEEKLY_20261011/integration/run_integrated_suites.sh`
(sibling repos on PYTHONPATH at the pins in ARTIFACTS.json).
