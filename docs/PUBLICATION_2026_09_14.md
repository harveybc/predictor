# Default-branch publication, 2026-09-14

Purpose: make the research program, current proposal and usable components
discoverable from the repositories' default GitHub pages. This is a selective
publication, not a blanket merge of experimental branches or a service release.

## Scope by repository

| Repository | Default branch | Included in this publication |
|---|---|---|
| predictor | master | Current proposal PDF/LaTeX, alternative-proposal archive, corrected index, repository map and README; standalone application code unchanged |
| financial-data | master | File-lake adapter from `13e6b1f47`, including regression tests; corrected README and service documentation |
| preprocessor | master | Research scope and explicit link to the newer causal-transform implementation; packaged code unchanged |
| feature-eng | master | Onboarding and governed-run profile through `7118b83`; new README context; later integration work remains separate |
| feature-extractor | master | Onboarding and governed-run profile through `384ad8d`; documented integration limitations |
| doin-plugins | master | Clear separation of shipped domain plugins and research eligibility adapters |
| doin-domains | master | Public-facing project description replacing internal team terminology; versioned research links |
| agent-multi | master | Data-centric research context and explicit version boundary for campaign work |
| gym-fx | master | Simulator version, execution-accounting and comparison requirements |
| lts | main | Offline evaluation guidance and separate link to later paper-execution research |
| prediction_provider | main | Agent guide from `316c315`, preserving the newer main-branch README; clarification of serving and oracle controls |
| heuristic-strategy | master | Onboarding through `79977b2` and reproducible backtest guidance |
| synthetic-datagen | master | Onboarding through `0dd2552` and distinction between synthetic market data and known-noise calibration |
| timeseries-gan | master | Onboarding through `8f4e848`; historical status and successor made explicit |

`data-gov`, `doin-core` and `doin-node` already had their principal current
committed implementations on their default branches at inspection. This pass
does not claim that their uncommitted work or every historical side branch has
been incorporated. The other repositories outside the table were left unchanged.

Large campaign branches remain separate: they contain scientific records,
code under review and integration changes beyond a documentation refresh.
The README links identify concrete snapshots rather than promising that all
new research interfaces are installed by cloning the default branch.

## Verification

- Financial lake: **58 passed, 1 skipped** on the selected implementation.
  This includes the real threaded HTTP-server regression. The skip is retained,
  not counted as a pass.
- Feature-eng governed profile: **1 passed** with an explicit data-gov checkout.
- Feature-extractor governed profile: **1 passed** with the same explicit
  dependency. Without that checkout the tests skip; the skip is not integration
  evidence. These tests validate profile construction, not model performance.
- Predictor: CLI help succeeded; the documented two-epoch CPU demonstration
  completed with exit 0 after generating package entry-point metadata. The
  first attempt without metadata stopped at plugin discovery. No packages
  were upgraded, no GPUs were used, and no scientific conclusion is drawn.
- All manually changed README and new document links were checked against
  their publication trees. Git whitespace checks were also performed on the
  edited files. Eight pre-existing trailing spaces in the copied current LaTeX
  were preserved so its source hash remains identical; that source was excluded
  from the final whitespace-only check, not silently reformatted.
- Historical sample results overwritten by the isolated predictor smoke test
  were restored in that isolated clone; they are not part of the publication.
- No complete cross-repository suite or fresh-environment installation is
  claimed for this documentation pass.

## Proposal identity

The current proposal is copied from committed source `174cb886` without
editing its content. It is a 15-page PDF titled *Diseno de representaciones
temporales modulares mediante procesamiento diferenciado de variables*.

| Artifact | SHA-256 |
|---|---|
| Current PDF | `6545e62820d061e90b56aa7366da82bc8baec118c884b21d090eb9777baacf11` |
| Current LaTeX | `1290be6998e77421562df03d09294597e948f7e6893541547c2b61cd6c9042af` |

Earlier proposals remain identified as alternatives or historical formulations.
The archive's older RL-selection PDF is not presented as the current modular
proposal. Bibliographic papers by other authors were not republished in this pass.

## Integration discipline

All 14 selected updates were pushed to their actual default branches and
checked against GitHub: each README matches the selected local bytes, and the
current proposal PDF and LaTeX match the hashes above. Thirteen of these
repositories are public. `doin-domains` remains private and requires access;
its visibility was not changed. The linked `data-gov`, `doin-core` and
`doin-node` repositories were also checked as public.

All edits were made in isolated publication checkouts. Existing development
worktrees, deployed services, data stores and running experiments were not
modified or restarted. Updates to default branches use ordinary fast-forward
pushes, never force pushes. A concurrent remote change requires reconciliation
before publishing that repository.

The primary predictor checkout has an invalid local checkpoint reference that
prevents normal fetch. Its publication was prepared from an independent clone
of GitHub's master; no local references or development work were deleted to
work around that repository-maintenance issue.
