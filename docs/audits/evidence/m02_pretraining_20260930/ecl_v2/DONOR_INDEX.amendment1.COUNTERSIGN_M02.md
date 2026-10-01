# M02 countersign: DONOR_INDEX amendment 1

- Amendment: `DONOR_INDEX.amendment1.json`, sha256 2aaba33d3fa5a0b2ea3aabaa4529750d0063d755cf8da285454c131795b4bebe.
- Base index: 8a6bb20389bdb2b03ad0dbc2e39feb973efc05462f4919720624b2dbeb7025a4.
- Decision: **COUNTERSIGNED**.

## How it was checked

The check ran on worker_b under crispdm-run with a 512M cap, using only the standard library. Each of the 322 donors was compared with M02's own records:

- M02's `.keras`, `.manifest.json` and `.provenance.json` files are unchanged against the base index (322/322).
- Each alongside `.manifest.v2.json` has the sha256 the amendment states.
- Its embedded manifest, manifest sha, model sha and weights sha equal the schema-1 sidecar.
- `conditioning_contract` is OPERATIONAL and `learned_corpus.kind` is TRAIN_ONLY.
- Declaration ca1098ed, input manifest eca31ec1 and source config 4d5402a0 match both the amendment and M02's provenance.
- `pretrained_weights_source` is null.
- **Branches (321):**
  - data_sha256 = train input b430df32.
  - support names the train_validation input 981108e1.
  - Reconstruction MAE, MSE and rows (3645) are exactly the record's train_validation values.
  - The space is the model input (ECL z_train). M02 agrees with this space.
- **Core (1):**
  - data_sha256 = 788ffcfb and support = cc82b930. These are the fused materialization shas recorded in M02's core.provenance.json and PRETRAIN.json. The core autoencoder was trained on exactly those bytes.
  - FUSION.json binds them to inputs b430df32 (train) and 981108e1 (train_validation), so the TRAIN-only lineage holds.
  - Reconstruction equals PRETRAIN core.reconstruction.train_validation, in the fused branch-latent space. M02 agrees with this space.
- Every derivation source file has the sha256 the amendment cites.

## Scope

This countersign only confirms that the amendment follows from M02's records. It certifies nothing about the donors' forecasting utility. Reconstruction error is not forecasting skill.

Satoshi, successor technical lead, 2026-10-01
