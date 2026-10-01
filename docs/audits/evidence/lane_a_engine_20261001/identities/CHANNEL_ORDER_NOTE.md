The first run of tools/lane_a_identities.py recorded two channel-order digests (json_compact,
newline_joined) that do not use M04's recipe. M04's builder (tools/modular_doin_ecl_npz.py:112)
hashes ",".join(names). Recomputed on worker_b from the same validation.npz under crispdm-run 1G:
321 names, sha256 84f8490d90c050e9558268d2b93754294aeb3f8315b614c21755262b8df444d9 — equal to the
data manifest's channel_order_sha256. The tool now records that recipe as comma_joined_m04_recipe.
