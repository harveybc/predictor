# FS4 runner integration test: px.rv5 x inner_2023 x three arms (worker_b, RTX 4090)

Runner: feature-extractor satoshi/fs4-task-runner-20261007 @deffa53f722ce27696489de73820d32e0f00ccc5
Test receipts are real COMPLETE terminals in the queue (plan 2b511808...), delivered through the controller
`complete` by the lease owner `worker_b-integration` (claim by --task-id, runner executed by hand once).

Invocation per arm (claim JSON is passed in FS4_CLAIM_JSON because crispdm-run gives its child /dev/null as stdin):

    env CUDA_VISIBLE_DEVICES=<empty for RAW/RANDOM | physical UUID for TRAINED_ENCODER> [CUDA_DEVICE_ORDER=PCI_BUS_ID FS4_GPU_UUID=<uuid>] \
        FS4_CLAIM_JSON='<claim>' FS4_RUNNER_PYTHON=<tensorflow python> FS4_LD_LIBRARY_PATH_FILE=<cu12 loader list> \
      crispdm-run -q -m 4437M -t 2h -n fs4-it-<ARM> -- <checkout>/tools/fs4_runner.sh \
        --input eurusd_ps1_batch_001=<ps1>/batch_001/features_train.parquet \
        --input eurusd_ps1_batch_002=<ps1>/batch_002/features_train.parquet \
        --input eurusd_ps1_batch_003=<ps1>/batch_003/features_train.parquet --output-root <durable dir>

Cap 4437M = 1.25 x the cost pilot's cgroup peak (3,722,018,816 B). Measured runner peaks (cgroup): RAW 0.47 GB,
RANDOM 1.25 GB, TRAINED 2.76 GB, all within the cap. Host name and home paths are redacted in the committed copies.
