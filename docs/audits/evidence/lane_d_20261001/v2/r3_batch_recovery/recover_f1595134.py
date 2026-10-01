"""Additive, inference-only recovery on the worker holding the original checkpoint.

Run through the existing admission wrapper (no queue/retry, 3914M, 20m).
Pass this source as a shell-quoted argument to python -B -c, NOT via stdin:
the admission wrapper does not forward stdin to the scoped child. Remote prefix:
$HOME/.local/bin/crispdm-run -m 3914M -t 20m -n m04-r3-score-recovery --
$HOME/anaconda3/envs/tensorflow/bin/python -B -c <shell-quoted-source>

No patch/deployment into the producing checkout: its existing CLI already accepts
--batch-size. Only a new recovery directory is written; the historical receipt,
model, queue and attempt states are not modified. Refuse a busy GPU or used output.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    home = Path.home()
    root = home / ".local/state/crispdm-data-foundation/m04_doin_20260930"
    campaign = root / "campaign_corrected_r0_v2"
    source = campaign / ("attempts/f15951344af39361/train-1/bridge/"
                         "b7967a827c5e16c7357a314d387aaf8b9f2d794a9dffe26aab65e5e35a54bd45")
    receipt_path = source / "accepted.json"
    request_path = source / "request.json"
    receipt = json.loads(receipt_path.read_text())
    request = json.loads(request_path.read_text())
    config = request["config"]
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    expected_cid = "f15951344af39361fb15ae639cb048bc341f9fc59050326a835dd5063b4ee4e4"
    if config_sha != expected_cid or receipt["digests"]["config_sha256"] != expected_cid:
        raise ValueError("Historical configuration identity mismatch")
    if config["evaluator"]["batch_size"] != 64:
        raise ValueError("Historical batch_size is not 64")
    revision = "a615c212c0a44e717f9707be6e62eb2cd4d5efae"
    checkout = home / "Documents/GitHub/.worktrees/predictor-d-pin-a615c212"
    actual_pin = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual_pin != revision or receipt["bridge"]["predictor_revision"] != revision:
        raise ValueError("Producing pin mismatch")
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain",
                                     "--untracked-files=no"], text=True)
    if dirty.strip():
        raise ValueError("Producing checkout has tracked modifications")
    validation = Path(request["validation_path"])
    model = Path(receipt["artifacts"]["best_model"])
    if digest_file(validation) != receipt["digests"]["validation_sha256"]:
        raise ValueError("Validation hash mismatch")
    if digest_file(model) != receipt["digests"]["model_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    device = receipt["environment"]["cuda_visible_devices"]
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                         "--format=csv,noheader"], text=True)
    if any(row.split(",", 1)[0].strip() == device for row in processes.splitlines()):
        raise RuntimeError("Producing GPU is busy; defer without launching scoring")
    destination = root / "recovery_f1595134_batch64_20261001"
    destination.mkdir(exist_ok=False)
    output = destination / "verification.json"
    watched = [receipt_path, request_path, model, validation]
    before = {str(path): digest_file(path) for path in watched}
    env = dict(os.environ)
    declaration = json.loads((campaign / "CAMPAIGN.json").read_text())
    env["LD_LIBRARY_PATH"] = declaration["executor"]["extra_env"]["LD_LIBRARY_PATH"]
    env.update(CUDA_VISIBLE_DEVICES=device, TF_DETERMINISTIC_OPS="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="1", M04_HOST_ROLE="worker_a",
               CUDA_CACHE_MAXSIZE="2147483648", PYTHONPATH=str(checkout))
    command = [receipt["bridge"]["predictor_python"], "-B", "-m", "tools.modular_checkpoint_scorer",
               "--receipt", str(receipt_path), "--validation", str(validation),
               "--output", str(output), "--batch-size", "64"]
    started = time.monotonic()
    with (destination / "scorer.log").open("x") as log:
        completed = subprocess.run(command, cwd=checkout, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=1100)
    after = {str(path): digest_file(path) for path in watched}
    result = json.loads(output.read_text()) if output.exists() else {}
    exact = (completed.returncode == 0 and result.get("verdict") == "VERIFIED"
             and result.get("exact_match") is True and before == after
             and result.get("receipt_sha256") == before[str(receipt_path)])
    recovery = {"schema": "m04.r3.additive_recovery.v1", "cid": expected_cid,
                "producing_revision": actual_pin, "config_sha256": config_sha,
                "batch_size": 64, "source": "original request.config, authenticated against receipt digest",
                "training_invoked": False, "queue_mutated": False, "original_bytes_unchanged": before == after,
                "input_hashes": before, "exit_code": completed.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "verification_sha256": digest_file(output) if output.exists() else None,
                "exact_match": result.get("exact_match"), "objective": result.get("objective"),
                "status": "VERIFIED_ADDITIVE" if exact else "NOT_VERIFIED",
                "historical_failed_attempts_preserved": True}
    with (destination / "RECOVERY.json").open("x") as stream:
        stream.write(json.dumps(recovery, indent=2) + "\n")
    print(json.dumps(recovery), flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
