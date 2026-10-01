import importlib.util, pathlib, sys

spec = importlib.util.spec_from_file_location("m06w", pathlib.Path(__file__).resolve().parents[1] / "tools/m06_status_writer.py")
w = importlib.util.module_from_spec(spec)
sys.argv = ["m06_status_writer.py"]
spec.loader.exec_module(w)


def test_sampler_temperature_wins_over_probe():
    dv = {"sampler_temperature_c": 61.0, "temperature_c": 55.0, "gpu": {"vram_used_mib": 10.0, "vram_total_mib": 100.0}}
    v = w.gpu_vitals(dv, {"cap_bytes": 5, "cgroup_peak_bytes": 3, "stage": "fit"})
    assert v["temperature_c"] == 61.0 and v["temperature_source"].startswith("m06-gpu-sampler")
    assert v["vram_used_mib"] == 10.0 and v["job_cap_bytes"] == 5 and v["job_stage"] == "fit"


def test_probe_temperature_is_the_fallback_and_absent_job_is_none():
    v = w.gpu_vitals({"temperature_c": 55.0, "gpu": {}}, None)
    assert v["temperature_c"] == 55.0 and v["temperature_source"] == "writer probe"
    assert v["job_cap_bytes"] is None and v["job_eta"] is None


def test_missing_temperature_stays_missing():
    v = w.gpu_vitals({"gpu": {}}, None)
    assert v["temperature_c"] is None


def test_scrub_removes_addresses_and_host_names_but_not_versions():
    out = w.scrub({"argv": "ssh: cm-u@192.0.2.10 [mux]", "k": "box1 at 198.51.100.7", "v": "TF 2.21.0", "f": 0.851406}, ["box1"])
    assert out["argv"] == "ssh: cm-u@<ip> [mux]" and out["k"] == "<host> at <ip>"
    assert out["v"] == "TF 2.21.0" and out["f"] == 0.851406
