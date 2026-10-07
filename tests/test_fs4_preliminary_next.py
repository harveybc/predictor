from tools.fs4_preliminary_next import choose_task


MANIFEST = {"schema": "fs3.preliminary_gpu_triage_bundle.v1", "final_selection": False,
            "populations": [{"population_id": "EURUSD", "gpu_feature_ids": ["chosen"]},
                            {"population_id": "ETH", "gpu_feature_ids": ["eth_chosen"]}]}


def test_preliminary_gpu_task_never_claims_deferred_feature():
    tasks = [{"task_id": "a", "population_id": "EURUSD", "feature_id": "deferred",
              "arm": "TRAINED_ENCODER", "fold_id": "fold-1", "state": "PENDING"},
             {"task_id": "b", "population_id": "EURUSD", "feature_id": "chosen",
              "arm": "TRAINED_ENCODER", "fold_id": "fold-1", "state": "PENDING"}]
    assert choose_task(tasks, MANIFEST) == "b"


def test_preliminary_gpu_task_prefers_business_population():
    tasks = [{"task_id": "a", "population_id": "ETH", "feature_id": "eth_chosen",
              "arm": "TRAINED_ENCODER", "fold_id": "fold-1", "state": "PENDING"},
             {"task_id": "b", "population_id": "EURUSD", "feature_id": "chosen",
              "arm": "TRAINED_ENCODER", "fold_id": "fold-1", "state": "PENDING"}]
    assert choose_task(tasks, MANIFEST) == "b"


def test_cpu_arms_use_the_same_shortlist():
    tasks = [{"task_id": "a", "population_id": "EURUSD", "feature_id": "deferred",
              "arm": "RAW", "fold_id": "fold-1", "state": "PENDING"},
             {"task_id": "b", "population_id": "EURUSD", "feature_id": "chosen",
              "arm": "RAW", "fold_id": "fold-1", "state": "PENDING"}]
    assert choose_task(tasks, MANIFEST, "RAW") == "b"
    assert choose_task(tasks, MANIFEST, "RANDOM_ENCODER") is None


def test_preliminary_gpu_task_refuses_invalid_manifest():
    bad = {**MANIFEST, "final_selection": True}
    try:
        choose_task([], bad)
    except ValueError as error:
        assert "INVALID_PRELIMINARY_MANIFEST" in str(error)
    else:
        raise AssertionError("invalid manifest accepted")
