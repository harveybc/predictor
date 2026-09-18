"""Arithmetic acceptance checks; no training or production data access."""

import json
import math


def check():
    period, window, horizon = 8, 17, 1
    assert (window - 1) / period == 2
    sizes = [4096, 512, 1024]
    start = 0
    partitions = []
    for size in sizes:
        first = start + window - 1
        last = first + size - 1
        end = last + horizon
        partitions.append({"decisions": [first, last], "support": [start, end]})
        start = end + 1
    assert partitions == [
        {"decisions": [16, 4111], "support": [0, 4112]},
        {"decisions": [4129, 4640], "support": [4113, 4641]},
        {"decisions": [4658, 5681], "support": [4642, 5682]},
    ]
    for previous, current in zip(partitions, partitions[1:]):
        assert previous["support"][1] < current["support"][0]
    for part, size in zip(partitions, sizes):
        first, last = part["decisions"]
        assert last - first + 1 == size
        assert first - window + 1 == part["support"][0]
        assert last + horizon == part["support"][1]
    n = sum(sizes) + 3 * (window + horizon - 1)
    assert n == start == 5683
    rf = 1 + 2 * sum([1, 2, 4, 8, 16, 32])
    parameters = 4 * 16 + 5 * (3 * 16 + 1) * 16 + 17
    assert rf == 127 and parameters == 4001
    values = [math.sin(2 * math.pi * t / period) for t in range(n)]
    recurrence_error = max(
        abs(values[t + 1] - (math.sqrt(2) * values[t] - values[t - 1]))
        for t in range(1, n - 1)
    )
    assert recurrence_error <= 1e-10
    first, last = partitions[2]["decisions"]
    persistence_mae = sum(abs(values[t + 1] - values[t])
                          for t in range(first, last + 1)) / sizes[2]
    assert math.isclose(persistence_mae, 0.5, abs_tol=1e-10)
    for p, w, h in [(8, 17, 1), (16, 33, 2), (32, 65, 4)]:
        assert (w - 1) / p == 2 and h / p == 0.125 and rf >= w
    return {"period": period, "window": window, "horizon": horizon,
            "raw_samples": n, "partitions": partitions,
            "receptive_field": rf, "parameters": parameters,
            "recurrence_max_error": recurrence_error,
            "persistence_test_mae": persistence_mae,
            "status": "ARITHMETIC_CHECKED_NOT_ML_VALIDATED"}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
