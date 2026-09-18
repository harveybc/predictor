"""Arithmetic/design tests only; not the production generator or ML evidence."""

import json
import math
import unittest


def reverse_six(i):
    return int(format(i, "06b")[::-1], 2)


def cases(split, n_phases=64):
    if split == "train":
        return [(14 + 4*j, 4*reverse_six(i)) for j in range(8) for i in range(n_phases)]
    if split == "validation":
        return [(15 + 4*j, 8*i + 2) for j in range(7) for i in range(32)]
    if split == "test":
        return [(17 + 4*j, 4*i + 3) for j in range(7) for i in range(64)]
    raise ValueError(split)


def identity(a, q, end):
    return a, tuple((256*t + q) % 2048 for t in range(end-16, end+1))


def identities(items):
    return {identity(a, q, end) for a, q in items for end in range(16, 24)}


def series(a, q):
    return [(a/28)*math.sin(math.pi*t/4 + math.pi*q/1024) for t in range(25)]


class DesignTests(unittest.TestCase):
    def test_counts_and_nesting(self):
        previous = set()
        for phases, expected in [(8, 512), (16, 1024), (32, 2048), (64, 4096)]:
            items = cases("train", phases)
            current = identities(items)
            self.assertEqual(len(current), expected)
            self.assertTrue(previous <= current)
            self.assertEqual(sorted(q for a, q in items if a == 14),
                             list(range(0, 256, 256//phases)))
            previous = current
        self.assertEqual(len(identities(cases("validation"))), 1792)
        self.assertEqual(len(identities(cases("test"))), 3584)

    def test_disjoint_and_interior_amplitudes(self):
        splits = [cases(s) for s in ("train", "validation", "test")]
        for i, items in enumerate(splits):
            for other in splits[i+1:]:
                self.assertFalse(identities(items) & identities(other))
                self.assertFalse({a for a, _ in items} & {a for a, _ in other})
                self.assertFalse({q for _, q in items} & {q for _, q in other})
        for items in splits[1:]:
            self.assertTrue(all(0.5 < a/28 < 1.5 for a, _ in items))

    def test_phase_equivalence_and_duplicates(self):
        self.assertEqual(identity(14, 0, 16), identity(14, 2048, 16))
        self.assertEqual(identities([(14, 0)]), identities([(14, 256)]))
        self.assertEqual(len(identities([(14, 0)]*512)), 8)

    def test_numeric_diversity_and_recurrence(self):
        global_windows = set()
        count = 0
        for split in ("train", "validation", "test"):
            for a, q in cases(split):
                x = series(a, q)
                for end in range(16, 24):
                    key = tuple(round(v, 11) for v in x[end-16:end+1])
                    self.assertNotIn(key, global_windows)
                    global_windows.add(key)
                    self.assertLessEqual(abs(x[end+1] - math.sqrt(2)*x[end] + x[end-1]), 1e-10)
                    count += 1
        self.assertEqual(count, 9472)
        self.assertEqual(sum(len(cases(s))*25 for s in ("train", "validation", "test")), 29600)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DesignTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({"status": "DESIGN_CHECKED_NOT_TRAINED",
                      "train_unique_windows": [512, 1024, 2048, 4096],
                      "validation_windows": 1792, "test_windows": 3584,
                      "raw_values": 29600, "neural_fits_planned": 12}, indent=2))
