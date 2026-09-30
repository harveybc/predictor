"""Traffic's remaining horizons (h192, h336) priced WITHOUT a monotonicity assumption.

The dictamen `docs/audits/work_plan/MUSASHI_AUDIT_23B2EFA3_2026_09_29.md` is explicit: *"la horquilla
12-15 h sigue siendo proyeccion, no cota garantizada por dos horizontes"*.  The 12.08 - 15.02
GPU-hour bracket published in `SATOSHI_TRAFFIC_TRAIN_PILOT_2026_09_29.md` §4 prices h192 and h336 at
the measured rates of h96 and h720, and that brackets the truth only if the per-step cost is monotone
in the prediction length.  Two points cannot establish monotonicity.  So this module separates what
the existing evidence really settles from what it does not, and refuses to hand back the bracket
unless the caller asks for it by name and accepts the label.

**What is priced here, and why each class is what it says.**

* `DERIVED_EXACT_FROM_THE_LOADER` --- window and batch counts for every split at every horizon.  They
  come from the author's own split arithmetic over the registered row count, they need no GPU and no
  bytes, and the same arithmetic reproduces the sealed characterization's counts at **all four**
  horizons, including the two nobody ran.
* `DERIVED_EXACT_FROM_THE_DECLARATION` --- parameters, gradient bytes, Adam's slot bytes and the
  checkpoint's bytes.  `pred_len` enters the model in exactly one place: `models/TimeFilter.py:57`,
  `self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)`.  A `Linear` of that shape is
  affine in `pred_len` by construction, with slope `dim * num_patches + 1` per output step, so these
  quantities are *algebra*, not an extrapolation --- and the algebra reproduces both measured records
  to the byte.  That is a check of the derivation, not its justification.
* `UNPRICED_WITHOUT_A_RUN` --- the per-step and per-validation-batch **times**, and therefore every
  hour figure at h192 and h336.  A GPU's per-step cost is set by kernel selection and occupancy at
  the shape actually launched; those change in steps, not smoothly, so a rate at an unmeasured shape
  is not derivable from two other shapes.  It is `UNKNOWN`.  Never a number.

The consequence, stated plainly: **the twelve-cell total is not bounded by the evidence that exists.**
Six cells (three seeds at h96, three at h720) are priced from their own measured rates.  The other six
need two 30-step children that no allocation covers, and this module names the request instead of
inventing the number.
"""
from __future__ import annotations

import argparse
import json
import math

SCHEMA = "df_tsl_remaining_horizons.v1"

MEASURED = "MEASURED"
UNKNOWN = "UNKNOWN"
EXACT_LOADER = "DERIVED_EXACT_FROM_THE_LOADER"
EXACT_DECL = "DERIVED_EXACT_FROM_THE_DECLARATION"
UNPRICED = "UNPRICED_WITHOUT_A_RUN"

# The sealed recipe's own terms.  Nothing here is a choice made by this module.
DESIGN = {
    "dataset": "traffic",
    "rows": 17544,                 # the registered traffic.csv, hourly
    "seq_len": 96,
    "patch_len": 96,
    "d_model": 512,
    "batch_size": 16,
    "epochs": 30,
    "train_ratio": 0.7,
    "test_ratio": 0.2,
    "drop_last": False,            # data_provider/data_factory.py at the pinned commit, every flag
    "horizons": [96, 192, 336, 720],
    "seeds_per_horizon": 3,
    "head_declaration": "models/TimeFilter.py:57  self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)",
    "pred_len_occurrences_in_the_model": ("two: stored on the module, and as the head's out_features. "
                                          "The backbone is built from seq_len * n_vars // patch_len "
                                          "and never sees pred_len"),
}

# The two records this module is allowed to read numbers from, by digest.  Published in
# docs/audits/work_plan/SATOSHI_TRAFFIC_TRAIN_PILOT_2026_09_29.md at 91a4c410.
MEASURED_PILOTS = {
    96: {"record_sha256_prefix": "ea143639", "params": 7_306_748, "gradient_bytes": 29_164_928,
         "optimizer_slot_bytes": 58_330_040, "checkpoint_bytes": 29_248_117,
         "steady_step_seconds": 0.123025, "validation_seconds_per_batch": 0.058385,
         "checkpoint_write_seconds": 0.0248, "cgroup_peak_bytes": 1_851_158_528,
         "device_allocated_peak_bytes": int(4.816 * (1 << 30)),
         "device_reserved_peak_bytes": int(5.365 * (1 << 30)),
         "child_wall_seconds": 15.41, "child_cpu_seconds": 15.20, "timed_updates": 32},
    720: {"record_sha256_prefix": "06966c3d", "params": 7_626_860, "gradient_bytes": 30_445_376,
          "optimizer_slot_bytes": 60_890_936, "checkpoint_bytes": 30_528_565,
          "steady_step_seconds": 0.153422, "validation_seconds_per_batch": 0.180086,
          "checkpoint_write_seconds": 0.0272, "cgroup_peak_bytes": 2_179_903_488,
          "device_allocated_peak_bytes": int(4.884 * (1 << 30)),
          "device_reserved_peak_bytes": int(5.514 * (1 << 30)),
          "child_wall_seconds": 22.62, "child_cpu_seconds": 22.27, "timed_updates": 32},
}


class HorizonRefusal(SystemExit):
    """A number that would need a run nobody authorized is not produced."""


# --- the loader's own arithmetic -------------------------------------------------------------------

def num_patches(seq_len: int = None, patch_len: int = None) -> int:
    seq_len = DESIGN["seq_len"] if seq_len is None else int(seq_len)
    patch_len = DESIGN["patch_len"] if patch_len is None else int(patch_len)
    return int((seq_len - patch_len) / patch_len + 1)


def n_batches(windows: int, batch_size: int, *, drop_last: bool = False) -> int:
    if int(windows) <= 0:
        return 0
    return (int(windows) // int(batch_size)) if drop_last else -(-int(windows) // int(batch_size))


def split_windows(pred_len: int, *, rows: int = None, seq_len: int = None,
                  train_ratio: float = None, test_ratio: float = None) -> dict:
    """The author's `Dataset_Custom` borders, as arithmetic over the registered row count.

    num_train = int(n * 0.7), num_test = int(n * 0.2), num_vali = n - num_train - num_test;
    border1s = [0, num_train - seq_len, n - num_test - seq_len]; border2s = [num_train,
    num_train + num_vali, n]; and a split's window count is border2 - border1 - seq_len - pred_len + 1.
    """
    n = DESIGN["rows"] if rows is None else int(rows)
    L = DESIGN["seq_len"] if seq_len is None else int(seq_len)
    tr = DESIGN["train_ratio"] if train_ratio is None else float(train_ratio)
    te = DESIGN["test_ratio"] if test_ratio is None else float(test_ratio)
    num_train, num_test = int(n * tr), int(n * te)
    num_vali = n - num_train - num_test
    b1 = [0, num_train - L, n - num_test - L]
    b2 = [num_train, num_train + num_vali, n]
    out = {}
    for i, flag in enumerate(("train", "vali", "test")):
        out[flag] = max(0, b2[i] - b1[i] - L - int(pred_len) + 1)
    return out


def split_geometry(pred_len: int, *, batch_size: int = None) -> dict:
    bs = DESIGN["batch_size"] if batch_size is None else int(batch_size)
    w = split_windows(pred_len)
    return {"pred_len": int(pred_len), "batch_size": bs, "class": EXACT_LOADER,
            "splits": {f: {"windows": w[f], "batches": n_batches(w[f], bs, drop_last=DESIGN["drop_last"]),
                           "final_batch_rows": (w[f] % bs) or (bs if w[f] else 0)} for f in w},
            "basis": ("the author's own split borders over the registered row count and the sealed "
                      "batch size. No bytes are read and no device is touched")}


# --- the declaration's own algebra -----------------------------------------------------------------

def head_slope(*, d_model: int = None, patches: int = None) -> dict:
    """Parameters added per output step: a Linear(dim*num_patches, pred_len) contributes
    (dim*num_patches) weights and 1 bias per output step."""
    d = DESIGN["d_model"] if d_model is None else int(d_model)
    p = num_patches() if patches is None else int(patches)
    return {"slope_params_per_output_step": d * p + 1, "d_model": d, "num_patches": p,
            "declaration": DESIGN["head_declaration"], "class": EXACT_DECL}


def _affine_through_measured(field: str) -> dict:
    """Fit y = a + b*pred_len on the two measured records and REFUSE unless it reproduces both
    exactly.  For these fields affinity is the module's algebra, so an inexact fit means the
    derivation is wrong, not that the data is noisy."""
    h1, h2 = 96, 720
    y1, y2 = MEASURED_PILOTS[h1][field], MEASURED_PILOTS[h2][field]
    if (y2 - y1) % (h2 - h1) != 0:
        raise HorizonRefusal(f"REFUSED: {field} is not integral-affine in pred_len over the two "
                             f"measured records, so it may not be derived this way")
    b = (y2 - y1) // (h2 - h1)
    a = y1 - b * h1
    for h in (h1, h2):
        if a + b * h != MEASURED_PILOTS[h][field]:
            raise HorizonRefusal(f"REFUSED: the affine law for {field} does not reproduce the "
                                 f"measured record at h{h}")
    return {"intercept": a, "slope_per_output_step": b, "field": field, "class": EXACT_DECL,
            "checked_against": [f"h{h1}", f"h{h2}"]}


def static_terms(pred_len: int) -> dict:
    """Parameter-driven bytes at any horizon, from the head's declaration.

    The slope of the parameter count MUST equal the head's declared slope; if it does not, the
    derivation is refused rather than reported.
    """
    laws = {f: _affine_through_measured(f) for f in
            ("params", "gradient_bytes", "optimizer_slot_bytes", "checkpoint_bytes")}
    slope = head_slope()["slope_params_per_output_step"]
    if laws["params"]["slope_per_output_step"] != slope:
        raise HorizonRefusal(
            f"REFUSED: the measured parameter counts imply {laws['params']['slope_per_output_step']} "
            f"parameters per output step and the head declares {slope}. One of the two is wrong and "
            f"no horizon is priced from a law that does not match the code")
    out = {"pred_len": int(pred_len), "class": EXACT_DECL, "head_slope_params_per_output_step": slope,
           "laws": laws, "values": {}}
    for f, law in laws.items():
        out["values"][f] = int(law["intercept"] + law["slope_per_output_step"] * int(pred_len))
    out["values"]["parameter_bytes_float32"] = 4 * out["values"]["params"]
    out["measured_here"] = int(pred_len) in MEASURED_PILOTS
    return out


def head_activation_bytes(pred_len: int, *, batch_size: int = None, channels: int = 862,
                          tensors: int = 1) -> dict:
    """One head-output-shaped float32 tensor: batch x channels x pred_len x 4 bytes.

    Reported as a COMPONENT, never as a peak: a peak is what an allocator did, and an allocator's
    arena, fragmentation and workspace choices are not the sum of tensor algebra.
    """
    bs = DESIGN["batch_size"] if batch_size is None else int(batch_size)
    per = bs * int(channels) * int(pred_len) * 4
    return {"bytes_per_tensor": per, "tensors": int(tensors), "bytes": per * int(tensors),
            "class": "COMPONENT_OF_A_PEAK_NOT_A_PEAK",
            "reading": "an analytic tensor size. It does not predict the allocator's peak and is "
                       "never reported as one"}


# --- the time nobody measured ----------------------------------------------------------------------

def per_step_seconds(pred_len: int) -> dict:
    """The per-step time at one horizon.  Measured where it was measured; UNKNOWN where it was not."""
    h = int(pred_len)
    if h in MEASURED_PILOTS:
        p = MEASURED_PILOTS[h]
        return {"pred_len": h, "seconds": p["steady_step_seconds"], "status": MEASURED,
                "class": "MEASURED at this horizon", "record_sha256_prefix": p["record_sha256_prefix"]}
    return {"pred_len": h, "seconds": None, "status": UNKNOWN, "class": UNPRICED,
            "why": ("no pilot ran at this horizon. A GPU's per-step cost is set by the kernels chosen "
                    "for the shape actually launched and by occupancy at that shape; both change in "
                    "steps. Two other shapes do not determine it, and a monotone reading of two "
                    "points is an assumption, not a bound"),
            "what_would_settle_it": "one 30-step child at this horizon, on the admitted device"}


def cell_hours(pred_len: int) -> dict:
    """Hours for one cell (one seed) at one horizon.  A horizon with no measured rate has no hours."""
    g = split_geometry(pred_len)
    rate = per_step_seconds(pred_len)
    out = {"pred_len": int(pred_len), "epochs": DESIGN["epochs"],
           "train_batches": g["splits"]["train"]["batches"],
           "vali_batches": g["splits"]["vali"]["batches"],
           "test_batches_counts_only": g["splits"]["test"]["batches"],
           "per_step": rate}
    if rate["status"] != MEASURED:
        return {**out, "epoch_seconds": None, "cell_hours": None, "status": UNKNOWN, "class": UNPRICED,
                "why": "the per-step and per-validation-batch rates at this horizon are UNKNOWN, so "
                       "the hours are UNKNOWN. They are not filled in from another horizon"}
    p = MEASURED_PILOTS[int(pred_len)]
    epoch = (rate["seconds"] * out["train_batches"]
             + p["validation_seconds_per_batch"] * (out["vali_batches"] + out["test_batches_counts_only"])
             + p["checkpoint_write_seconds"])
    return {**out, "epoch_seconds": epoch, "cell_hours": epoch * DESIGN["epochs"] / 3600.0,
            "status": MEASURED, "class": "DERIVED from THIS horizon's measured rates and the loader's "
                                         "own batch counts",
            "includes_a_derived_term": ("the author's per-epoch logging pass over the test loader is "
                                        "priced from this horizon's measured validation rate and the "
                                        "sealed test batch count; its memory was never measured")}


def price(*, horizons=None) -> dict:
    """The whole picture: what the existing evidence settles, and what it does not."""
    hs = list(DESIGN["horizons"] if horizons is None else horizons)
    rows, priced_hours, unpriced = [], 0.0, []
    for h in hs:
        c = cell_hours(h)
        st = static_terms(h)
        rows.append({"pred_len": h, "seeds": DESIGN["seeds_per_horizon"], "geometry": split_geometry(h),
                     "static_terms": st["values"], "static_terms_class": st["class"],
                     "head_output_tensor": head_activation_bytes(h), "time": c})
        if c["status"] == MEASURED:
            priced_hours += c["cell_hours"] * DESIGN["seeds_per_horizon"]
        else:
            unpriced.append(h)
    out = {"schema": SCHEMA, "dataset": DESIGN["dataset"], "design": DESIGN,
           "measured_horizons": sorted(MEASURED_PILOTS), "horizons": rows,
           "priced_cells": (len(hs) - len(unpriced)) * DESIGN["seeds_per_horizon"],
           "priced_gpu_hours": priced_hours,
           "unpriced_horizons": unpriced,
           "unpriced_cells": len(unpriced) * DESIGN["seeds_per_horizon"],
           "twelve_cell_total": None,
           "twelve_cell_total_status": UNKNOWN,
           "twelve_cell_total_why": (
               "six of the twelve cells have no measured per-step rate. Pricing them from the other "
               "horizons' rates is exactly the monotonicity assumption the audit refused, so the "
               "total is UNKNOWN. What is established is the priced subtotal above and two holes"),
           "superseded_figure": {
               "value": "12.08 - 15.02 GPU-hours",
               "where": "SATOSHI_TRAFFIC_TRAIN_PILOT_2026_09_29.md section 4, at 91a4c410",
               "status": "WITHDRAWN AS A BOUND",
               "why": ("it prices h192 and h336 at the measured rates of h96 and h720, which brackets "
                       "the truth only if the per-step cost is monotone in the prediction length. Two "
                       "points do not establish that. It survives only as a conditional projection, "
                       "and only when a caller names the assumption")},
           "minimal_measurement_that_would_close_it": minimal_request()}
    return out


def conditional_bracket(*, assume_monotone_in_pred_len: bool = False) -> dict:
    """The old bracket, and it is handed back only to a caller who names the assumption."""
    if not assume_monotone_in_pred_len:
        raise HorizonRefusal(
            "REFUSED: a bracket over h192 and h336 requires assuming the per-step cost is monotone in "
            "the prediction length. The audit refused that assumption, so this function will not "
            "return a bracket unless the caller passes assume_monotone_in_pred_len=True and carries "
            "the label with the number")
    fastest = min(MEASURED_PILOTS, key=lambda k: MEASURED_PILOTS[k]["steady_step_seconds"])
    slowest = max(MEASURED_PILOTS, key=lambda k: MEASURED_PILOTS[k]["steady_step_seconds"])
    totals = {}
    for name, src in (("lower", fastest), ("upper", slowest)):
        s = 0.0
        for h in DESIGN["horizons"]:
            g = split_geometry(h)
            p = MEASURED_PILOTS[h] if h in MEASURED_PILOTS else MEASURED_PILOTS[src]
            epoch = (p["steady_step_seconds"] * g["splits"]["train"]["batches"]
                     + p["validation_seconds_per_batch"] * (g["splits"]["vali"]["batches"]
                                                            + g["splits"]["test"]["batches"])
                     + p["checkpoint_write_seconds"])
            s += epoch * DESIGN["epochs"] * DESIGN["seeds_per_horizon"]
        totals[name] = s / 3600.0
    return {"hours_lower": totals["lower"], "hours_upper": totals["upper"],
            "class": "CONDITIONAL_ON_AN_UNPROVEN_ASSUMPTION",
            "assumption": "the per-step cost is monotone in the prediction length",
            "status_if_the_assumption_fails": "the interval does not contain the truth and is not a bound",
            "not_a_bound": True}


def minimal_request() -> dict:
    """What would actually settle h192 and h336, priced from the two children that already ran."""
    walls = [MEASURED_PILOTS[h]["child_wall_seconds"] for h in MEASURED_PILOTS]
    cpus = [MEASURED_PILOTS[h]["child_cpu_seconds"] for h in MEASURED_PILOTS]
    peak = max(MEASURED_PILOTS[h]["cgroup_peak_bytes"] for h in MEASURED_PILOTS)
    return {
        "children": 2, "horizons": [192, 336], "updates_each": MEASURED_PILOTS[96]["timed_updates"],
        "wall_seconds_each_observed_range": [min(walls), max(walls)],
        "cpu_seconds_each_observed_range": [min(cpus), max(cpus)],
        "requested_wall_seconds_total": 600,
        "requested_cpu_seconds_total": 600,
        "declared_cap_bytes": 8 * (1 << 30),
        "cap_provenance": ("the 8 GiB already sealed at 8d461628 and observed ample: the worst "
                           "measured training peak over both pilots is %d B" % peak),
        "cap_rule": "asked once at the declared size; never re-asked smaller to pass an admission",
        "what_it_buys": ("a measured per-step and per-validation rate at h192 and h336, which removes "
                         "the monotonicity assumption from the twelve-cell figure"),
        "what_it_does_not_buy": ("no accuracy, no test split, no selection, no campaign. A 30-step "
                                 "child is a cost measurement"),
        "authority": ("NOT HELD. The Traffic authorization named h96 and h720. These two horizons have "
                      "no allocation, so they are not run and their hours stay UNKNOWN"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=None)
    p.add_argument("--conditional-bracket", action="store_true",
                   help="also print the withdrawn bracket, labelled as conditional")
    a = p.parse_args(argv)
    rec = price()
    if a.conditional_bracket:
        rec["conditional_bracket"] = conditional_bracket(assume_monotone_in_pred_len=True)
    text = json.dumps(rec, indent=1, default=str)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
