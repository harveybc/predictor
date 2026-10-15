"""A pre-fit infrastructure refusal must not starve another matched arm."""

from tools.i6dmo.worker import _pick


class QueueView:
    """Read-only queue contract; no process or model is started."""

    def __init__(self, interruptions):
        self.interruptions = interruptions

    def units(self):
        return [dict(unit_id=arm, state="QUEUED", identity_sha256=arm,
                     kind="fit", device="gpu", spec={"depends_on": []})
                for arm in ("mg1.fit", "mg2m.fit")]

    def running(self):
        return []

    def refused_roles(self, unit_id):
        return ["preferred"]

    def attempts(self, unit_id=None):
        return [a for a in self.interruptions
                if unit_id is None or a["unit_id"] == unit_id]


SPEC = {"roles": {"preferred": {"gpus": [{"heavy": True}], "order": 0,
                                  "host_key": "preferred"}}}


def plan(mode="MATCHED_SHORT_CONTRAST"):
    return {"mode": mode, "units": [{"unit_id": "mg1.fit"}, {"unit_id": "mg2m.fit"}]}


def refusal(arm):
    return dict(unit_id=arm, outcome="INTERRUPTED", cause="INFRA_PREFIT:NO_GPU_VISIBLE@preferred")


def test_untried_arm_precedes_persistently_refused_arm():
    q = QueueView([refusal("mg1.fit")] * 3)
    assert _pick(q, plan(), SPEC, "preferred")["unit_id"] == "mg2m.fit"


def test_equal_refusal_counts_preserve_declared_preference():
    q = QueueView([refusal("mg1.fit"), refusal("mg2m.fit")])
    assert _pick(q, plan(), SPEC, "preferred")["unit_id"] == "mg1.fit"


def test_fairness_continues_after_other_arm_was_attempted():
    q = QueueView([refusal("mg1.fit")] * 4 + [refusal("mg2m.fit")])
    assert _pick(q, plan(), SPEC, "preferred")["unit_id"] == "mg2m.fit"


def test_legacy_campaign_dispatch_order_is_unchanged():
    q = QueueView([refusal("mg1.fit")] * 3)
    assert _pick(q, plan("LEGACY_BASELINE"), SPEC, "preferred")["unit_id"] == "mg1.fit"


def test_post_training_failures_are_not_refusal_rotation():
    q = QueueView([dict(unit_id="mg1.fit", outcome="FAILED", cause="TYPED:NONFINITE_LOSS")])
    assert _pick(q, plan(), SPEC, "preferred")["unit_id"] == "mg1.fit"
