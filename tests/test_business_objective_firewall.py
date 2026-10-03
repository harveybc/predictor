"""Acceptance and mutation tests for the external-metric firewall."""

from __future__ import annotations

import json
import unittest

from tools.business_objective_firewall import (
    Actor,
    BusinessObjectiveFirewall,
    FirewallError,
    Phase,
    ProcedureIdentity,
    WeekDisposition,
    WeekObservation,
)


EXPECTED_WEEKS = ("2025-W01", "2025-W02", "2025-W03")


def validation_payload() -> dict[str, object]:
    return {
        "candidate_id": "candidate-7",
        "validation_metrics": {"mae": 0.21, "naive_mae": 0.25},
        "hyperparameters": {"learning_rate": 0.001},
    }


def sealed_firewall() -> BusinessObjectiveFirewall:
    firewall = BusinessObjectiveFirewall.create(EXPECTED_WEEKS)
    firewall = firewall.record_selection_payload(Actor.CANDIDATE_WORKER, validation_payload())
    firewall = firewall.seal_procedure(
        selected_candidate_id="candidate-7",
        procedure={"cadence": "weekly", "update": "full_retrain"},
        validation_summary={"mae": 0.21, "naive_mae": 0.25},
    )
    return firewall


class BusinessObjectiveFirewallTests(unittest.TestCase):
    def test_selection_payload_accepts_validation_and_opaque_week_ids(self) -> None:
        firewall = BusinessObjectiveFirewall.create(EXPECTED_WEEKS)
        payload = validation_payload() | {"expected_test_week_ids": list(EXPECTED_WEEKS)}
        updated = firewall.record_selection_payload(Actor.SELECTOR, payload)

        self.assertEqual(updated.phase, Phase.VALIDATION_SELECTION)
        self.assertEqual(updated.opaque_test_week_ids(), EXPECTED_WEEKS)
        self.assertEqual(len(updated.selection_receipts), 1)

    def test_all_selection_actors_are_denied_test_material(self) -> None:
        mutations = {
            Actor.CANDIDATE_WORKER: {"x_test": [[1.0]]},
            Actor.SELECTOR: {"test_labels": [1]},
            Actor.OPTIMIZER_CALLBACK: {"test_metrics": {"mae": 0.1}},
            Actor.DOIN_PAYLOAD: {"test_arrays": {"targets": [1.0]}},
        }
        for actor, mutation in mutations.items():
            with self.subTest(actor=actor):
                firewall = BusinessObjectiveFirewall.create(EXPECTED_WEEKS)
                with self.assertRaisesRegex(FirewallError, "external test material"):
                    firewall.record_selection_payload(actor, validation_payload() | mutation)

    def test_holdout_alias_is_denied_but_latest_checkpoint_is_not_test_data(self) -> None:
        firewall = BusinessObjectiveFirewall.create(EXPECTED_WEEKS)
        with self.assertRaisesRegex(FirewallError, "external test material"):
            firewall.record_selection_payload(
                Actor.SELECTOR,
                {"holdout_labels": [0, 1]},
            )

        updated = firewall.record_selection_payload(
            Actor.SELECTOR,
            {"latest_checkpoint": "checkpoint-7"},
        )
        self.assertEqual(len(updated.selection_receipts), 1)

    def test_callback_requesting_test_metrics_is_rejected(self) -> None:
        firewall = BusinessObjectiveFirewall.create(EXPECTED_WEEKS)
        with self.assertRaises(FirewallError):
            firewall.record_selection_payload(
                Actor.OPTIMIZER_CALLBACK,
                {"request": {"test_metrics": True}},
            )

    def test_test_label_mutation_during_validation_cannot_change_identity(self) -> None:
        labels_a = [0, 1, 0]
        labels_b = [1, 0, 1]

        first = sealed_firewall()
        labels_a[:] = labels_b
        second = sealed_firewall()

        self.assertEqual(first.procedure_identity, second.procedure_identity)
        self.assertNotIn("test_labels", json.dumps(first.to_dict()).lower())

    def test_identity_is_sealed_before_test_can_open(self) -> None:
        firewall = BusinessObjectiveFirewall.create(EXPECTED_WEEKS)
        with self.assertRaisesRegex(FirewallError, "sealed procedure"):
            firewall.open_test_traversal()

        opened = sealed_firewall().open_test_traversal()
        self.assertEqual(opened.phase, Phase.TEST_TRAVERSAL)
        self.assertIsInstance(opened.procedure_identity, ProcedureIdentity)

    def test_ordered_traversal_requires_terminal_disposition_for_every_week(self) -> None:
        firewall = sealed_firewall().open_test_traversal()
        with self.assertRaisesRegex(FirewallError, "expected 2025-W01"):
            firewall.observe_week(
                WeekObservation.scored("2025-W02", {"mae": 0.2})
            )

        firewall = firewall.observe_week(
            WeekObservation.scored("2025-W01", {"mae": 0.3})
        )
        with self.assertRaisesRegex(FirewallError, "missing terminal dispositions"):
            firewall.close_test_traversal()

    def test_silently_omitted_week_prevents_metric_release(self) -> None:
        firewall = sealed_firewall().open_test_traversal()
        firewall = firewall.observe_week(
            WeekObservation.scored("2025-W01", {"mae": 0.3})
        )
        with self.assertRaises(FirewallError):
            firewall.observe_week(
                WeekObservation.scored("2025-W03", {"mae": 0.1})
            )
        with self.assertRaises(FirewallError):
            firewall.aggregated_test_metrics()

    def test_reselection_is_rejected_after_test_starts(self) -> None:
        firewall = sealed_firewall().open_test_traversal()
        firewall = firewall.observe_week(
            WeekObservation.scored("2025-W01", {"mae": 0.3})
        )
        with self.assertRaisesRegex(FirewallError, "selection is closed"):
            firewall.record_selection_payload(Actor.SELECTOR, validation_payload())
        with self.assertRaisesRegex(FirewallError, "already sealed"):
            firewall.seal_procedure("candidate-8", {}, {})

    def test_crash_resume_is_serializable_and_duplicate_safe(self) -> None:
        firewall = sealed_firewall().open_test_traversal()
        observation = WeekObservation.scored("2025-W01", {"mae": 0.3})
        firewall = firewall.observe_week(observation)

        resumed = BusinessObjectiveFirewall.from_json(firewall.to_json())
        replayed = resumed.observe_week(observation)
        self.assertEqual(replayed, resumed)
        self.assertEqual(len(replayed.test_observations), 1)

        conflicting = WeekObservation.scored("2025-W01", {"mae": 0.4})
        with self.assertRaisesRegex(FirewallError, "conflicting duplicate"):
            resumed.observe_week(conflicting)

    def test_metrics_release_only_after_complete_traversal(self) -> None:
        firewall = sealed_firewall().open_test_traversal()
        firewall = firewall.observe_week(
            WeekObservation.scored("2025-W01", {"mae": 0.3, "skill": 0.1})
        )
        firewall = firewall.observe_week(
            WeekObservation(
                week_id="2025-W02",
                disposition=WeekDisposition.FAILED,
                metrics=(),
                reason="provider timeout",
            )
        )
        firewall = firewall.observe_week(
            WeekObservation.scored("2025-W03", {"mae": 0.1, "skill": 0.3})
        )

        closed = firewall.close_test_traversal()
        self.assertEqual(closed.phase, Phase.CLOSED)
        self.assertEqual(
            closed.aggregated_test_metrics(),
            {"mae": 0.2, "skill": 0.2},
        )
        self.assertEqual(len(closed.test_observations), len(EXPECTED_WEEKS))

    def test_corrupt_serialized_state_is_rejected(self) -> None:
        state = sealed_firewall().open_test_traversal().to_dict()
        state["expected_test_week_ids"] = ["2025-W01", "2025-W01"]
        with self.assertRaises(FirewallError):
            BusinessObjectiveFirewall.from_dict(state)


if __name__ == "__main__":
    unittest.main()
