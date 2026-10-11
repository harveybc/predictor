"""Swarm evaluation bridge: one candidate -> one admitted, ledger-fenced evaluator child per fold.

Called by the shared DOIN wrapper (doin_plugins.predictor.optimizer in
``i6n_swarm`` mode) from the node's executor thread. For each plan fold it:

1. derives the plan-independent ``fit_id`` and registers the unit;
2. claims it in the ledger: a verified terminal is ADOPTED (no fit), a held
   lease is waited on (never a concurrent duplicate), a refusal is final;
3. runs A's evaluator (``evaluate`` on stdin) inside crispdm-run with the
   fencing token and attempt lineage, renewing the lease while it runs;
4. verifies the receipt belongs to THIS candidate/fold/config before the
   ledger accepts it.

Fitness is the unweighted mean over all plan folds of the per-fold
``tools.i6n_search._score`` value, computed by lane E's
``tools.i6n_search_plan.score_folds`` when it is installed; any missing fold
means no fitness. Nothing here imports TensorFlow.
"""
import json
import math
import os
from pathlib import Path
import threading
import time

from tools import i6n_search as search
from tools.i6n_swarm import common, ledger as L, runner as R

CONTEXT_SCHEMA = "i6n.swarm.context.v1"
WIRE_SCHEMAS = ("i6n-shared-deap-v1", "i6n-shared-deap-v2")


class WaitExceeded(RuntimeError):
    pass


def _reference_score_folds(plan, receipts, supports):
    """Same formula as lane E's score_folds; used only when E's module is absent."""
    folds = list(plan["folds"]["fold_ids"])
    if set(receipts) != set(folds) or set(supports) != set(folds):
        raise ValueError("no fitness before all plan folds are complete")
    per_fold = {f: search._score(receipts[f], search._support(supports[f]))[0] for f in folds}
    return dict(status="COMPLETE", fitness=math.fsum(per_fold.values()) / len(per_fold), per_fold=per_fold,
                aggregation="unweighted_mean_over_folds", aggregator="B_REFERENCE_PENDING_E_MERGE")


def resolve_score_folds():
    try:
        from tools.i6n_search_plan import score_folds
    except ImportError:
        return _reference_score_folds
    return score_folds


def fold_fit_id(swarm, fold, config):
    return common.fit_id(experiment=swarm["experiment"], fold=fold["fold"], effective_config=config,
                         context=fold["context"], support=fold["support"])


def size_class(config, classes):
    widths = [config["core"]["width"], *config["core"]["reducer_filters"], config["core"].get("adapter_width") or 0]
    for item in classes:
        if (config["samples"] <= item["max_samples"] and max(widths) <= item["max_width"]
                and config["delta_x"] >= item.get("min_delta_x_hours", 1)):
            return item["class_id"]
    return None


def unit_spec(swarm, fold, wire, config, node=None, stage="search", dag_node=None):
    """Node-independent unit spec: every peer derives the same immutable spec for a fit_id."""
    cls = size_class(config, swarm.get("classes", [])) if swarm.get("classes") else None
    return dict(stage=stage, dag_node=dag_node or swarm.get("dag_node", f"search:h{swarm['horizon']}"),
                horizon=swarm["horizon"], fold=fold["fold"], experiment=swarm["experiment"],
                genotype_canonical_sha256=wire["identity"], effective_config_sha256=common.sha256_json(config),
                search_candidate_sha256=common.search_candidate_sha256(swarm.get("plan_sha256"), swarm["horizon"],
                                                                     wire["identity"]),
                device_class="gpu_admitted", fixture=bool(swarm.get("fixture")), size_class=cls,
                effective_config=config, genotype=wire.get("genotype", wire["genome"]),
                binding_sha256=fold.get("binding_sha256"))


def receipt_document(fit_id, swarm, fold, wire, config, receipt):
    """Verify an evaluator receipt belongs to this fit, then wrap it for the ledger."""
    search._score(receipt, search._support(fold["support"]))
    cr = receipt.get("metadata", {}).get("candidate_receipt")
    if not isinstance(cr, dict):
        raise L.ReceiptMismatch("metadata.candidate_receipt missing")
    expected = dict(schema_version="i6n.candidate_receipt.v1", experiment=swarm["experiment"], fold=fold["fold"],
                    effective_config_sha256=common.sha256_json(config), genotype_canonical_sha256=wire["identity"],
                    horizon_hours=swarm["horizon"], seed=common.SEED)
    for key, value in expected.items():
        if cr.get(key) != value:
            raise L.ReceiptMismatch(f"candidate_receipt.{key} does not belong to this fit")
    return dict(fit_id=fit_id, fold=fold["fold"], experiment=swarm["experiment"],
                effective_config_sha256=expected["effective_config_sha256"],
                genotype_canonical_sha256=wire["identity"], shared_receipt=receipt, candidate_receipt=cr)


def decode_wire(wire, generation):
    if not isinstance(wire, dict) or wire.get("schema") not in WIRE_SCHEMAS:
        raise ValueError("unsupported shared genome wire")
    if wire.get("generation") != generation:
        raise ValueError("wire generation mismatch")
    genome = search.normalize_genome(wire["genome"])
    if genome != wire["genome"] or search.effective_identity(genome) != wire["identity"]:
        raise ValueError("wire identity does not match its genome")
    return genome, search.effective_config(genome)


def locate_status(output_root, request):
    """Find A's STATUS.json for exactly this request identity (newest attempt directory)."""
    root = Path(output_root) / "candidates"
    best = None
    if not root.is_dir():
        return None
    for status in root.glob("*/STATUS.json"):
        try:
            body = json.loads(status.read_text())
        except (OSError, ValueError):
            continue
        ident = body.get("identity", {})
        if (ident.get("effective_config") == request["config"] and ident.get("support") == request["support"]
                and ident.get("context") == request["context"]):
            if best is None or status.stat().st_mtime > best.stat().st_mtime:
                best = status
    return best


class Bridge:
    def __init__(self, node, ledger, *, boot_id=None, clock=time.time, sleep=time.sleep, stop_event=None,
                 score_folds=None, plan=None, admit=None):
        self.node = node
        self.ledger = ledger
        self.boot_id = boot_id or common.boot_id()
        self.clock, self.sleep = clock, sleep
        self.stop_event = stop_event or threading.Event()
        self.score_folds = score_folds or resolve_score_folds()
        self.plan = plan
        self.admit = admit  # optional callable(unit_spec) -> {admit, cause, exclusive_host, cap_bytes}
        lease = node["lease"]
        self.ttl, self.renew_s = float(lease["ttl_seconds"]), float(lease["renew_seconds"])
        self.max_wait, self.poll = float(lease["max_wait_seconds"]), float(lease["poll_seconds"])

    # ---------------------------------------------------------------- folds
    def _env(self):
        return R.device_env(self.node["device"], os.environ)

    def _argv(self, fit_id, spec):
        child = list(self.node["evaluator_argv"])
        launcher = self.node.get("launcher")
        if launcher is None:
            if not self.node.get("fixture"):
                raise R.NoFallback("production evaluation requires the admitted launcher")
            return child
        caps = self.node.get("class_caps", {}).get(spec.get("size_class") or "")
        if caps is None:
            raise WaitExceeded(f"CLASS_NOT_ADMITTED: {spec.get('size_class')}")
        return R.build_argv(dict(launcher, memory_bytes=caps["child_memory_bytes"],
                                 wall_seconds=caps["child_wall_seconds"]), fit_id[:24], child)

    def run_fold(self, fit_id, spec, request, verify):
        """Claim/adopt/run one fold. Returns ("COMPLETE", doc) or (state, cause)."""
        deadline = self.clock() + self.max_wait
        exclusive = self.node.get("exclusive_host", True)
        while True:
            if self.admit is not None:
                decision = self.admit(spec)
                if not decision["admit"]:
                    if self.clock() >= deadline:
                        return "WAIT_RESOURCE", decision["cause"]
                    self.sleep(self.poll)
                    continue
                exclusive = decision.get("exclusive_host", exclusive)
            claim = self.ledger.claim(fit_id, self.node["owner"], self.node["host_key"], self.boot_id,
                                      self.node["device"]["cls"], self.ttl, exclusive_host=exclusive)
            if claim["result"] == "ADOPT":
                return "COMPLETE", claim["terminal"]["receipt"]
            if claim["result"] == "REFUSED":
                return "FAILED", claim["cause"]
            if claim["result"] in ("HELD", "WAIT_RESOURCE"):
                if self.clock() >= deadline or self.stop_event.is_set():
                    return claim["result"], claim.get("cause") or "LEASE_HELD_BY_PEER"
                self.sleep(self.poll)
                continue
            break
        token, attempt = claim["token"], claim["attempt"]
        payload = dict(request, attempt=dict(index=attempt["index"], parent_status_sha256=attempt["parent_status_sha256"],
                                             fencing_token=str(token)))
        if attempt["kind"] == "RESUME":
            payload["resume_checkpoint"] = attempt["checkpoint"]
        try:
            argv, env = self._argv(fit_id, spec), self._env()
        except Exception as exc:
            self.ledger.report(fit_id, token, "NOT_STARTED", f"{type(exc).__name__}: {exc}")
            raise
        caps = self.node.get("class_caps", {}).get(spec.get("size_class") or "", {})
        status_root = request["context"].get("output_root") if isinstance(request["context"], dict) else None

        def progress():
            if not status_root:
                return None
            status = locate_status(status_root, request)
            if status is None:
                return None
            body = json.loads(status.read_text())
            return {k: body.get(k) for k in ("state", "progress", "elapsed_seconds", "eta_seconds")}

        outcome = R.run_attempt(argv, env, json.dumps(payload, allow_nan=False),
                                wall_seconds=caps.get("child_wall_seconds", 0),
                                renew=lambda p=None: self.ledger.renew(fit_id, token, ttl=self.ttl, progress=p),
                                renew_interval=self.renew_s, ttl=self.ttl, stop_event=self.stop_event,
                                progress_reader=progress, cwd=self.node.get("workdir"),
                                log_dir=Path(self.node["logs_dir"]) / fit_id[:16], guard=True)
        status_sha = None
        if status_root:
            found = locate_status(status_root, request)
            status_sha = common.file_sha256(found) if found else None
        if outcome["kind"] == "OK":
            try:
                doc = verify(outcome["receipt"])
            except (ValueError, KeyError, TypeError) as exc:
                self.ledger.report(fit_id, token, "FAILED", f"RECEIPT_REJECTED: {exc}", parent_status_sha256=status_sha)
                return "FAILED", f"RECEIPT_REJECTED: {exc}"
            verdict = self.ledger.complete(fit_id, token, doc)
            if verdict == "DUPLICATE_DISCARDED":
                return "COMPLETE", self.ledger.terminal(fit_id)["receipt"]
            return "COMPLETE", doc
        self.ledger.report(fit_id, token, outcome["kind"], outcome["cause"], parent_status_sha256=status_sha)
        return ("INTERRUPTED" if outcome["kind"] == "INTERRUPTED" else
                "WAIT_RESOURCE" if outcome["kind"] == "NOT_STARTED" else "FAILED"), outcome["cause"]

    # ---------------------------------------------------------------- candidate
    def evaluate_wire(self, wire, generation, swarm):
        """Return the DOIN result for one shared genome, or raise ScoreRejected (no invented score)."""
        try:
            if swarm.get("schema") != CONTEXT_SCHEMA:
                raise ValueError("swarm context schema mismatch")
            genome, config = decode_wire(wire, generation)
            if genome["horizon"] != swarm["horizon"]:
                raise ValueError("genome horizon differs from the swarm domain")
            self._env()  # refuse a GPU node without its device before any claim
        except (ValueError, R.NoFallback) as exc:
            raise search.ScoreRejected(f"I6N swarm refused candidate: {exc}") from exc
        receipts, supports, causes = {}, {}, {}
        for fold in swarm["folds"]:
            fit = fold_fit_id(swarm, fold, config)
            spec = unit_spec(swarm, fold, wire, config, self.node)
            try:
                self.ledger.ensure_unit(fit, spec, budget=swarm.get("budget_new_candidates"))
            except L.BudgetExhausted as exc:
                raise search.ScoreRejected(str(exc)) from exc
            request = dict(config=config, generation=generation, context=fold["context"], support=fold["support"])
            state, value = self.run_fold(fit, spec, request,
                                         lambda r, fit=fit, fold=fold: receipt_document(fit, swarm, fold, wire, config, r))
            supports[fold["fold"]] = fold["support"]
            if state == "COMPLETE":
                receipts[fold["fold"]] = value["shared_receipt"]
            else:
                causes[fold["fold"]] = f"{state}: {value}"
        if causes:
            raise search.ScoreRejected(f"I6N folds not complete, no fitness: {causes}")
        plan = self.plan or dict(folds=dict(fold_ids=[f["fold"] for f in swarm["folds"]]),
                                 fitness=dict(aggregation="unweighted_mean_over_folds"))
        try:
            scored = self.score_folds(plan, receipts, supports)
        except Exception as exc:
            raise search.ScoreRejected(f"I6N fold aggregation refused: {exc}") from exc
        result = dict(fitness=scored["fitness"], metrics=dict(folds=receipts, per_fold=scored["per_fold"],
                                                              aggregation=scored.get("aggregation")),
                      hyper_dict=config, candidate_identity=wire["identity"], seed=common.SEED,
                      direction=common.DIRECTION)
        # Dashboard fields: pooled sums across folds; val_* means INNER, never outer/TEST.
        for split, prefix in (("train", "train"), ("inner", "val")):
            count = sum(r["counts"][split] for r in receipts.values())
            result[f"{prefix}_mae"] = math.fsum(r["error_sums"][split] for r in receipts.values()) / count
            result[f"{prefix}_naive_mae"] = math.fsum(r["naive_sums"][split] for r in receipts.values()) / count
        return result


def node_from_env():
    path = os.environ.get("I6N_SWARM_NODE_CONFIG")
    if not path:
        raise ValueError("I6N_SWARM_NODE_CONFIG is not set")
    return json.loads(Path(path).read_text())


def open_ledger(node):
    ledger = node["ledger"]
    if "path" in ledger:
        return L.Ledger(ledger["path"])
    from tools.i6n_swarm.lease_service import LedgerClient
    return LedgerClient(ledger["url"], ledger["token_file"])
