#!/usr/bin/env python
"""
NEAT-style Optimizer Plugin (Parameters-as-Genes)

Implements a NeuroEvolution of Augmenting Topologies inspired approach
where hyperparameters are treated as genes that can be activated/deactivated.
Key features:
  - Variable-length genomes: each individual has a subset of active parameters
  - Speciation: groups individuals with similar parameter structures
  - Fitness sharing: within-species fitness adjustment to maintain diversity
  - Structural mutations: add/remove parameters organically
  - Value mutations: standard gaussian mutation on parameter values
  - Innovation numbers: global tracking for crossover alignment
  - Level-2 early stopping: patience-based convergence detection (same as default)

Uses the same candidate_worker subprocess for evaluation as default_optimizer.
"""

import copy
import random
import numpy as np
import time
import json
import gc
import os
import sys
import subprocess
import tempfile
import pty
import select
from pathlib import Path
from app.plugin_loader import load_plugin

# Reverse mapping: GA encodes activation as int [0..7], model needs string.
ACTIVATION_INDEX_TO_NAME = [
    "relu", "elu", "selu", "tanh", "sigmoid", "swish", "gelu", "leaky_relu",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(p):
    if not p:
        return None
    try:
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp)
        return str((_repo_root() / pp).resolve())
    except Exception:
        return str(p)


def _json_sanitize(obj):
    """Recursively convert numpy/tensor types into JSON-serializable Python types."""
    try:
        import tensorflow as tf
        if isinstance(obj, tf.Tensor):
            obj = obj.numpy()
    except Exception:
        pass
    if callable(obj) and not isinstance(obj, type):
        return None
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()
                if k != "optimization_callbacks" and k != "_non_serializable_keys"}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def _atomic_json_dump(path, payload):
    """Write JSON atomically to avoid corrupt files on crashes."""
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(_json_sanitize(payload), f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# ── NEAT Genome Representation ───────────────────────────────

class NeatGene:
    """A single gene representing one hyperparameter."""
    __slots__ = ("innovation", "param_name", "value")

    def __init__(self, innovation, param_name, value):
        self.innovation = innovation
        self.param_name = param_name
        self.value = value

    def copy(self):
        return NeatGene(self.innovation, self.param_name, self.value)


class NeatGenome:
    """Variable-length genome for NEAT-style optimization."""

    def __init__(self):
        self.genes = {}          # innovation_number -> NeatGene
        self.fitness = None      # Raw fitness (lower is better)
        self.adjusted_fitness = None  # After fitness sharing
        self.species_id = None
        # Evaluation metrics
        self.val_mae = None
        self.naive_mae = None
        self.train_mae = None
        self.train_naive_mae = None
        self.test_mae = None
        self.test_naive_mae = None
        self.model_summary = None
        self.hyper_dict = None

    def to_hyper_dict(self, param_types):
        """Convert genome to hyperparameter dict for evaluation."""
        hyper = {}
        for gene in self.genes.values():
            key = gene.param_name
            val = gene.value
            if key == "use_log1p_features":
                hyper[key] = ["typical_price"] if int(round(val)) == 1 else None
            elif key == "positional_encoding":
                hyper[key] = bool(int(round(val)))
            elif key == "activation":
                act_idx = max(0, min(int(round(val)), len(ACTIVATION_INDEX_TO_NAME) - 1))
                hyper[key] = ACTIVATION_INDEX_TO_NAME[act_idx]
            elif param_types.get(key) == "int":
                hyper[key] = int(round(val))
            else:
                hyper[key] = val
        return hyper

    @property
    def active_params(self):
        return sorted(gene.param_name for gene in self.genes.values())

    @property
    def complexity(self):
        return len(self.genes)

    def deep_copy(self):
        g = NeatGenome()
        g.genes = {k: v.copy() for k, v in self.genes.items()}
        g.species_id = self.species_id
        return g

    def to_serializable(self):
        """For checkpoint saving."""
        return {
            "genes": {str(k): {"innovation": v.innovation, "param_name": v.param_name, "value": v.value}
                      for k, v in self.genes.items()},
            "fitness": self.fitness,
            "species_id": self.species_id,
        }

    @classmethod
    def from_serializable(cls, data):
        """Restore from checkpoint."""
        g = cls()
        for k, v in data.get("genes", {}).items():
            inn = int(k)
            g.genes[inn] = NeatGene(inn, v["param_name"], v["value"])
        g.fitness = data.get("fitness")
        g.species_id = data.get("species_id")
        return g


# ── Innovation Tracking ──────────────────────────────────────

class InnovationTracker:
    """Global innovation number assignment for parameters."""

    def __init__(self):
        self._counter = 0
        self._param_to_innovation = {}

    def get_innovation(self, param_name):
        if param_name not in self._param_to_innovation:
            self._counter += 1
            self._param_to_innovation[param_name] = self._counter
        return self._param_to_innovation[param_name]

    def to_serializable(self):
        return {"counter": self._counter, "map": dict(self._param_to_innovation)}

    @classmethod
    def from_serializable(cls, data):
        t = cls()
        t._counter = data.get("counter", 0)
        t._param_to_innovation = data.get("map", {})
        return t


# ── Species ──────────────────────────────────────────────────

class Species:
    """A group of structurally similar genomes."""

    def __init__(self, species_id, representative):
        self.id = species_id
        self.representative = representative  # NeatGenome
        self.members = []
        self.best_fitness = float("inf")
        self.generations_without_improvement = 0

    @property
    def size(self):
        return len(self.members)


def compatibility_distance(g1, g2, full_bounds, c1=1.0, c3=0.4):
    """Compute NEAT compatibility distance between two genomes.
    c1: coefficient for structural difference (disjoint/excess genes)
    c3: coefficient for value difference in matching genes
    """
    inn1 = set(g1.genes.keys())
    inn2 = set(g2.genes.keys())
    matching = inn1 & inn2
    disjoint_excess = len((inn1 - inn2) | (inn2 - inn1))
    N = max(len(g1.genes), len(g2.genes), 1)

    # Normalized weight difference for matching genes
    if matching:
        diffs = []
        for i in matching:
            gene1 = g1.genes[i]
            low, high = full_bounds[gene1.param_name]
            range_val = (high - low) if high != low else 1
            diffs.append(abs(g1.genes[i].value - g2.genes[i].value) / range_val)
        w_diff = sum(diffs) / len(diffs)
    else:
        w_diff = 0

    return c1 * disjoint_excess / N + c3 * w_diff


def speciate(population, species_list, full_bounds, threshold):
    """Assign each genome to a species based on compatibility distance."""
    # Clear old members
    for sp in species_list:
        sp.members = []

    unassigned = list(population)
    for genome in unassigned:
        placed = False
        for sp in species_list:
            if compatibility_distance(genome, sp.representative, full_bounds) < threshold:
                sp.members.append(genome)
                genome.species_id = sp.id
                placed = True
                break
        if not placed:
            # Create new species
            new_id = max((s.id for s in species_list), default=0) + 1
            new_sp = Species(new_id, genome.deep_copy())
            new_sp.members.append(genome)
            genome.species_id = new_id
            species_list.append(new_sp)

    # Remove empty species
    species_list[:] = [s for s in species_list if s.members]

    # Update representatives (random member from each species)
    for sp in species_list:
        sp.representative = random.choice(sp.members).deep_copy()


def adjust_fitness(species_list):
    """Fitness sharing: adjusted_fitness = raw_fitness / species_size."""
    for sp in species_list:
        for genome in sp.members:
            if genome.fitness is not None and np.isfinite(genome.fitness):
                genome.adjusted_fitness = genome.fitness / max(sp.size, 1)
            else:
                genome.adjusted_fitness = float("inf")


# ── Mutation Operators ───────────────────────────────────────

def mutate_add_param(genome, all_params, full_bounds, innovation_tracker, add_prob=0.15):
    """Structural mutation: add a random new parameter."""
    if random.random() > add_prob:
        return False
    active = {g.param_name for g in genome.genes.values()}
    candidates = [p for p in all_params if p not in active]
    if not candidates:
        return False
    new_param = random.choice(candidates)
    inn = innovation_tracker.get_innovation(new_param)
    low, high = full_bounds[new_param]
    if isinstance(low, int) and isinstance(high, int):
        value = random.randint(low, high)
    else:
        value = random.uniform(low, high)
    genome.genes[inn] = NeatGene(inn, new_param, value)
    return True


def mutate_remove_param(genome, min_params=2, remove_prob=0.05):
    """Structural mutation: remove a random parameter."""
    if random.random() > remove_prob or len(genome.genes) <= min_params:
        return False
    remove_key = random.choice(list(genome.genes.keys()))
    del genome.genes[remove_key]
    return True


def mutate_values(genome, full_bounds, mutpb=0.2):
    """Gaussian mutation on parameter values."""
    mutated = False
    for gene in genome.genes.values():
        if random.random() < mutpb:
            low, high = full_bounds[gene.param_name]
            if isinstance(low, int) and isinstance(high, int):
                gene.value = random.randint(low, high)
            else:
                sigma = (high - low) * 0.1
                gene.value = max(low, min(high, gene.value + random.gauss(0, sigma)))
            mutated = True
    return mutated


def clamp_genome(genome, full_bounds):
    """Ensure all gene values are within bounds."""
    for gene in genome.genes.values():
        low, high = full_bounds[gene.param_name]
        gene.value = max(low, min(high, gene.value))


# ── Crossover ────────────────────────────────────────────────

def neat_crossover(parent1, parent2):
    """NEAT-style crossover. Fitter parent contributes disjoint/excess genes."""
    # Ensure parent1 is fitter (lower fitness = better)
    if parent2.fitness is not None and (parent1.fitness is None or parent2.fitness < parent1.fitness):
        parent1, parent2 = parent2, parent1

    child = NeatGenome()
    p1_inn = set(parent1.genes.keys())
    p2_inn = set(parent2.genes.keys())

    # Matching genes: randomly inherit from either parent
    for inn in p1_inn & p2_inn:
        gene = random.choice([parent1.genes[inn], parent2.genes[inn]]).copy()
        child.genes[inn] = gene

    # Disjoint/excess genes: inherit from fitter parent (parent1)
    for inn in p1_inn - p2_inn:
        child.genes[inn] = parent1.genes[inn].copy()

    return child


# ── Plugin Class ─────────────────────────────────────────────

class Plugin:
    """NEAT-style optimizer plugin with parameters-as-genes."""

    plugin_params = {
        "population_size": 20,
        "n_generations": 10,
        "mutpb": 0.2,
        "optimization_patience": 6,
        "hyperparameter_bounds": {
            "learning_rate": (1e-5, 1e-2),
            "num_layers": (1, 5),
            "layer_size": (16, 256),
        },
        # NEAT-specific defaults
        "neat_initial_params": None,       # List of initial params (None = first 2 from bounds)
        "neat_add_param_prob": 0.15,       # Probability of adding a parameter
        "neat_remove_param_prob": 0.05,    # Probability of removing a parameter
        "neat_compatibility_threshold": 2.0,  # Speciation distance threshold
        "neat_min_params": 2,              # Minimum active parameters per genome
        "neat_survival_rate": 0.5,         # Fraction of species that reproduces
        "neat_interspecies_mate_rate": 0.01,  # Cross-species mating probability
        "neat_elitism": 1,                 # Number of elites per species
    }
    plugin_debug_vars = ["population_size", "n_generations", "mutpb", "optimization_patience"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def optimize(self, predictor_plugin, preprocessor_plugin, config):
        """Run NEAT-style optimization.

        Same interface as default_optimizer: returns dict of best hyperparameters.
        """
        # ── Setup ────────────────────────────────────────────
        if "predictor_plugin" in config:
            config["plugin"] = config["predictor_plugin"]
        elif "plugin" not in config:
            config["plugin"] = "default_predictor"

        target_plugin_name = config.get("target_plugin", "default_target")
        try:
            target_class, _ = load_plugin("target.plugins", target_plugin_name)
            target_plugin = target_class()
            target_plugin.set_params(**config)
        except Exception as e:
            print(f"Failed to load Target Plugin: {e}")
            raise

        full_bounds = config.get("hyperparameter_bounds", self.params.get("hyperparameter_bounds"))
        all_params = list(full_bounds.keys())

        # Determine parameter types
        param_types = {}
        for key in all_params:
            low, up = full_bounds[key]
            param_types[key] = "int" if isinstance(low, int) and isinstance(up, int) else "float"

        # NEAT config
        population_size = config.get("population_size", self.params.get("population_size", 20))
        n_generations = config.get("n_generations", self.params.get("n_generations", 10))
        patience = config.get("optimization_patience", self.params.get("optimization_patience", 6))
        add_param_prob = config.get("neat_add_param_prob", self.params["neat_add_param_prob"])
        remove_param_prob = config.get("neat_remove_param_prob", self.params["neat_remove_param_prob"])
        compat_threshold = config.get("neat_compatibility_threshold", self.params["neat_compatibility_threshold"])
        min_params = config.get("neat_min_params", self.params["neat_min_params"])
        survival_rate = config.get("neat_survival_rate", self.params["neat_survival_rate"])
        interspecies_mate_rate = config.get("neat_interspecies_mate_rate", self.params["neat_interspecies_mate_rate"])
        neat_elitism = config.get("neat_elitism", self.params["neat_elitism"])
        mutpb = config.get("mutpb", self.params.get("mutpb", 0.2))

        # Initial parameters
        initial_params = config.get("neat_initial_params")
        if not initial_params:
            initial_params = all_params[:min(min_params, len(all_params))]

        # ── Innovation tracking ──────────────────────────────
        innovation_tracker = InnovationTracker()
        # Pre-assign innovation numbers for all params (deterministic ordering)
        for p in all_params:
            innovation_tracker.get_innovation(p)

        # ── Initialize tracking state ────────────────────────
        self.eval_counter = 0
        self.total_eval_counter = 0
        self.current_gen = 0
        self.best_fitness_so_far = float("inf")
        self.patience_counter = 0
        self.best_val_mae_so_far = None
        self.best_naive_mae_so_far = None
        self.best_test_mae_so_far = None
        self.best_test_naive_mae_so_far = None
        self.best_train_mae_so_far = None
        self.best_train_naive_mae_so_far = None
        self.best_params_so_far = {}
        self.best_at_gen_start = float("inf")

        # NEAT-specific tracking (exposed to dashboard)
        self.neat_species_count = 0
        self.neat_avg_complexity = 0
        self.neat_max_complexity = 0
        self.neat_min_complexity = 0
        self.neat_species_details = []

        # ── Callbacks ────────────────────────────────────────
        _opt_callbacks = config.get("optimization_callbacks", {})

        # ── Create initial population ────────────────────────
        def _create_genome(params_list):
            g = NeatGenome()
            for p in params_list:
                inn = innovation_tracker.get_innovation(p)
                low, high = full_bounds[p]
                if isinstance(low, int) and isinstance(high, int):
                    val = random.randint(low, high)
                else:
                    val = random.uniform(low, high)
                g.genes[inn] = NeatGene(inn, p, val)
            return g

        population = [_create_genome(initial_params) for _ in range(population_size)]
        species_list = []
        best_genome = None
        stats_history = []

        print(f"\n{'='*80}")
        print(f"[NEAT] NEAT-style Optimization Starting")
        print(f"[NEAT] Population: {population_size} | Generations: {n_generations} | Patience: {patience}")
        print(f"[NEAT] Initial parameters ({len(initial_params)}): {initial_params}")
        print(f"[NEAT] All available parameters ({len(all_params)}): {all_params}")
        print(f"[NEAT] Compatibility threshold: {compat_threshold}")
        print(f"[NEAT] Add param prob: {add_param_prob} | Remove param prob: {remove_param_prob}")
        print(f"{'='*80}\n")

        # ── Resume logic ────────────────────────────────────
        resume_enabled = config.get("optimization_resume", False)
        resume_path = _resolve_repo_path(config.get("optimization_resume_file"))
        params_path = _resolve_repo_path(config.get("optimization_parameters_file"))
        start_gen = 0

        if resume_enabled and resume_path and os.path.exists(resume_path):
            try:
                with open(resume_path, "r") as f:
                    checkpoint = json.load(f)
                # Restore innovation tracker
                if "innovation_tracker" in checkpoint:
                    innovation_tracker = InnovationTracker.from_serializable(checkpoint["innovation_tracker"])
                # Restore population
                if "population" in checkpoint:
                    population = [NeatGenome.from_serializable(gd) for gd in checkpoint["population"]]
                    print(f"[NEAT RESUME] Loaded {len(population)} genomes from checkpoint")
                # Restore optimizer state
                opt_state = checkpoint.get("optimizer_state", {})
                if opt_state.get("best_fitness_so_far") is not None:
                    self.best_fitness_so_far = float(opt_state["best_fitness_so_far"])
                if opt_state.get("best_val_mae_so_far") is not None:
                    self.best_val_mae_so_far = float(opt_state["best_val_mae_so_far"])
                if opt_state.get("best_naive_mae_so_far") is not None:
                    self.best_naive_mae_so_far = float(opt_state["best_naive_mae_so_far"])
                if opt_state.get("best_test_mae_so_far") is not None:
                    self.best_test_mae_so_far = float(opt_state["best_test_mae_so_far"])
                if opt_state.get("best_test_naive_mae_so_far") is not None:
                    self.best_test_naive_mae_so_far = float(opt_state["best_test_naive_mae_so_far"])
                if opt_state.get("best_train_mae_so_far") is not None:
                    self.best_train_mae_so_far = float(opt_state["best_train_mae_so_far"])
                if opt_state.get("best_train_naive_mae_so_far") is not None:
                    self.best_train_naive_mae_so_far = float(opt_state["best_train_naive_mae_so_far"])
                if opt_state.get("best_params_so_far"):
                    self.best_params_so_far = opt_state["best_params_so_far"]
                if opt_state.get("total_eval_counter") is not None:
                    self.total_eval_counter = int(opt_state["total_eval_counter"])
                if opt_state.get("patience_counter") is not None:
                    self.patience_counter = int(opt_state["patience_counter"])
                # Fallback: load best_params from optimization_parameters_file
                # for checkpoints saved before best_params_so_far was persisted
                if not self.best_params_so_far and params_path and os.path.exists(params_path):
                    try:
                        with open(params_path, "r") as pf:
                            self.best_params_so_far = json.load(pf)
                        print(f"[NEAT RESUME] Loaded best_params from {params_path}")
                    except Exception:
                        pass
                self.best_at_gen_start = float(self.best_fitness_so_far)
                start_gen = checkpoint.get("generation", 0) + 1
                print(f"[NEAT RESUME] Resuming from generation {start_gen}, "
                      f"best_fitness={self.best_fitness_so_far:.6f}, patience={self.patience_counter}")
            except Exception as e:
                print(f"[NEAT RESUME] Failed to load checkpoint: {e}")
                start_gen = 0
        elif resume_enabled and params_path and os.path.exists(params_path):
            # Load champion params into population[0]
            try:
                with open(params_path, "r") as f:
                    champ_params = json.load(f)
                champ_genome = NeatGenome()
                for p, v in champ_params.items():
                    if p in full_bounds:
                        inn = innovation_tracker.get_innovation(p)
                        # Convert special types back to numeric
                        if p == "use_log1p_features":
                            v = 1 if v else 0
                        elif p == "positional_encoding":
                            v = 1 if v else 0
                        elif p == "activation" and isinstance(v, str):
                            v = ACTIVATION_INDEX_TO_NAME.index(v) if v in ACTIVATION_INDEX_TO_NAME else 0
                        champ_genome.genes[inn] = NeatGene(inn, p, float(v))
                if champ_genome.genes:
                    population[0] = champ_genome
                    print(f"[NEAT RESUME] Injected champion with {len(champ_genome.genes)} params into population[0]")
            except Exception as e:
                print(f"[NEAT RESUME] Failed to load champion: {e}")

        end_gen = start_gen + n_generations

        # ── Evaluation function ──────────────────────────────
        def eval_genome(genome, gen):
            """Evaluate a genome using subprocess worker."""
            self.eval_counter += 1
            self.total_eval_counter += 1

            hyper_dict = genome.to_hyper_dict(param_types)
            genome.hyper_dict = hyper_dict

            print(f"\n--- [NEAT] Evaluating Candidate {self.eval_counter}/{population_size} | "
                  f"Gen {gen}/{end_gen - 1} | Active Params: {genome.complexity}/{len(all_params)} | "
                  f"Species: {genome.species_id or '?'} | Total Evals: {self.total_eval_counter} ---")
            print(f"Active: {genome.active_params}")
            print(f"Params: {hyper_dict}")

            new_config = config.copy()
            new_config.update(hyper_dict)

            # Subprocess isolation
            for k in ("memory_log_file", "optimizer_resource_log_file", "batch_memory_log_file"):
                if new_config.get(k):
                    new_config[k] = _resolve_repo_path(new_config.get(k))
            new_config.setdefault(
                "memory_log_tag",
                f"neat_gen{gen}_cand{int(self.eval_counter)}",
            )

            with tempfile.TemporaryDirectory(prefix="neat_cand_") as td:
                in_path = os.path.join(td, "input.json")
                out_path = os.path.join(td, "output.json")
                model_path = os.path.join(td, "candidate_model.keras")
                new_config["_doin_model_save_path"] = model_path
                with open(in_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "gen": gen,
                        "cand": int(self.eval_counter),
                        "config": _json_sanitize(new_config),
                        "hyper": hyper_dict,
                    }, f)

                env = os.environ.copy()
                env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "1")
                env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
                env.setdefault("PYTHONUNBUFFERED", "1")

                cmd = [sys.executable, "-u", "-m",
                       "optimizer_plugins.candidate_worker",
                       "--input", in_path, "--output", out_path]

                try:
                    master_fd, slave_fd = pty.openpty()
                    p = subprocess.Popen(
                        cmd, env=env, stdin=slave_fd, stdout=slave_fd,
                        stderr=slave_fd, close_fds=True,
                    )
                    os.close(slave_fd)

                    while True:
                        r, _, _ = select.select([master_fd], [], [], 0.2)
                        if master_fd in r:
                            try:
                                data = os.read(master_fd, 4096)
                            except OSError:
                                data = b""
                            if not data:
                                break
                            try:
                                sys.stdout.buffer.write(data)
                                sys.stdout.buffer.flush()
                            except Exception:
                                pass
                        if p.poll() is not None:
                            continue

                    returncode = p.wait()
                    try:
                        os.close(master_fd)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"CRITICAL: Worker spawn failed: {e}")
                    genome.fitness = float("inf")
                    return float("inf")

                if returncode != 0:
                    print(f"CRITICAL: Worker failed (returncode={returncode})")
                    genome.fitness = float("inf")
                    return float("inf")

                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception:
                    genome.fitness = float("inf")
                    return float("inf")

                fitness = float(payload.get("fitness", float("inf")))
                genome.fitness = fitness
                genome.val_mae = float(payload.get("val_mae", float("inf")))
                genome.naive_mae = float(payload.get("naive_mae", float("inf")))
                genome.train_mae = float(payload.get("train_mae", float("inf")))
                genome.train_naive_mae = float(payload.get("train_naive_mae", float("inf")))
                genome.test_mae = float(payload.get("test_mae", float("inf")))
                genome.test_naive_mae = float(payload.get("test_naive_mae", float("inf")))
                genome.model_summary = payload.get("model_summary", "")

                # Check for new champion
                is_new_champion = False
                if np.isfinite(fitness) and fitness < float(self.best_fitness_so_far):
                    self.best_fitness_so_far = float(fitness)
                    self.best_val_mae_so_far = genome.val_mae if np.isfinite(genome.val_mae) else self.best_val_mae_so_far
                    self.best_naive_mae_so_far = genome.naive_mae if np.isfinite(genome.naive_mae) else self.best_naive_mae_so_far
                    self.best_test_mae_so_far = genome.test_mae if np.isfinite(genome.test_mae) else self.best_test_mae_so_far
                    self.best_test_naive_mae_so_far = genome.test_naive_mae if np.isfinite(genome.test_naive_mae) else self.best_test_naive_mae_so_far
                    self.best_train_mae_so_far = genome.train_mae if np.isfinite(genome.train_mae) else self.best_train_mae_so_far
                    self.best_train_naive_mae_so_far = genome.train_naive_mae if np.isfinite(genome.train_naive_mae) else self.best_train_naive_mae_so_far
                    is_new_champion = True
                    self.best_params_so_far = hyper_dict.copy()

                    # Save champion
                    pf = config.get("optimization_parameters_file", "optimization_parameters.json")
                    resolved_pf = _resolve_repo_path(pf) or pf
                    try:
                        _atomic_json_dump(resolved_pf, hyper_dict)
                        print(f"  [NEAT CHAMPION] Parameters saved to {resolved_pf}")
                    except Exception as e:
                        print(f"  [NEAT CHAMPION] Failed to save: {e}")

                    # Callback: on_new_champion
                    _cb_new_champ = _opt_callbacks.get("on_new_champion")
                    if _cb_new_champ:
                        try:
                            _model_b64 = None
                            if os.path.exists(model_path):
                                import base64
                                with open(model_path, "rb") as mf:
                                    _model_b64 = base64.b64encode(mf.read()).decode("ascii")
                            _champ_metrics = {
                                "fitness": fitness,
                                "val_mae": genome.val_mae, "val_naive_mae": genome.naive_mae,
                                "train_mae": genome.train_mae, "train_naive_mae": genome.train_naive_mae,
                                "test_mae": genome.test_mae, "test_naive_mae": genome.test_naive_mae,
                                "_model_b64": _model_b64,
                            }
                            _champ_stage = {
                                "stage": 1, "total_stages": 1,
                                "generation": gen,
                                "candidate": int(self.eval_counter),
                                "total_candidates_evaluated": int(self.total_eval_counter),
                                "neat_species_count": self.neat_species_count,
                                "neat_complexity": genome.complexity,
                            }
                            _cb_new_champ(hyper_dict, fitness, _champ_metrics, gen, _champ_stage)
                        except Exception as _cb_err:
                            print(f"  [NEAT] Champion broadcast error: {_cb_err}")

                # Print result summary
                print(f"\n{'='*80}")
                print(f"[NEAT] CANDIDATE RESULT | Gen {gen}/{end_gen - 1} | "
                      f"Candidate {self.eval_counter}/{population_size} | "
                      f"Complexity: {genome.complexity} params | Total Evals: {self.total_eval_counter}")
                print(f"Active Parameters: {', '.join(genome.active_params)}")
                print(f"{'-'*80}")
                print(f"  TRAINING   -> MAE: {genome.train_mae:.6f} | Naive: {genome.train_naive_mae:.6f}")
                print(f"  VALIDATION -> MAE: {genome.val_mae:.6f} | Naive: {genome.naive_mae:.6f}")
                print(f"  TEST       -> MAE: {genome.test_mae:.6f} | Naive: {genome.test_naive_mae:.6f}")
                print(f"  FITNESS: {fitness:.6f}{'  *** NEW CHAMPION ***' if is_new_champion else ''}")
                print(f"  Champion fitness: {float(self.best_fitness_so_far):.6f} | "
                      f"Patience: {self.patience_counter}/{patience}")
                print(f"{'='*80}")

                return fitness

        # ── Re-broadcast existing champion from checkpoint ────
        # After a chain reset the NEAT checkpoint may already hold a champion
        # that the chain has never seen.  Fire the callback immediately so the
        # chain records it and can start generating blocks.
        if (np.isfinite(self.best_fitness_so_far)
                and self.best_params_so_far
                and self.best_fitness_so_far < float("inf")):
            _cb_resume = _opt_callbacks.get("on_new_champion")
            if _cb_resume:
                try:
                    _resume_metrics = {
                        "fitness": self.best_fitness_so_far,
                        "val_mae": self.best_val_mae_so_far or 0.0,
                        "val_naive_mae": self.best_naive_mae_so_far or 0.0,
                        "train_mae": self.best_train_mae_so_far or 0.0,
                        "train_naive_mae": self.best_train_naive_mae_so_far or 0.0,
                        "test_mae": self.best_test_mae_so_far or 0.0,
                        "test_naive_mae": self.best_test_naive_mae_so_far or 0.0,
                        "_model_b64": None,
                    }
                    _resume_stage = {
                        "stage": 1, "total_stages": 1,
                        "generation": start_gen,
                        "candidate": 0,
                        "total_candidates_evaluated": int(self.total_eval_counter),
                    }
                    print(f"[NEAT] Re-broadcasting checkpoint champion "
                          f"(fitness={self.best_fitness_so_far:.6f}) to chain")
                    _cb_resume(self.best_params_so_far, self.best_fitness_so_far,
                               _resume_metrics, start_gen, _resume_stage)
                    print(f"[NEAT] Checkpoint champion broadcast complete")
                except Exception as _e:
                    print(f"[NEAT] Checkpoint champion broadcast error: {_e}")

        # ── Main generation loop ─────────────────────────────
        start_opt = time.time()

        # Evaluate initial population
        print(f"\n[NEAT] Evaluating initial population ({population_size} genomes)...")
        self.current_gen = start_gen
        self.eval_counter = 0
        _force_advance_flag = False

        for genome in population:
            if genome.fitness is None:
                eval_genome(genome, start_gen)

                # Between-candidates callback
                _cb_between = _opt_callbacks.get("on_between_candidates")
                if _cb_between:
                    try:
                        _bc_stage = {
                            "stage": 1, "total_stages": 1,
                            "generation": start_gen,
                            "candidate_num": int(self.eval_counter),
                            "total_candidates": population_size,
                            "total_candidates_evaluated": int(self.total_eval_counter),
                            "fitness": genome.fitness,
                            "val_mae": genome.val_mae,
                            "train_mae": genome.train_mae,
                            "val_naive_mae": genome.naive_mae,
                            "train_naive_mae": genome.train_naive_mae,
                            "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far != float("inf") else None,
                            "candidate_params": genome.hyper_dict,
                            "model_summary": genome.model_summary,
                            "neat_species_count": self.neat_species_count,
                            "neat_complexity": genome.complexity,
                        }
                        _result = _cb_between(start_gen, int(self.eval_counter), _bc_stage)
                        if isinstance(_result, dict) and _result.get("_force_stage_advance"):
                            _force_advance_flag = True
                            break
                    except Exception as _cb_err:
                        print(f"  [NEAT] Between-candidates callback error: {_cb_err}")

        # Find best genome
        best_genome = min(population, key=lambda g: g.fitness if g.fitness is not None else float("inf"))
        self.best_at_gen_start = float(self.best_fitness_so_far)
        no_improve_counter = int(self.patience_counter)

        # Generation loop
        for gen in range(start_gen, end_gen):
            if _force_advance_flag:
                print(f"[NEAT] Force advance detected — stopping optimization")
                break

            gen_start_time = time.time()
            self.current_gen = gen
            self.eval_counter = 0
            print(f"\n{'='*80}")
            print(f"[NEAT] Generation {gen}/{end_gen - 1}")
            print(f"{'='*80}")

            best_at_gen_start = float(self.best_fitness_so_far)
            self.best_at_gen_start = best_at_gen_start

            # ── Callback: on_generation_start (migration IN) ──
            _cb_gen_start = _opt_callbacks.get("on_generation_start")
            if _cb_gen_start:
                try:
                    _stage_info = {
                        "stage": 1, "total_stages": 1,
                        "meta_mode": False,
                        "total_candidates_evaluated": int(self.total_eval_counter),
                        "population_size": population_size,
                        "neat_species_count": self.neat_species_count,
                        "neat_avg_complexity": self.neat_avg_complexity,
                    }
                    _migrant_params = _cb_gen_start(population, None, None, gen, _stage_info)
                    if isinstance(_migrant_params, dict) and _migrant_params.get("_force_stage_advance"):
                        print(f"  [NEAT] Network signalled stage advance — ending optimization")
                        break
                    if _migrant_params and isinstance(_migrant_params, dict) and not _migrant_params.get("_force_stage_advance"):
                        # Inject network champion as a NeatGenome
                        migrant_genome = NeatGenome()
                        for p, v in _migrant_params.items():
                            if p in full_bounds:
                                inn = innovation_tracker.get_innovation(p)
                                if p == "use_log1p_features":
                                    v = 1 if v else 0
                                elif p == "positional_encoding":
                                    v = 1 if v else 0
                                elif p == "activation" and isinstance(v, str):
                                    v = ACTIVATION_INDEX_TO_NAME.index(v) if v in ACTIVATION_INDEX_TO_NAME else 0
                                migrant_genome.genes[inn] = NeatGene(inn, p, float(v))
                        if migrant_genome.genes:
                            # Replace worst individual
                            worst_idx = max(range(len(population)),
                                            key=lambda i: population[i].fitness if population[i].fitness is not None else float("inf"))
                            population[worst_idx] = migrant_genome
                            print(f"  [NEAT MIGRATION] Injected network champion ({len(migrant_genome.genes)} params)")
                except Exception as _cb_err:
                    print(f"  [NEAT] gen_start callback error: {_cb_err}")

            # ── Speciation ───────────────────────────────────
            speciate(population, species_list, full_bounds, compat_threshold)
            adjust_fitness(species_list)

            # Update NEAT tracking stats
            self.neat_species_count = len(species_list)
            complexities = [g.complexity for g in population]
            self.neat_avg_complexity = sum(complexities) / len(complexities) if complexities else 0
            self.neat_max_complexity = max(complexities) if complexities else 0
            self.neat_min_complexity = min(complexities) if complexities else 0
            self.neat_species_details = [
                {"id": sp.id, "size": sp.size,
                 "best_fitness": min((g.fitness for g in sp.members if g.fitness is not None), default=float("inf")),
                 "avg_complexity": sum(g.complexity for g in sp.members) / max(sp.size, 1)}
                for sp in species_list
            ]

            print(f"[NEAT] Species: {self.neat_species_count} | "
                  f"Complexity: avg={self.neat_avg_complexity:.1f} min={self.neat_min_complexity} max={self.neat_max_complexity}")
            for sp in species_list:
                best_f = min((g.fitness for g in sp.members if g.fitness is not None), default=float("inf"))
                print(f"  Species {sp.id}: size={sp.size}, best_fitness={best_f:.6f}")

            # ── Reproduction ─────────────────────────────────
            # Calculate offspring allocation per species (proportional to adjusted fitness sum)
            total_adjusted = sum(
                sum(g.adjusted_fitness for g in sp.members if g.adjusted_fitness is not None and np.isfinite(g.adjusted_fitness))
                for sp in species_list
            )

            new_population = []

            for sp in species_list:
                # Sort members by fitness (lower is better)
                sp.members.sort(key=lambda g: g.fitness if g.fitness is not None else float("inf"))

                # Elitism: keep best individuals from each species
                for elite in sp.members[:neat_elitism]:
                    new_population.append(elite.deep_copy())

                # Calculate how many offspring this species gets
                if total_adjusted > 0 and np.isfinite(total_adjusted):
                    sp_adjusted = sum(
                        g.adjusted_fitness for g in sp.members
                        if g.adjusted_fitness is not None and np.isfinite(g.adjusted_fitness)
                    )
                    # Inverted because lower fitness = better, so lower adjusted = better
                    # Use (1/adjusted_fitness_sum) for proportional allocation
                    inv_adjusted = 1.0 / max(sp_adjusted, 1e-10)
                else:
                    inv_adjusted = 1.0

                # Select survival pool
                survival_count = max(1, int(len(sp.members) * survival_rate))
                survivors = sp.members[:survival_count]

                # Produce offspring (will be sized correctly after all species processed)
                n_offspring = max(0, int(round(population_size * inv_adjusted / max(
                    sum(1.0 / max(
                        sum(g.adjusted_fitness for g in s.members if g.adjusted_fitness is not None and np.isfinite(g.adjusted_fitness)),
                        1e-10
                    ) for s in species_list), 1e-10))) - neat_elitism)

                for _ in range(n_offspring):
                    if len(survivors) < 2 or random.random() < 0.25:
                        # Mutation only
                        parent = random.choice(survivors)
                        child = parent.deep_copy()
                    else:
                        # Crossover
                        if random.random() < interspecies_mate_rate and len(species_list) > 1:
                            # Interspecies mating
                            other_sp = random.choice([s for s in species_list if s.id != sp.id])
                            p2 = random.choice(other_sp.members)
                        else:
                            p2 = random.choice(survivors)
                        p1 = random.choice(survivors)
                        child = neat_crossover(p1, p2)

                    # Mutations
                    mutate_add_param(child, all_params, full_bounds, innovation_tracker, add_param_prob)
                    mutate_remove_param(child, min_params, remove_param_prob)
                    mutate_values(child, full_bounds, mutpb)
                    clamp_genome(child, full_bounds)
                    child.fitness = None  # Mark for evaluation
                    new_population.append(child)

            # Ensure population size is maintained
            while len(new_population) < population_size:
                # Add random individuals if we're short
                new_population.append(_create_genome(initial_params))
            new_population = new_population[:population_size]

            population = new_population

            # ── Evaluate unevaluated genomes ─────────────────
            _force_advance_flag = False
            for genome in population:
                if genome.fitness is None:
                    eval_genome(genome, gen)

                    # Between-candidates callback
                    _cb_between = _opt_callbacks.get("on_between_candidates")
                    if _cb_between:
                        try:
                            _bc_stage = {
                                "stage": 1, "total_stages": 1,
                                "generation": gen,
                                "candidate_num": int(self.eval_counter),
                                "total_candidates": sum(1 for g in population if g.fitness is None) + int(self.eval_counter),
                                "total_candidates_evaluated": int(self.total_eval_counter),
                                "fitness": genome.fitness,
                                "val_mae": genome.val_mae,
                                "train_mae": genome.train_mae,
                                "val_naive_mae": genome.naive_mae,
                                "train_naive_mae": genome.train_naive_mae,
                                "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far != float("inf") else None,
                                "candidate_params": genome.hyper_dict,
                                "model_summary": genome.model_summary,
                                "neat_species_count": self.neat_species_count,
                                "neat_complexity": genome.complexity,
                            }
                            _result = _cb_between(gen, int(self.eval_counter), _bc_stage)
                            if isinstance(_result, dict) and _result.get("_force_stage_advance"):
                                print(f"  [NEAT] Force stage advance between candidates")
                                _force_advance_flag = True
                                break
                        except Exception as _cb_err:
                            print(f"  [NEAT] Between-candidates callback error: {_cb_err}")

            if _force_advance_flag:
                print(f"[NEAT] Breaking generation loop for force advance")
                break

            # ── Update best genome ───────────────────────────
            gen_best = min(population, key=lambda g: g.fitness if g.fitness is not None else float("inf"))
            if gen_best.fitness is not None and (best_genome is None or gen_best.fitness < (best_genome.fitness or float("inf"))):
                best_genome = gen_best.deep_copy()
                best_genome.fitness = gen_best.fitness
                best_genome.hyper_dict = gen_best.hyper_dict

            # ── Patience check ───────────────────────────────
            current_best_fitness = min(
                (g.fitness for g in population if g.fitness is not None),
                default=float("inf"),
            )
            if current_best_fitness < best_at_gen_start:
                no_improve_counter = 0
                self.patience_counter = 0
                print(f"  [NEAT] New best found! fitness={current_best_fitness:.6f}")
            else:
                no_improve_counter += 1
                self.patience_counter = no_improve_counter
                print(f"  [NEAT] No improvement for {no_improve_counter} generations")

            # ── Statistics ───────────────────────────────────
            gen_end_time = time.time()
            gen_duration = gen_end_time - gen_start_time
            valid_fitnesses = [g.fitness for g in population if g.fitness is not None and np.isfinite(g.fitness)]
            avg_fitness = sum(valid_fitnesses) / len(valid_fitnesses) if valid_fitnesses else float("inf")

            stats_history.append({
                "generation": gen,
                "duration": gen_duration,
                "avg_fitness": avg_fitness,
                "best_fitness_gen": current_best_fitness,
                "champion_fitness_global": float(self.best_fitness_so_far),
                "champion_validation_mae_global": self.best_val_mae_so_far,
                "champion_validation_naive_mae_global": self.best_naive_mae_so_far,
                "species_count": self.neat_species_count,
                "avg_complexity": self.neat_avg_complexity,
            })

            # Save statistics
            stats_data = {
                "optimizer_type": "neat",
                "total_time_elapsed": gen_end_time - start_opt,
                "candidates_evaluated_so_far": int(self.total_eval_counter),
                "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far is not None else None,
                "champion_validation_mae": float(self.best_val_mae_so_far) if self.best_val_mae_so_far is not None else None,
                "champion_validation_naive_mae": float(self.best_naive_mae_so_far) if self.best_naive_mae_so_far is not None else None,
                "champion_test_mae": float(self.best_test_mae_so_far) if self.best_test_mae_so_far is not None else None,
                "champion_test_naive_mae": float(self.best_test_naive_mae_so_far) if self.best_test_naive_mae_so_far is not None else None,
                "champion_train_mae": float(self.best_train_mae_so_far) if self.best_train_mae_so_far is not None else None,
                "champion_train_naive_mae": float(self.best_train_naive_mae_so_far) if self.best_train_naive_mae_so_far is not None else None,
                "neat_species_count": self.neat_species_count,
                "neat_avg_complexity": self.neat_avg_complexity,
                "neat_max_complexity": self.neat_max_complexity,
                "neat_min_complexity": self.neat_min_complexity,
                "neat_species_details": self.neat_species_details,
                "history": stats_history,
            }
            stats_file = config.get("optimization_statistics", "optimization_stats.json")
            resolved_stats_file = _resolve_repo_path(stats_file) or stats_file
            try:
                _atomic_json_dump(resolved_stats_file, stats_data)
            except Exception as e:
                print(f"  [NEAT] Failed to save statistics: {e}")

            # Save resume checkpoint
            if resume_path:
                checkpoint = {
                    "generation": gen,
                    "population": [g.to_serializable() for g in population],
                    "innovation_tracker": innovation_tracker.to_serializable(),
                    "optimizer_state": {
                        "best_fitness_so_far": float(self.best_fitness_so_far) if self.best_fitness_so_far != float("inf") else None,
                        "best_val_mae_so_far": float(self.best_val_mae_so_far) if self.best_val_mae_so_far is not None else None,
                        "best_naive_mae_so_far": float(self.best_naive_mae_so_far) if self.best_naive_mae_so_far is not None else None,
                        "best_test_mae_so_far": float(self.best_test_mae_so_far) if self.best_test_mae_so_far is not None else None,
                        "best_test_naive_mae_so_far": float(self.best_test_naive_mae_so_far) if self.best_test_naive_mae_so_far is not None else None,
                        "best_train_mae_so_far": float(self.best_train_mae_so_far) if self.best_train_mae_so_far is not None else None,
                        "best_train_naive_mae_so_far": float(self.best_train_naive_mae_so_far) if self.best_train_naive_mae_so_far is not None else None,
                        "best_params_so_far": self.best_params_so_far if self.best_params_so_far else None,
                        "total_eval_counter": int(self.total_eval_counter),
                        "patience_counter": int(self.patience_counter),
                    },
                }
                try:
                    _atomic_json_dump(resume_path, checkpoint)
                except Exception:
                    pass

            # ── Callback: on_generation_end ──────────────────
            _cb_gen_end = _opt_callbacks.get("on_generation_end")
            if _cb_gen_end:
                try:
                    _gen_end_info = {
                        "stage": 1, "total_stages": 1,
                        "meta_mode": False,
                        "generation": gen,
                        "total_candidates_evaluated": int(self.total_eval_counter),
                        "population_size": population_size,
                        "no_improve_counter": no_improve_counter,
                        "patience": patience,
                        "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far is not None else None,
                        "champion_val_mae": float(self.best_val_mae_so_far) if self.best_val_mae_so_far is not None else None,
                        "champion_naive_mae": float(self.best_naive_mae_so_far) if self.best_naive_mae_so_far is not None else None,
                        "champion_test_mae": float(self.best_test_mae_so_far) if self.best_test_mae_so_far is not None else None,
                        "champion_train_mae": float(self.best_train_mae_so_far) if self.best_train_mae_so_far is not None else None,
                        "champion_train_naive_mae": float(self.best_train_naive_mae_so_far) if self.best_train_naive_mae_so_far is not None else None,
                        "avg_fitness": avg_fitness,
                        "best_fitness_gen": current_best_fitness,
                        "neat_species_count": self.neat_species_count,
                        "neat_avg_complexity": self.neat_avg_complexity,
                        "neat_species_details": self.neat_species_details,
                    }
                    _cb_gen_end(population, best_genome, None, gen, _gen_end_info, stats_data)
                except Exception as _cb_err:
                    print(f"  [NEAT] Generation end callback error: {_cb_err}")

            # Early stopping
            if no_improve_counter >= patience:
                print(f"\n[NEAT] Early stopping triggered after {gen + 1} generations (patience={patience})")
                break

        # ── Extract best result ──────────────────────────────
        end_opt = time.time()
        print(f"\n[NEAT] Optimization completed in {end_opt - start_opt:.2f} seconds")
        print(f"[NEAT] Total evaluations: {self.total_eval_counter}")
        print(f"[NEAT] Final species count: {self.neat_species_count}")

        if best_genome is None or best_genome.fitness is None:
            best_genome = min(population, key=lambda g: g.fitness if g.fitness is not None else float("inf"))

        best_hyper = best_genome.to_hyper_dict(param_types)
        print(f"[NEAT] Best hyperparameters ({len(best_hyper)} params): {best_hyper}")
        print(f"[NEAT] Best fitness: {self.best_fitness_so_far:.6f}")

        return best_hyper
