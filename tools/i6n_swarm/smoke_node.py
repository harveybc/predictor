"""Run ONE real DOIN node (doin_node.UnifiedNode) with the shared predictor wrapper in i6n_swarm mode.

Identical to ``doin-node --config`` except plugin discovery: the shared wrapper
``doin_plugins.predictor.optimizer.PredictorOptimizer`` is instantiated and
registered explicitly (entry-point metadata is not installed in pinned source
trees). Configuration loading, identity, networking, shared-population claims,
results, reproduction and shutdown are the node's own code.

    python -m tools.i6n_swarm.smoke_node --config <doin_node.json>
"""
import argparse
import asyncio
import copy
import logging
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), stream=sys.stderr,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    from doin_node.cli import load_config, load_identity, run_node
    from doin_node.unified import UnifiedNode
    from doin_plugins.predictor.optimizer import PredictorOptimizer

    config = load_config(args.config, {})
    identity = load_identity(None, config.data_dir)
    node = UnifiedNode(config, identity)
    for domain_id, role in node._domain_roles.items():
        if role.optimize and role.optimization_plugin == "predictor":
            plugin = PredictorOptimizer()
            plugin.configure(copy.deepcopy(role.optimization_config))
            node.register_optimizer_plugin(domain_id, plugin)
    asyncio.run(run_node(node))


if __name__ == "__main__":
    main()
