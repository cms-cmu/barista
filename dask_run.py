#!/usr/bin/env python
import argparse
import logging
import warnings

import dask_awkward as dak
from distributed import Client, LocalCluster
from rich.logging import RichHandler
from rich.pretty import pretty_repr

import dask
from src.storage.eos import EOS

warnings.filterwarnings("ignore")

from analysis_dask import load_configs
from analysis_dask._io import _FileDumper, write_string
from analysis_dask.setup import setup

setup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["INFO", "DEBUG", "WARNING", "ERROR"],
        help="Set logging level.",
    )
    parser.add_argument(
        "--diagnostics",
        default=None,
        dest="diagnostics",
        help="The path to store the diagnostic reports. If not provided, diagnostics will be skipped.",
    )
    args = parser.parse_args()
    # diagnostics
    if args.diagnostics:
        diagnostic = EOS(args.diagnostics)
        diagnostic_dumper = _FileDumper()
        diagnostic_dumper.register(write_string, "svg")
    else:
        diagnostic = None
    # logging
    logging_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=logging_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(level=logging_level, markup=True)],
    )

    # load configs
    #   tasks: the main analysis tasks (dask tasks)
    #   client: the client to run the main tasks (distributed client)
    #   post-tasks: the tasks that run locally after all main tasks are done (any callable with 1 argument)
    logging.info("Loading configs...\n" + "\n".join(args.configs))
    configs = load_configs(*args.configs)
    tasks = configs["tasks"]
    post_tasks = configs.get("post-tasks") or []
    if "client" in configs:
        client = configs["client"]
        logging.info(f"Using client: {client}")
    else:
        client = Client(
            LocalCluster(n_workers=1, threads_per_worker=1, processes=False)
        )
        logging.warning("No client is specified. Fallback to single-threading.")

    # analyze tasks
    if diagnostic:
        # task graph
        visualizer = dict(filename=None, format="svg", engine="graphviz")
        path = diagnostic / "task_graph.svg"
        diagnostic_dumper(
            dask.visualize(tasks, **visualizer, optimize_graph=False).data, path
        )
        logging.info(f"Task graph stored in {path}.")
        path = diagnostic / "task_graph_optimized.svg"
        diagnostic_dumper(
            dask.visualize(tasks, **visualizer, optimize_graph=True).data, path
        )
        logging.info(f"Optimized task graph stored in {path}.")
        # necessary columns
        columns = {k: list(v) for k, v in dak.report_necessary_columns(tasks).items()}
        logging.info(pretty_repr(columns))
        diagnostic_dumper(columns, diagnostic / "necessary_columns.json")

    # compute tasks
    logging.info("Computing tasks...")
    (results,) = client.compute(tasks, sync=True)

    # run post-tasks locally
    logging.info("Running post-tasks locally...")
    for task in post_tasks:
        task(results)

    client.shutdown()
