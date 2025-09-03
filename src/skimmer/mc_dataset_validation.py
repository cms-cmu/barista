import logging
import math
from argparse import ArgumentParser
from collections import defaultdict

import fsspec
import numpy as np
import uproot
import yaml
from dask.delayed import delayed
from dask.distributed import Client, LocalCluster
from rich.logging import RichHandler

from mc_weight_outliers import OutlierByMedian

_NANOAOD = "nanoAOD"


for loggers in [
    "lpcjobqueue",
    *map("distributed.{}".format, ("core", "worker", "scheduler", "nanny")),
]:
    logger = logging.getLogger(loggers)
    logger.setLevel(logging.ERROR)

MultipleDatasets = "MultipleDatasets"
InvalidFile = "InvalidFile"
NoFile = "NoFile"
NoReplica = "NoReplica"
MismatchedCount = "MismatchedCount"
MismatchedSumw = "MismatchedSumw"
Outliers = "Outliers"


class fetch_metadata:
    def __init__(self):
        from dbs.apis.dbsClient import DbsApi

        self.dbs = DbsApi("https://cmsweb.cern.ch/dbs/prod/global/DBSReader")

    def __call__(
        self, dataset: tuple[tuple[str, str], str]
    ) -> tuple[tuple[str, str], dict]:
        key, path = dataset
        metadata = self.dbs.listFileSummaries(dataset=path)
        if len(metadata) > 1:
            return key, {"path": path, "errors": [{"type": MultipleDatasets}]}
        metadata = metadata[0]
        nevents = metadata["num_event"]
        nfiles = metadata["num_file"]
        if nfiles == 0:
            return key, {"path": path, "errors": [{"type": NoFile}]}
        files = self.dbs.listFiles(dataset=path, detail=True)
        files = [
            dict(path=file["logical_file_name"], nevents=file["event_count"])
            for file in files
        ]
        logging.info(f"Fetched metadata for {key}")
        return key, {
            "files": files,
            "nevents": nevents,
            "nfiles": nfiles,
        }


class fetch_replicas:
    def __init__(self, sites: list[str]):
        self.sites = sites

    def __call__(self, datasets: dict):
        from rucio.client import Client as RucioClient

        dids = []
        for dataset in datasets.values():
            dids.extend(
                {"scope": "cms", "name": file["path"]} for file in dataset["files"]
            )
        replicas = {}
        client = RucioClient()
        for replica in client.list_replicas(dids=dids, schemes=["root"]):
            pfns = {v["rse"]: k for k, v in replica["pfns"].items()}
            replicas[replica["name"]] = {k: pfns[k] for k in self.sites if k in pfns}
        return replicas


class sanity_check:
    def __init__(self, threshold: float = 100):
        self.threshold = threshold

    @delayed
    def __call__(self, nevents: int, replicas: dict[str, str]):
        result = {}
        errors = []
        for site, url in replicas.items():
            try:
                with uproot.open(url) as file:
                    result = self.analyze(file, nevents)
                    break
            except Exception as e:
                errors.append(
                    {
                        "type": InvalidFile,
                        "site": site,
                        "url": url,
                        "error": str(e),
                    }
                )
        if errors:
            result["errors"] = result.get("errors", []) + errors
        return result

    def analyze(self, file: uproot.ReadOnlyDirectory, nevents: int):
        result = {}
        errors = []
        runs = file["Runs"].arrays(["genEventCount", "genEventSumw"])
        count = int(np.sum(runs["genEventCount"]))
        sumw = float(np.sum(runs["genEventSumw"].to_numpy().astype(np.float64)))
        if count != nevents:
            errors.append(
                {
                    "type": MismatchedCount,
                    "DAS": nevents,
                    "Runs": count,
                }
            )
        events = file["Events"].arrays(["event", "genWeight"])
        genWeight = events.genWeight
        minimum = float(np.min(np.abs(genWeight)))
        _sumw = float(
            np.sum(events.genWeight.to_numpy().astype(np.float64))
        )  # avoid numerical error
        if count != len(events):
            errors.append(
                {
                    "type": MismatchedCount,
                    "Runs": count,
                    "Events": len(events),
                }
            )
        if not math.isclose(_sumw, sumw, abs_tol=minimum):
            errors.append(
                {
                    "type": MismatchedSumw,
                    "Runs": sumw,
                    "Events": _sumw,
                    "minimum": minimum,
                }
            )
        outlier_checker = OutlierByMedian(self.threshold)
        outliers = events[~outlier_checker(genWeight)]
        if len(outliers):
            result["outliers"] = [
                {"event": int(e.event), "genWeight": float(e.genWeight)}
                for e in outliers
            ]
            result["median"] = float(outlier_checker.last_median)
        if errors:
            result["errors"] = errors
        return result


if __name__ == "__main__":
    from src.utils.argparser import DefaultFormatter

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=False, show_path=False, markup=True)],
    )
    argparser = ArgumentParser(formatter_class=DefaultFormatter)
    argparser.add_argument(
        "-m",
        "--metadatas",
        required=True,
        help="path to metadata files",
        nargs="+",
        action="extend",
        default=[],
    )
    argparser.add_argument(
        "-d",
        "--datasets",
        help="dataset names, if not provided, all MC datasets will be used",
        nargs="+",
        action="extend",
        default=[],
    )
    argparser.add_argument(
        "-o",
        "--output",
        help="output directory",
        default=".",
    )
    argparser.add_argument(
        "--threshold",
        type=float,
        help="threshold to determine outliers comparing to the median",
        default=200,
    )
    argparser.add_argument(
        "--sites",
        help="priority of sites to read root files",
        nargs="+",
        default=["T3_US_FNALLPC"],
    )
    argparser.add_argument(
        "--workers",
        type=int,
        help="max number of workers when analyzing the events",
        default=8,
    )
    argparser.add_argument(
        "--condor",
        action="store_true",
        help="submit analysis jobs to condor",
    )
    argparser.add_argument(
        "--dashboard",
        help="dashboard address",
        default=":10200",
    )
    args = argparser.parse_args()

    # outputs
    errors: list[dict] = []
    outliers: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    # fetch datasets
    datasets: dict[tuple[str, str], str] = {}
    dataset_lists = {}
    for metadata in args.metadatas:
        with fsspec.open(metadata, "rt") as f:
            dataset_lists.update(yaml.safe_load(f).get("datasets", {}))
    for dataset in args.datasets or dataset_lists:
        if dataset not in dataset_lists:
            continue
        metadata = dataset_lists[dataset]
        for year in metadata:
            if not isinstance(datatiers := metadata[year], dict):
                continue  # not a dataset
            if _NANOAOD not in datatiers:
                continue  # no nanoAOD
            nanoaod = datatiers[_NANOAOD]
            if not isinstance(nanoaod, str):
                continue  # not MC
            datasets[(dataset, year)] = nanoaod
    # fetch files
    logging.info("[blue]Fetching metadata from DAS.[/blue]")
    files = dict(map(fetch_metadata(), datasets.items()))
    for k, v in [*files.items()]:
        if "errors" in v:
            errors.append(v)
            del files[k]
    # fetch replicas
    logging.info("[blue]Fetching replicas.[/blue]")
    replicas = fetch_replicas(sites=args.sites)(files)
    # analysis
    analyzer = sanity_check(threshold=args.threshold)
    logging.info("[blue]Analyzing root files.[/blue]")
    results = {}
    for k, files in files.items():
        if files is None:
            continue
        results[k] = {}
        for file in files["files"]:
            replica = replicas.get(file["path"])
            if not replica:
                errors.append(
                    {
                        "path": file["path"],
                        "errors": [{"type": NoReplica, "sites": args.sites}],
                    }
                )
                continue
            results[k][file["path"]] = analyzer(
                nevents=file["nevents"], replicas=replica
            )
    # submit to dask
    if args.condor:
        from lpcjobqueue import LPCCondorCluster

        cluster = LPCCondorCluster(
            transfer_input_files=["coffea4bees/analysis/"],
            shared_temp_directory="/tmp",
            cores=1,
            memory="2GB",
            ship_env=False,
            scheduler_options={"dashboard_address": args.dashboard},
            worker_extra_args=[
                "--worker-port 10000:10100",
                "--nanny-port 10100:10200",
            ],
        )
    else:
        cluster = LocalCluster(dashboard_address=args.dashboard, threads_per_worker=1)
    cluster.adapt(minimum=1, maximum=args.workers)
    client = Client(cluster)
    logging.info("[blue]Running dask jobs.[/blue]")
    results: dict[str, dict[str, dict]] = client.compute(results, sync=True)
    # collect results
    for k, v in results.items():
        _outliers = []
        for file, result in v.items():
            _error: list = result.get("errors", [])
            if "outliers" in result:
                _outlier = result["outliers"]
                _outliers.extend(e["event"] for e in _outlier)
                _error.append(
                    {
                        "type": "Outliers",
                        "events": _outlier,
                        "median": result["median"],
                        "threshold": args.threshold,
                    }
                )
            if _error:
                errors.append(
                    {
                        "path": file,
                        "errors": _error,
                    }
                )
        if _outliers:
            _outliers = sorted(_outliers)
            outliers[k[0]][k[1]] = _outliers
            logging.error(f"Outliers found for {k}: {_outliers}")
        else:
            logging.info(f"No outliers found for {k}")

    from src.storage.eos import EOS

    output = EOS(args.output)
    yaml.SafeDumper.add_representer(
        defaultdict, lambda dumper, data: dumper.represent_dict(data.items())
    )
    with fsspec.open(output / "outliers.yml", "wt") as f:
        yaml.safe_dump(outliers, f)
    with fsspec.open(output / "errors.yml", "wt") as f:
        yaml.safe_dump(errors, f)
