from collections import defaultdict
from itertools import chain, cycle

import fsspec
from src.classifier.config.setting import IO, ResultKey
from src.classifier.task import Analysis, ArgParser


class LossROC(Analysis):
    argparser = ArgParser()

    def analyze(self, results: list[dict]):
        return _collect_loss_roc(results=results)


def _walk_benchmark_hyperparameter(history):
    for stage in history:
        name = stage["name"]
        if "benchmarks" in stage:
            yield stage["benchmarks"], {"stage": name}
        elif "training" in stage:
            for epoch in stage["training"]:
                pars = epoch["hyperparameters"] | {"stage": name}
                if lr := pars.get("learning rate", None):
                    pars["learning rate"] = lr[0]
                yield epoch["benchmarks"], pars


def _dict_list():
    return defaultdict(list)


def _dict_dict():
    return defaultdict(dict)


class _collect_loss_roc:
    _target = {"scalars", "roc"}
    # fmt: off
    _line = {
        "training": "solid",
        "validation": "dashed",
        None: cycle(("dotted", "dotdash", "dashdot")),
    }
    _marker = {
        "training": "square",
        "validation": "circle",
        None: cycle(("triangle", "diamond", "hex", "star", "plus")),
    }
    _color = {
        None: cycle(('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'))
    }
    # fmt: on

    @classmethod
    def _style_attr(cls, col: dict[str], key: str):
        if key not in col:
            col[key] = next(col[None])
        return col[key]

    @classmethod
    def _style(cls, datasets: list[str], classifiers: list[str]):
        styles = {"line": defaultdict(dict), "scatter": defaultdict(dict)}
        for dataset in datasets:
            styles["line"][dataset]["line_dash"] = cls._style_attr(cls._line, dataset)
            styles["scatter"][dataset]["marker"] = cls._style_attr(cls._marker, dataset)
        for classifier in classifiers:
            color = cls._style_attr(cls._color, classifier)
            styles["line"][classifier]["color"] = color
            styles["scatter"][classifier]["color"] = color
        return styles

    def __new__(cls, results: list, inline_resources=False):
        import src.data_formats.numpy as npext
        import pandas as pd

        # fetch variables
        plot = {}
        datasets = set()
        # dtypes
        int64 = {"epoch"}
        # grouped data
        g_phases: list[pd.DataFrame] = []
        g_classifiers = defaultdict(list)
        g_data = defaultdict(dict)
        g_rocs = defaultdict(_dict_dict)
        # initialize data
        for model in chain.from_iterable(map(lambda x: x[ResultKey.models], results)):
            name = model["name"].replace("__", ",").replace("_", ":")
            _data = defaultdict(list)
            _rocs = defaultdict(_dict_list)
            _phases = []
            for benchmark, hyperparameter in _walk_benchmark_hyperparameter(
                model["history"]
            ):
                if not benchmark or any(
                    not (cls._target <= set(v)) for v in benchmark.values()
                ):
                    continue
                datasets.update(benchmark)
                for k, v in benchmark.items():
                    # roc
                    aucs = {f"AUC: {r['name']}": r["AUC"] for r in v["roc"]}
                    plot.update(aucs)
                    for r in v["roc"]:
                        _rocs[f"ROC: {r['name']}"][k].append(
                            pd.DataFrame(
                                {
                                    "FPR": npext.from_.base64(r["FPR"]),
                                    "TPR": npext.from_.base64(r["TPR"]),
                                }
                            )
                        )
                    # scalars
                    scalars = v["scalars"]
                    plot.update(scalars)
                    # update data
                    _data[k].append(scalars | aucs)
                _phases.append(hyperparameter)
                for k, v in hyperparameter.items():
                    if isinstance(v, int):
                        int64.add(k)
            _phases = pd.DataFrame(_phases)
            for k in int64:
                _phases[k] = _phases[k].astype(pd.Int64Dtype())
            group = None
            for i, df in enumerate(g_phases):
                if df.equals(_phases):
                    group = i
                    break
            if group is None:
                group = len(g_phases)
                g_phases.append(_phases)
            g_classifiers[group].append(name)
            g_data[group] |= {(name, k): pd.DataFrame(v) for k, v in _data.items()}
            for k, v in _rocs.items():
                g_rocs[group][k] |= {(name, kk): vv for kk, vv in v.items()}

        jobs = []
        datasets = sorted(datasets)
        for group in range(len(g_phases)):
            classifiers = sorted(g_classifiers[group])
            milestones = g_phases[group].columns.to_list()
            milestones.remove("epoch")
            kwargs = dict(
                group=group,
                inline=inline_resources,
                phase=g_phases[group],
                style=cls._style(datasets, classifiers),
                category={
                    "dataset": datasets,
                    "classifier": classifiers,
                },
            )
            jobs.append(
                _plot_loss_auc(
                    plot=sorted(plot),
                    plot_data=g_data[group],
                    phase_milestone=milestones,
                    **kwargs,
                )
            )
            jobs.append(_list_loss_auc(plot_data=g_data[group], **kwargs))
            jobs.append(
                _plot_roc(
                    data=g_rocs[group],
                    x_axis=("FPR", "False Positive Rate"),
                    y_axis=("TPR", "True Positive Rate"),
                    figure_kwargs={
                        "height": 600,
                        "width": 1000,
                    },
                    **kwargs,
                )
            )
        return jobs


class _plot_loss_auc:
    filename = "loss-auc-{group}.html"
    title = "Loss and AUC - {group}"

    def __init__(self, group: int, inline: bool = False, **kwargs):
        self._group = group + 1
        self._inline = inline
        self._kwargs = kwargs

    @property
    def plot(self):
        from src.classifier.monitor.plot.basic import plot_multiphase_scalar

        return plot_multiphase_scalar

    def __call__(self):
        from bokeh.embed import file_html
        from bokeh.resources import CDN, INLINE
        from src.classifier.monitor import Index

        resources = INLINE if self._inline else CDN
        path = IO.report / "HCR" / self.filename.format(group=self._group)
        title = self.title.format(group=self._group)

        page = file_html(
            self.plot(**self._kwargs),
            title=title,
            resources=resources,
        )
        with fsspec.open(path, "wt") as f:
            f.write(page)
        Index.add("HCR Benchmark", title, path)


class _list_loss_auc(_plot_loss_auc):
    filename = "loss-auc-table-{group}.html"
    title = "Loss and AUC Table (last epoch)- {group}"

    @property
    def plot(self):
        from src.classifier.monitor.plot.basic import list_last_scalar

        return list_last_scalar


class _plot_roc(_plot_loss_auc):
    filename = "roc-{group}.html"
    title = "ROC - {group}"

    @property
    def plot(self):
        from src.classifier.monitor.plot.basic import plot_multiphase_curve

        return plot_multiphase_curve
