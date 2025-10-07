import json

import fsspec
from classifier.task import Analysis, ArgParser

from ..setting import IO


class Usage(Analysis):
    argparser = ArgParser()
    argparser.add_argument(
        "--usage",
        help="the path to usage data. If not provided, the current usage data will be used.",
        default=None,
    )
    argparser.add_argument(
        "--time-step",
        help="the minimum plotting time step in seconds.",
        type=float,
        default=0,
    )

    def analyze(self, _):
        path = self.opts.usage
        if path is None:
            from classifier.monitor.usage import Usage as _Usage

            data = _Usage._serialize()
            if data is None:
                return []
        else:
            with fsspec.open(self.opts.input) as f:
                data = json.load(f)
        return [_usage_report(data, self.opts.time_step)]


class _usage_report:
    def __init__(self, data: dict, step: float):
        self._data = data
        self._step = step

    def __call__(self):
        from classifier.monitor.usage.analyze import generate_report

        generate_report(self._data, IO.report / "usage", time_step=self._step)
