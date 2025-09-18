from collections import defaultdict

import fsspec
from src.storage.eos import EOS, PathLike
from classifier.config.setting import IO

from ..core import MonitorProxy, post_to_monitor


class Index(MonitorProxy):
    _urls: dict[str, list[tuple[str, str]]]

    def __init__(self):
        self._urls = defaultdict(list)

    @post_to_monitor(wait_for_return=True, acquire_lock=True)
    def add(cls, category: str, title: str, path: PathLike):
        if IO.report.is_null:
            return
        path = EOS(path)
        if not path.is_null:
            cls._urls[category].append((title, path.relative_to(IO.report)))

    @classmethod
    def render(cls):
        if cls._urls:
            from jinja2 import Environment, FileSystemLoader

            page = Environment(
                loader=FileSystemLoader(EOS(__file__).parent / "html"),
            ).get_template("report_index.html")
            urls = []
            for cat in sorted(cls._urls):
                urls.append((cat, sorted(cls._urls[cat], key=lambda x: x[0])))
            page = page.render(categories=urls)
            with fsspec.open(IO.report / "index.html", "wt") as f:
                f.write(page)
