import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import importlib
import logging
from itertools import chain

from classifier.task import main
from rich.logging import RichHandler

from .utils.import_check import ImportTracker, walk_packages


def walk_configs():
    failed = False
    with import_checker() as _import_checker:
        checkers = [_import_checker]
        for module in chain(
            *[
                walk_packages(
                    "/".join(
                        main.EntryPoint._fetch_config("", main._ModCtx(test=test))[:-1]
                    ),
                    Path(__file__).resolve().parents[2],
                )
                for test in [False, True]
            ]
        ):
            logging.info(f'Checking "{module}"')
            for checker in checkers:
                failed |= checker(module)

    return failed


class import_checker:
    def __init__(self):
        self._tracker = ImportTracker(
            # fmt: off
            [
                "torch",
                "numpy", "pandas", "numba",
                "awkward", "uproot",
                "bokeh",
            ]
            # fmt: on
        )

    def __enter__(self):
        self._tracker.__enter__()
        return self

    def __exit__(self, *_):
        self._tracker.__exit__()

    def __call__(self, module: str):
        failed = False
        try:
            importlib.import_module(module)
        except Exception as e:
            logging.error(f'Failed to import "{module}": {e}', exc_info=e)
            failed = True
        if not failed:
            if len(self._tracker.tracked) > 0:
                logging.error(f'Module "{module}" imports {self._tracker.tracked}')
        self._tracker.reset()
        return failed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=False, show_path=False, markup=True)],
    )
    failed = walk_configs()
    if failed:
        exit(1)
