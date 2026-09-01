import importlib
import logging

from rich.logging import RichHandler

from .utils.import_check import ImportTracker, walk_packages


def walk_configs():
    import os
    failed = False
    with import_checker() as _import_checker:
        checkers = [_import_checker]
        ext_env = os.environ.get("CLASSIFIER_CONFIG_PATHS", "")
        config_paths = ["src/classifier/config", "src/classifier/test/config"]
        if ext_env:
            for ext in ext_env.split(":"):
                if ext:
                    config_paths.insert(0, f"{ext}/classifier/config")
        for config_path in config_paths:
            if not os.path.exists(config_path):
                continue
            for module in walk_packages(config_path):
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
        self._safe = set()
        return self

    def __exit__(self, *_):
        self._safe = None
        self._tracker.__exit__()

    def __call__(self, module: str):
        failed = False
        if module in self._safe:
            return False
        try:
            importlib.import_module(module)
        except Exception as e:
            logging.error(f'Failed to import "{module}": {e}', exc_info=e)
            failed = True
        if not failed:
            if len(self._tracker.tracked) > 0:
                logging.error(f'Module "{module}" imports {self._tracker.tracked}')
            else:
                self._safe.update(self._tracker.imported)
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
