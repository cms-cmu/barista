from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.classifier.config.analysis.HCR._loss_roc import _collect_loss_roc
from src.classifier.task import Analysis, ArgParser, parse

if TYPE_CHECKING:
    import numpy.typing as npt


class CompareResults(Analysis):
    argparser = ArgParser()

    argparser.add_argument(
        "--result",
        type=str,
        help="Path to the result file or the dot separated key to the object in the result argument.",
    )
    argparser.add_argument(
        "--reference",
        help=f"Path to the reference file or a {parse.EMBED} object",
        required=True,
    )

    def _get_result(self, result):
        from src.classifier.task.parse._dict import _fetch_key

        if result is None:
            result = parse.mapping(self.opts.result)
        else:
            if self.opts.result:
                for k in self.opts.result.split("."):
                    result = _fetch_key(result, k)
        return result

    def _get_reference(self):
        reference = self.opts.reference
        if isinstance(reference, str):
            reference = parse.mapping(reference)
        return reference


class CompareFloat(Analysis):
    argparser = ArgParser()

    argparser.add_argument(
        "--atol", type=float, help="Absolute tolerance", default=1e-6
    )
    argparser.add_argument(
        "--rtol", type=float, help="Relative tolerance", default=1e-6
    )

    def _compare_float(self, a: npt.ArrayLike, b: npt.ArrayLike):
        import numpy as np

        return np.allclose(a, b, atol=self.opts.atol, rtol=self.opts.rtol)


class CompareRecursiveJSON(CompareFloat, CompareResults):
    def _compare(self, a, b):
        if type(a) is not type(b):
            return False
        match a:
            case dict():
                return self._compare_mapping(a, b)
            case list():
                return self._compare_list(a, b)
            case float():
                return self._compare_float(a, b)
            case str() | int() | bool() | None:
                return a == b
            case _:
                return False

    def _compare_list(self, a, b):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not self._compare(a[i], b[i]):
                return False
        return True

    def _compare_mapping(self, a: dict[str], b: dict[str]):
        if not set(a.keys()) == set(b.keys()):
            return False
        for k in a.keys():
            if not self._compare(a[k], b[k]):
                return False
        return True

    def _job(self, result, reference):
        logging.info("result", yaml.safe_dump(result))
        if not self._compare(result, reference):
            from rich.pretty import pretty_repr

            logging.error(
                "Result does not match reference",
                "result:",
                pretty_repr(result),
                "reference:",
                pretty_repr(reference),
            )
            raise ValueError("Result does not match reference")
        else:
            logging.info("Result matches reference")

    def analyze(self, result=None):
        result = self._get_result(result)
        reference = self._get_reference()

        return [partial(self._job, result, reference)]


def _tree():
    return defaultdict(_tree)


def _tree_to_dict(tree: dict):
    return {
        k: _tree_to_dict(v) if isinstance(v, defaultdict) else v
        for k, v in tree.items()
    }


class CompareTrainingHCR(CompareRecursiveJSON):
    argparser = ArgParser()
    argparser.remove_argument("--result")

    def _get_result(self, result):
        import pandas as pd

        from src.utils import unique

        _, _, groups = _collect_loss_roc._collect_data(result)
        outputs = []
        for group in groups:
            dfs = {
                k: pd.concat([group.data[k], group.phase], axis=1)
                for k in sorted(group.data)
            }
            columns = sorted(
                unique(chain.from_iterable(map(lambda x: x.columns, dfs.values())))
            )
            output = _tree()
            for ks, df in dfs.items():
                cd = output
                for k in ks:
                    cd = cd[k]
                for col in columns:
                    if col in df.columns:
                        val = df[col].iloc[-1]
                        if pd.isna(val):
                            val = None
                        else:
                            try:
                                val = val.item()
                            except AttributeError:
                                pass
                    cd[col] = val
            outputs.append(_tree_to_dict(output))
        return outputs


class CompareRootFile(CompareFloat):
    argparser = ArgParser()

    argparser.add_argument(
        "--result",
        type=Path,
        help="Path to the result file.",
        required=True,
    )
    argparser.add_argument(
        "--reference",
        type=Path,
        help="Path to the reference file",
        required=True,
    )

    def _job(self, result, reference):
        import pandas as pd

        from src.data_formats.root import Chunk, TreeReader

        reader = TreeReader()
        result = reader.arrays(Chunk(result), library="pd")
        reference = reader.arrays(Chunk(reference), library="pd")
        try:
            pd.testing.assert_frame_equal(
                result,
                reference,
                check_dtype=False,
                atol=self.opts.atol,
                rtol=self.opts.rtol,
            )
        except AssertionError as e:
            logging.error(
                "Result does not match reference",
                exc_info=e,
            )
            raise e
        else:
            logging.info(
                f"Result {str(self.opts.result)} matches reference {str(self.opts.reference)}"
            )

    def analyze(self, _=None):
        return [partial(self._job, self.opts.result, self.opts.reference)]
