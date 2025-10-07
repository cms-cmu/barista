from __future__ import annotations

import logging
from abc import abstractmethod
from collections import defaultdict
from functools import cache, cached_property, partial
from itertools import chain
from typing import TYPE_CHECKING, Iterable

from classifier.task import ArgParser, parse
from classifier.typetools import enum_dict

from ...setting.df import Columns
from ...setting.HCR import Input, InputBranch, MassRegion, NTag
from ...setting.ml import KFold
from ...state import Flags
from ...state.label import MultiClass
from .._root import LoadGroupedRoot
from . import _group

if TYPE_CHECKING:
    import pandas as pd


class _Derived:
    region_index: str = "region_index"
    ntag_index: str = "ntag_index"


def _sort_map(obj: dict[frozenset[str]]):
    obj = {(*sorted(k),): v for k, v in obj.items()}
    return {k: obj[k] for k in sorted(obj)}


def _debug_print_weight(df: pd.DataFrame):
    from rich.table import Table

    tables = []
    for region, byregion in df.groupby(_Derived.region_index):
        tables.append(f"In region {MassRegion(region).name}:")
        table = Table("Class", "Count", "Weight")
        for label, bylabel in byregion.groupby(Columns.label_index):
            table.add_row(
                str(MultiClass.labels[label]),
                str(len(bylabel)),
                str(bylabel[Columns.weight].sum()),
            )
        tables.append(table)
    logging.debug("The following events are loaded:", *tables)
    return df


class Common(LoadGroupedRoot):
    argparser = ArgParser()
    argparser.add_argument(
        "--branch",
        metavar="BRANCH",
        nargs="+",
        action="extend",
        default=[],
        help="additional branches",
    )
    argparser.add_argument(
        "--preprocess",
        metavar=("GROUP", "PROCESSOR"),
        nargs="+",
        action="append",
        default=[],
        help="additional preprocessors",
    )

    @cache
    def from_root(self, groups: frozenset[str]):
        from classifier.df.io import FromRoot

        friends = []
        for k, v in self.friends.items():
            if k <= groups:
                friends.extend(v)

        pres = []
        for g in chain(self._preprocess_from_opts, self._preprocess_by_group):
            pres.extend(g(groups))
        pres.extend(self.preprocessors)

        return FromRoot(
            friends=friends,
            branches=self._branches.intersection,
            preprocessors=pres,
        )

    @cached_property
    def _preprocess_from_opts(self):
        return [
            _group.fullmatch(
                parse.split_nonempty(opts[0], ","),
                processors=[partial(parse.instance, opts[1:], "classifier")],
            )
            for opts in self.opts.preprocess
        ]

    @cached_property
    def _preprocess_by_group(self):
        return self.preprocess_by_group()

    def preprocess_by_group(self) -> Iterable[_group.ProcessorGenerator]:
        return ()

    @cached_property
    def _branches(self):
        return self.other_branches().union(
            InputBranch.feature_ancillary,
            InputBranch.feature_CanJet,
            InputBranch.feature_NotCanJet,
            self.opts.branch,
        )

    @abstractmethod
    def other_branches(self) -> set[str]: ...


class CommonTrain(Common):
    trainable = True

    argparser = ArgParser()
    argparser.add_argument(
        "--JCM-weight",
        metavar=("GROUPS", "PATH"),
        nargs=2,
        action="append",
        default=[],
        help="comma-separated groups and the path to the JCM weight file",
    )

    def __init__(self):
        super().__init__()
        from classifier.df.tools import drop_columns, map_selection_to_flag

        # fmt: off
        (
            self.to_tensor
            .add(KFold.offset, KFold.offset_dtype).columns(Columns.event)
            .add(Input.label, Columns.index_dtype).columns(Columns.label_index)
            .add(Input.region, Columns.index_dtype).columns(_Derived.region_index)
            .add(Input.weight, "float32").columns(Columns.weight)
            .add(Input.ancillary, "float32").columns(*InputBranch.feature_ancillary)
            .add(Input.CanJet, "float32").columns(*InputBranch.feature_CanJet, target=InputBranch.n_CanJet)
            .add(Input.NotCanJet, "float32").columns(*InputBranch.feature_NotCanJet, target=InputBranch.n_NotCanJet, pad_value=-1)
        )
        self.preprocessors.extend(
            [
                map_selection_to_flag(
                    **enum_dict(MassRegion)
                ).set(name=_Derived.region_index),
                map_selection_to_flag(
                    **enum_dict(NTag)
                ).set(name=_Derived.ntag_index),
                drop_columns(
                    "ZZSR", "ZHSR", "HHSR", "SR", "SB",
                    "fourTag", "threeTag",
                ),
            ]
        )
        if Flags.debug:
            self.postprocessors.append(_debug_print_weight)
        # fmt: on

    def other_branches(self):
        return {
            "ZZSR",
            "ZHSR",
            "HHSR",
            "SR",
            "SB",
            "fourTag",
            "threeTag",
            "passHLT",
            Columns.event,
            Columns.weight,
        }

    def preprocess_by_group(self):
        ps = []
        if self.opts.JCM_weight:
            from classifier.compatibility.JCM.fit import apply_JCM_from_list

            for opts in self.opts.JCM_weight:
                ps.append(
                    _group.fullmatch(
                        parse.split_nonempty(opts[0], ","),
                        processors=[partial(apply_JCM_from_list, path=opts[1])],
                    )
                )
        return ps

    def debug(self):
        import logging

        from classifier.config.state.label import MultiClass
        from rich.pretty import pretty_repr

        pres = defaultdict(list)
        for gs in self.files:
            for p in chain(self._preprocess_from_opts, self._preprocess_by_group):
                pres[gs].extend(p(gs))
        logging.debug(
            "friends:",
            pretty_repr(
                _sort_map(
                    {
                        k: [(v.name, v.branches) for v in vs]
                        for k, vs in self.friends.items()
                    }
                )
            ),
        )
        logging.debug("files:", pretty_repr(_sort_map(self.files)))
        logging.debug(
            "preprocessors:",
            pretty_repr(_sort_map(pres) | {"common": self.preprocessors}),
        )
        logging.debug("postprocessors:", pretty_repr(self.postprocessors))
        logging.debug("tensor:", pretty_repr(self.to_tensor._columns))
        logging.debug("all labels:", pretty_repr(MultiClass.labels))
        logging.debug("trainable labels:", pretty_repr(MultiClass.trainable_labels))


class CommonEval(Common):
    evaluable = True

    def __init__(self):
        super().__init__()

        # fmt: off
        (
            self.to_tensor
            .add(KFold.offset, KFold.offset_dtype).columns(Columns.event)
            .add(Input.ancillary, "float32").columns(*InputBranch.feature_ancillary)
            .add(Input.CanJet, "float32").columns(*InputBranch.feature_CanJet, target=InputBranch.n_CanJet)
            .add(Input.NotCanJet, "float32").columns(*InputBranch.feature_NotCanJet, target=InputBranch.n_NotCanJet, pad_value=-1)
        )
        # fmt: on

    def other_branches(self):
        return {
            Columns.event,
        }

    def preprocess_by_group(self):
        return [
            _group.add_year(),
        ]

    def debug(self):
        import logging

        from rich.pretty import pretty_repr

        logging.debug(
            "friends:",
            pretty_repr(
                _sort_map(
                    {
                        k: [(v.name, v.branches) for v in vs]
                        for k, vs in self.friends.items()
                    }
                )
            ),
        )
        logging.debug("files:", pretty_repr(_sort_map(self.files)))
        logging.debug("tensor:", pretty_repr(self.to_tensor._columns))
