from __future__ import annotations

import copy
import importlib
import inspect
import operator as op
import re
import sys
import traceback
from dataclasses import dataclass, field
from functools import partial
from inspect import _ParameterKind as ParKind
from os import PathLike, fspath, get_terminal_size
from types import MappingProxyType, MethodType
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
)

from ._io import FileLoader, _split_path, load_url, resolve_path
from ._patch import PatchAction, PatchedLoader
from ._utils import SimpleTree, block_divider, block_indent, check_reserved, format_repr

P = ParamSpec("P")
T = TypeVar("T")

ConfigSource = str | PathLike | dict
"""
str, ~os.PathLike, dict: A path to the config file or a nested dict.
"""


class _error_msg:
    other_lines = 15
    value_lines = 5

    def __new__(
        cls,
        error: Exception,
        path: str = None,
        key: str = ...,
        span: tuple[int, int] = None,
        value=...,
    ):
        error_msg = str(error)
        block_start, block_end = block_divider()
        info = f"When parsing config:\n{block_start}"
        if path is not None:
            info += f"\n{path}"
        if key is not ...:
            info += f"\n  {key}:\n"
            if span is not None:
                info += " " * (span[0] + 2) + "^" * (span[1] - span[0]) + "\n"
            if value is not ...:
                value_lines = max(
                    cls.value_lines,
                    get_terminal_size().lines - cls.other_lines - error_msg.count("\n"),
                )
                info += block_indent(format_repr(value, value_lines), "    ") + "\n"
        return SyntaxError(
            f"""
{info}{block_end}
the following exception occurred:
  {type(error).__name__}:
{block_indent(error_msg, "    ")}
{block_start}
Traceback:
{block_indent("".join(traceback.format_tb(error.__traceback__)), "    ")}
{block_end} 
"""
        )


if sys.version_info < (3, 11):

    class _ReservedTag:
        code = "code"

        select = "select"
        case = "case"
        include = "include"
        patch = "patch"

        literal = "literal"
        discard = "discard"
        comment = "comment"

        file = "file"
        type = "type"
        attr = "attr"
        extend = "extend"
        var = "var"
        ref = "ref"
        map = "map"

    _ReservedTag.__members__ = list(
        k for k in vars(_ReservedTag) if not k.startswith("_")
    )


else:
    from enum import StrEnum, auto

    class _ReservedTag(StrEnum):
        code = auto()

        select = auto()
        case = auto()
        include = auto()
        patch = auto()

        literal = auto()
        discard = auto()
        comment = auto()

        file = auto()
        type = auto()
        attr = auto()
        extend = auto()
        var = auto()
        ref = auto()
        map = auto()


class _NoTag:
    unique = None

    @staticmethod
    def has(_: str):
        return False

    @staticmethod
    def get(_: str):
        return ...

    @staticmethod
    def apply(*, key: str, value: str, **_):
        return key, value


class _MatchedTags:
    flags = frozenset(
        (
            _ReservedTag.case,
            _ReservedTag.code,
            _ReservedTag.literal,
            _ReservedTag.discard,
        )
    )
    uniques = frozenset(
        (
            _ReservedTag.select,
            _ReservedTag.include,
            _ReservedTag.patch,
        )
    )
    skips = frozenset(
        (
            _ReservedTag.code,
            _ReservedTag.comment,
        )
    )

    def __init__(
        self,
        tags: list[tuple[str, str]],
        parser: _Parser,
        debug: dict,
    ):
        self.debug = {
            "key": debug["key"],
            "spans": {
                "flags": {},
                "tags": [],
                "unique": None,
            },
        }
        self.parser = parser
        self.tags = list[tuple[str, str]]()
        self.parsed = dict[str, str]()
        self.unique: Optional[tuple[str, str]] = None

        unique_check = 0

        for (k, v), span in zip(tags, debug["spans"]):
            if k not in self.skips:
                unique_check += 1
            if k in self.flags:
                self.parsed[k] = v
                self.debug["spans"]["flags"][k] = span
            elif k in self.uniques:
                if self.unique is not None:
                    break
                self.unique = (k, v)
                self.debug["spans"]["unique"] = span
            else:
                self.tags.append((k, v))
                self.debug["spans"]["tags"].append(span)

        if self.unique is not None and unique_check > 1:
            raise _error_msg(
                error=ValueError(f"cannot use <{self.unique[0]}> with other tags"),
                path=self.parser.path,
                key=self.debug["key"],
                span=self.debug["spans"]["unique"],
            )

    def has(self, tag: str):
        return tag in self.parsed

    def get(self, tag: str):
        return self.parsed.get(tag, ...)

    def apply(self, *, key: str, value, local: dict):
        raw_value = value
        parsers = self.parser.custom.parsers
        for i, (tag_k, tag_v) in enumerate(self.tags):
            if (parser := parsers.get(tag_k)) is not None:
                try:
                    key, value = parser(
                        key=key,
                        value=value,
                        tag=tag_v,
                        tags=MappingProxyType(self.parsed),
                        local=local,
                        path=self.parser.path,
                    )
                except RecursionError:
                    raise
                except Exception as e:
                    raise _error_msg(
                        error=e,
                        path=self.parser.path,
                        key=self.debug["key"],
                        span=self.debug["spans"]["tags"][i],
                        value=raw_value,
                    ) from None
            self.parsed[tag_k] = tag_v
        return key, value

    def include(self, tag: str, paths: str | list[str], result: dict):
        if isinstance(paths, list):
            paths = self.parser.list(paths)
        else:
            paths = [paths]
        try:
            return self.parser.custom.parse(
                *resolve_path(self.parser.path, tag, *map(fspath, paths)),
                result=result,
            )
        except RecursionError:
            raise RecursionError("recursive include may exist.") from None
        except Exception:
            raise

    def patch(self, tag: Optional[str], key: str, value: Any):
        patch = self.parser.custom.patches
        if not isinstance(value, list):
            value = [value]
        match tag:
            case None | "absolute" | "relative":
                value = copy.deepcopy(value)
                for i, path in enumerate(
                    resolve_path(
                        self.parser.path, tag, *(patch["path"] for patch in value)
                    )
                ):
                    value[i]["path"] = path
                patch.register(value, key, self.parser.path)
            case "install":
                patch.install(value)
            case "uninstall":
                patch.uninstall(value)
            case _:
                raise ValueError(f"unknown method: {tag}")


class FlagParser:
    def __init__(self, *booleans, **enums: Iterable[str]):
        self._flags: dict[str, Optional[bool | str]] = {}
        self._categories: dict[str, str] = {}
        for k, vs in enums.items():
            self._flags[k] = None
            for v in vs:
                self._categories[v] = k
        for f in booleans:
            self._flags[f] = False
            self._categories[f] = ...

    @overload
    def __call__(self, flags: Optional[str], sep: Optional[str] = None): ...
    @overload
    def __call__(self, flags: Iterable[str]): ...
    def __call__(self, flags: Iterable[str], sep: str = None):
        if flags is None:
            flags = ()
        elif isinstance(flags, str):
            flags = flags.split(sep)
        parsed = self._flags.copy()
        for flag in flags:
            if cat := self._categories.get(flag):
                if cat == ...:
                    parsed[flag] = True
                elif parsed[cat] is not None:
                    raise ValueError(
                        f'flag "{flag}" and "{parsed[cat]}" cannot be used together.'
                    )
                else:
                    parsed[cat] = flag
            else:
                raise ValueError(f"unknown flag: {flag}.")
        return parsed


class TagParser(Protocol):
    """
    Tag parser protocol.
    """

    def __call__(
        self,
        *,
        key: Optional[str],
        value: Optional[Any],
        tag: Optional[str],
        tags: Optional[dict[str, Optional[str]]],
        local: Optional[dict],
        path: Optional[str],
    ) -> tuple[str, Any]:
        """
        Parameters
        ----------
        key: str, optional
            The current key.
        value: Any, optional
            The current value .
        tag: str, optional
            The value of the current tag.
        tags: dict[str, Optional[str]], optional
            All parsed tags of the current key.
        local: dict, optional
            All parsed items in the current dictionary.
        path: str or None, optional
            The path to the current config file.

        Returns
        -------
        tuple[str, Any]
            The key and value after parsing.
        """
        ...


@dataclass
class _TagParserWrapper:
    func: TagParser
    keys: frozenset[str]

    def __call__(self, *arg, **kwargs):
        return self.func(*arg, **{k: v for k, v in kwargs.items() if k in self.keys})

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return MethodType(self, instance)


def _tag_parser(func: Callable[P, T]) -> Callable[P, T]:
    kwargs = set()
    try:
        sig = inspect.signature(func)
    except ValueError:
        return func
    for k, v in sig.parameters.items():
        match v.kind:
            case ParKind.POSITIONAL_OR_KEYWORD | ParKind.KEYWORD_ONLY:
                kwargs.add(k)
            case ParKind.VAR_KEYWORD:
                return func
    return _TagParserWrapper(func=func, keys=frozenset(kwargs))


class TypeParser:  # tag: <type>
    def __init__(self, base: str = None):
        self.base = base

    def _new(self, tag: Optional[str], value: Any):
        # import module
        if tag is None:
            if not isinstance(value, str):
                raise ValueError("type name must be a string.")
            fullname = value
        else:
            fullname = tag
        attrs = fullname.split("::", 1)
        match len(attrs):
            case 1:
                mod, attrs = None, attrs[0]
            case 2:
                mod, attrs = attrs
            case _:
                raise ImportError(f'invalid import format "{fullname}".')
        if mod is None:
            if self.base is None:
                mod = "builtins"
            else:
                mod = self.base
        elif self.base is not None:
            mod = f"{self.base}.{mod}"

        obj = importlib.import_module(mod)
        attrs = attrs.split(".")
        if any(attrs):
            for attr in attrs:
                obj = getattr(obj, attr)

        if tag is None:
            return obj

        # parse args and kwargs
        if value is None:
            kwargs = {}
            args = []
        elif isinstance(value, dict):
            kwargs = value.copy()
            args = kwargs.pop(None, [])
        else:
            kwargs = {}
            args = value
        if not isinstance(args, list):
            args = [args]
        return obj(*args, **kwargs)

    @_tag_parser
    def __call__(self, tag: Optional[str], key: str, value: Any):
        return key, self._new(tag, value)


class KeyTypeParser(TypeParser):  # tag: <key>
    @_tag_parser
    def __call__(self, tag: Optional[str], key: str, value: Any):
        return self._new(tag, key), value


class AttrParser:  # tag: <attr>
    @_tag_parser
    def __call__(self, tag: Optional[str], key: str, value):
        for attr in tag.split("."):
            value = getattr(value, attr)
        return key, value


ExtendMethod = Callable[[Any, Any], Any]
"""
~typing.Callable[[Any, Any], Any]: A method to merge two values into one.
"""


class RecursiveExtend:
    def __init__(self, op: ExtendMethod):
        self.op = op

    def __call__(self, v1, v2):
        if isinstance(v1, dict) and isinstance(v2, dict):
            new = v1.copy()
            for k, v in v2.items():
                if k in v1:
                    new[k] = self(v1[k], v)
                else:
                    new[k] = v
        else:
            return self.op(v1, v2)
        return new


class ExtendParser:  # tag: <extend>
    methods = {
        None: RecursiveExtend(op.add),
        "add": RecursiveExtend(op.add),
        "and": op.and_,
        "or": op.or_,
    }
    reserved_methods = check_reserved("extend methods", methods)

    def __init__(self, methods: dict[str, ExtendMethod]):
        self.reserved_methods(methods)
        if methods:
            self.methods = methods | self.methods

    @_tag_parser
    def __call__(self, local: dict, tag: Optional[str], key: str, value):
        if key not in local:
            return key, value
        if tag not in self.methods:
            raise ValueError(f"unknown method: {tag}.")
        return key, self.methods[tag](local[key], value)


class VariableParser:  # tag: <var> <ref>
    __flags = FlagParser(method=("copy", "deepcopy"))

    def __init__(self):
        self.local = {}

    @staticmethod
    def _get_name(*args):
        for arg in args:
            if isinstance(arg, str):
                return arg
        raise ValueError("variable name cannot be None.")

    @_tag_parser
    def var(self, tag: Optional[str], key: str, value):
        self.local[self._get_name(tag, key)] = value
        return key, value

    @_tag_parser
    def ref(self, tag: Optional[str], key: str, value):
        try:
            name = self._get_name(value, key)
            obj = self.local[name]
            match self.__flags(tag)["method"]:
                case "copy":
                    obj = copy.copy(obj)
                case "deepcopy":
                    obj = copy.deepcopy(obj)
            return key, obj
        except KeyError:
            raise NameError(f'name "{name}" is not defined.') from None
        except Exception:
            raise


class FileParser:  # tag: <file>
    __flags = FlagParser("nocache", "nobuffer", path=("relative", "absolute"))

    @_tag_parser
    def __call__(self, path: str, tag: Optional[str], key: str, value):
        flags = self.__flags(tag, sep="|")
        use_cache = not flags["nocache"]
        obj = next(
            load_url(
                partial(
                    ConfigParser.io.load,
                    use_cache=use_cache,
                    use_buffer=not flags["nobuffer"],
                ),
                next(resolve_path(path, flags["path"], value)),
                parse_query=False,
            )
        )
        if use_cache:
            obj = copy.deepcopy(obj)
        return key, obj


class MapParser:  # tag: <map>
    @_tag_parser
    def __call__(self, key, value: list[dict]):
        return key, {v["key"]: v["val"] for v in value}


class _Parser:
    re_match = re.compile(r"(?P<key>.*?)\s*(?P<tags>(\<[^><]*\>\s*)*)\s*")
    re_split = re.compile(r"\<(?P<tag>[^><]*)\>")

    def __init__(self, path: Optional[str], custom: _ParserInitializer):
        self.path = path
        self.custom = custom

    def extract_tags(self, raw: Optional[str]) -> tuple[Optional[str], _MatchedTags]:
        if raw is None:
            return None, _NoTag
        matched = self.re_match.fullmatch(raw)
        if not matched:
            return raw, _NoTag
        tags, spans = [], []
        start = matched.start("tags")
        for tag in self.re_split.finditer(matched["tags"]):
            k = tag["tag"].split("=")
            span = tag.start() + start, tag.end() + start
            if len(k) == 1:
                v = None
            elif len(k) == 2:
                v = k[1]
            else:
                raise _error_msg(
                    error=ValueError("invalid tag format."),
                    path=self.path,
                    key=raw,
                    span=span,
                )
            tags.append((k[0], v))
            spans.append(span)
        key = matched["key"]
        if not key or key == "~":
            key = None
        return key, _MatchedTags(tags, self, {"key": raw, "spans": spans})

    def dict(
        self,
        pairs: dict[str, Any] | list[tuple],
        singleton: bool = False,
        result: dict = None,
    ):
        if result is None:
            result = {}
        if isinstance(pairs, dict):
            pairs = [*pairs.items()]
        while pairs:
            k, v = pairs.pop(0)
            if k is None or isinstance(k, str):
                k = self.extract_tags(k)
            key, tags = k
            value = self.eval(tags, v)
            try:
                match tags.unique:
                    case (_ReservedTag.select, tag):
                        pairs = self.select(tag, value) + pairs
                    case (_ReservedTag.include, tag):
                        tags.include(tag, value, result)
                    case (_ReservedTag.patch, tag):
                        tags.patch(tag, key, value)
            except SyntaxError:
                raise
            except Exception as e:
                raise _error_msg(
                    error=e,
                    path=self.path,
                    key=tags.debug["key"],
                    span=tags.debug["spans"]["unique"],
                    value=v,
                ) from None

            if tags.unique is None:
                self.setitem(tags, key, value, result)
        if (
            singleton
            and len(result) == 1
            and None in result
            and not tags.has(_ReservedTag.literal)
        ):
            return result[None]
        return result

    def list(self, data: list[Any]):
        parsed = []
        for v in data:
            if isinstance(v, dict):
                v = self.dict(v, singleton=True)
            elif isinstance(v, list):
                v = self.list(v)
            parsed.append(v)
        return parsed

    def eval(self, tags: _MatchedTags, v: Any):
        if tags.has(_ReservedTag.code):
            try:
                v = eval(v, None, self.custom.vars.local)
            except Exception as e:
                raise _error_msg(
                    error=e,
                    path=self.path,
                    key=tags.debug["key"],
                    span=tags.debug["spans"]["flags"][_ReservedTag.code],
                    value=v,
                ) from None
        return v

    def setitem(self, tags: _MatchedTags, key: str, value: Any, local: dict):
        if (
            self.custom.nested
            and key is not None
            and not tags.has(_ReservedTag.literal)
        ):
            keys = [*_split_path(key)]
            local = SimpleTree.init(local, keys[:-1])
            key = keys[-1]
        if isinstance(value, dict):
            value = self.dict(value)
        elif isinstance(value, list):
            value = self.list(value)
        key, value = tags.apply(
            key=key,
            value=value,
            local=local,
        )
        if not tags.has(_ReservedTag.discard):
            local[key] = value

    def select(self, tag: str, cases: list[dict[str]]):
        match tag:
            case None | "first":
                result = None
            case "all":
                result = []
            case _:
                raise ValueError(f"unknown method: {tag}.")
        for case in cases:
            pairs = []
            decision = False
            for k, v in case.items():
                key, tags = self.extract_tags(k)
                pair = ((key, tags), v)
                if tags.has(_ReservedTag.case):
                    value = bool(self.dict([pair])[key])
                    match operator := tags.get(_ReservedTag.case):
                        case None:
                            decision = value
                        case "or":
                            decision = decision | value
                        case "and":
                            decision = decision & value
                        case "xor":
                            decision = decision ^ value
                        case _:
                            raise _error_msg(
                                error=ValueError(f"unknown operator: {operator}."),
                                path=self.path,
                                key=tags.debug["key"],
                                span=tags.debug["spans"]["flags"][_ReservedTag.case],
                                value=v,
                            )
                else:
                    pairs.append(pair)
            if decision:
                if result is None:
                    return pairs
                result.extend(pairs)
        if result is None:
            return []
        return result


class _ParserInitializer:
    static_parsers = {
        _ReservedTag.file: FileParser(),
        _ReservedTag.type: TypeParser(),
        _ReservedTag.attr: AttrParser(),
        _ReservedTag.map: MapParser(),
    }
    reserved_tags = check_reserved("tags", _ReservedTag.__members__)

    def __init__(self, parser: ConfigParser):
        self.reserved_tags(parser.tag_parsers)
        extend_parser = ExtendParser(parser.extend_methods)

        self.nested = parser.nested
        self.vars = VariableParser()
        self.patches = PatchedLoader(parser)
        self.parsers = (
            {
                k: v if v is None else _tag_parser(v)
                for k, v in parser.tag_parsers.items()
            }
            | self.static_parsers
            | {
                _ReservedTag.extend: extend_parser,
                _ReservedTag.var: self.vars.var,
                _ReservedTag.ref: self.vars.ref,
            }
        )

    def parse(
        self,
        *path_or_dict: ConfigSource,
        result: Optional[dict] = None,
    ) -> dict:
        if result is None:
            result = {}
        for configs in path_or_dict:
            path = None
            if not isinstance(configs, dict):
                path = fspath(configs)
                configs = load_url(self.patches.load, path)
            else:
                configs = (configs,)
            parser = _Parser(path, self)
            for config in configs:
                if not isinstance(config, dict):
                    if config is None:
                        content = "None (empty file)"
                    else:
                        content = type(config).__name__
                    raise ValueError(
                        f'Config must be a dict, got a {content} in "{path}"'
                    )
                parser.dict(config, result=result)
        return result


@dataclass(kw_only=True)
class ConfigParser:
    """
    A customizable config parser.

    Parameters
    ----------
    nested : bool, optional, default=True
        Parse dot-separated keys into nested dicts.
    tag_parsers : dict[str, Optional[TagParser]], optional
        Custom tags.
    extend_methods : dict[str, ExtendMethod], optional
        Custom <extend> methods.
    patch_actions : dict[str, PatchAction], optional
        Custom patch actions.

    """

    nested: bool = True
    tag_parsers: dict[str, Optional[TagParser]] = field(default_factory=dict)
    extend_methods: dict[str, ExtendMethod] = field(default_factory=dict)
    patch_actions: dict[str, PatchAction] = field(default_factory=dict)

    io = FileLoader()
    """
    FileLoader: A config file loader with a shared cache.
    """

    def __call__(
        self, *path_or_dict: ConfigSource, result: Optional[dict] = None
    ) -> dict:
        """
        Load configs from multiple sources.

        Parameters
        ----------
        *path_or_dict : ConfigSource
            Dictionaries or paths to config files.
        result : dict, optional
            If provided, the configs will be loaded into this dict.

        Returns
        -------
        dict
            The loaded configs.
        """
        return _ParserInitializer(self).parse(*path_or_dict, result=result)
