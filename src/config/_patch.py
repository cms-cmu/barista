from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Protocol, cast

from ._io import _enter_path, _parse_url, _split_path
from ._utils import block_divider, block_indent, check_reserved, format_repr

if TYPE_CHECKING:
    from ._parser import ConfigParser


def _remove_last(seq: list, value: Any):
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] == value:
            del seq[i]
            return


class _error_msg:
    value_lines = 5

    def __new__(
        cls,
        action: str,
        error: Exception,
        target: str,
        source: str,
        name: str,
        value=...,
    ):
        block_start, block_end = block_divider()
        if value is ...:
            value_info = ""
        else:
            value_info = ":\n"
            value_info += block_indent(format_repr(value, cls.value_lines), "    ")
        return RuntimeError(
            f"""When patching:
  {target}
the following exception occurred:
  {type(error).__name__}:
{block_indent(str(error), "    ")}
during the {action} of action:
{block_start}
{source}
  {name or "<patch>"}{value_info}
{block_end}
"""
        )


class PatchAction(Protocol):
    """
    Patch action protocol.
    """

    def __call__(self, **kwargs) -> Callable[[dict], None]: ...


@dataclass
class _TargetAction:
    target: str

    def __post_init__(self):
        self.parts = [*_split_path(self.target)]

    def _enter_parent(self, data: dict) -> tuple[dict, str] | tuple[list, int]:
        data = _enter_path(data, self.parts[:-1])
        key = self.parts[-1]
        if isinstance(data, list):
            key = int(key)
        return data, key


@dataclass
class _TargetValueAction(_TargetAction):
    value: Any


class MkdirAction(_TargetAction):
    def __call__(self, data: dict):
        for part in self.parts:
            if part not in data:
                data[part] = {}
            data = data[part]


class UpdateAction(_TargetValueAction):
    def __call__(self, data: dict):
        _enter_path(data, self.parts).update(self.value)


class PopAction(_TargetAction):
    def __call__(self, data: dict):
        current, key = self._enter_parent(data)
        current.pop(key)


class SetAction(_TargetValueAction):
    def __call__(self, data: dict):
        current, key = self._enter_parent(data)
        current[key] = self.value


class InsertAction(_TargetValueAction):
    def __call__(self, data: dict):
        current, idx = self._enter_parent(data)
        current.insert(idx, self.value)


class AppendAction(_TargetValueAction):
    def __call__(self, data: dict):
        _enter_path(data, self.parts).append(self.value)


class ExtendAction(_TargetValueAction):
    def __call__(self, data: dict):
        _enter_path(data, self.parts).extend(self.value)


class _PatchLayer:
    def __init__(
        self, patches: list[dict], patch_actions: dict[str, PatchAction], debug: dict
    ):
        debug_actions = defaultdict(list)
        self.debug = debug | {"actions": debug_actions}
        self.patches = defaultdict[str, list[tuple[list[str], list[Callable]]]](list)

        for patch in patches:
            parsed, url = _parse_url(patch["path"])
            parts = [*_split_path(parsed.fragment)]
            actions = []
            for raw_action in cast(list[dict], patch["actions"]):
                try:
                    kwargs = raw_action.copy()
                    name = kwargs.pop("action")
                    actions.append(patch_actions[name](**kwargs))
                except Exception as e:
                    raise _error_msg(
                        action="initialization",
                        error=e,
                        target=patch["path"],
                        source=self.debug["source"],
                        name=self.debug.get("name"),
                        value=raw_action,
                    ) from None
            self.patches[url].append((parts, actions))
            debug_actions[url].append(patch)

    def __call__(self, url: str, data: dict):
        for i, (path, actions) in enumerate(self.patches[url]):
            j = None
            try:
                current = _enter_path(data, path)
                for j, action in enumerate(actions):
                    action(current)
            except Exception as e:
                raise _error_msg(
                    action="application",
                    error=e,
                    target=self.debug["actions"][url][i]["path"],
                    source=self.debug["source"],
                    name=self.debug.get("name"),
                    value=(
                        ...
                        if j is None
                        else self.debug["actions"][url][i]["actions"][j]
                    ),
                ) from None


class PatchedLoader:
    static_actions = {
        "mkdir": MkdirAction,  # dict
        "update": UpdateAction,  # dict
        "pop": PopAction,  # dict, list
        "set": SetAction,  # dict, list
        "insert": InsertAction,  # list
        "append": AppendAction,  # list
        "extend": ExtendAction,  # list
    }
    reserved_actions = check_reserved("patch actions", static_actions)

    def __init__(self, parser: ConfigParser):
        self.reserved_actions(parser.patch_actions)
        self.actions = parser.patch_actions | self.static_actions
        self.loader = parser.io.load

        self.registered = dict[int | str, _PatchLayer]()
        self.installed = defaultdict[str, int](int)
        self.patched = defaultdict[str, list[int | str]](list)
        self.cache = dict[str, Any]()

    def load(self, url: str):
        if url not in self.cache:
            data = self.loader(url)
            if patches := self.patched[url]:
                data = copy.deepcopy(data)
                for patch in patches:
                    self.registered[patch](url, data)
            self.cache[url] = data
        return self.cache[url]

    def register(self, patches: list[dict], name: Optional[str], source: str):
        debug = {"source": source}
        if name is None:
            name = len(self.registered)
        else:
            debug["name"] = name
        if name not in self.registered:
            self.registered[name] = _PatchLayer(patches, self.actions, debug)
            self.install((name,))

    def install(self, names: Iterable[str]):
        for name in names:
            for url in self.registered[name].patches:
                self.patched[url].append(name)
                self.cache.pop(url, None)
            self.installed[name] += 1

    def uninstall(self, names: Iterable[str]):
        for name in names:
            if self.installed[name] == 0:
                continue
            for url in self.registered[name].patches:
                _remove_last(self.patched[url], name)
                self.cache.pop(url, None)
            self.installed[name] -= 1
