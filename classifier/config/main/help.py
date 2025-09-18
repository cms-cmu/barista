import inspect
import re
from pathlib import Path
from textwrap import indent

import fsspec

from classifier.task import ArgParser, EntryPoint, Task, main, parse
from classifier.task.special import Deprecated, WorkInProgress
from classifier.task.task import _INDENT
from classifier.test.utils.import_check import walk_packages

from .. import setting as cfg
from ..state import Flags

_NOTES = [
    f"A special task/argument [blue]{main._FROM}[/blue]/[blue]{main._DASH}{main._FROM}[/blue] [yellow]file \[file ...][/yellow] can be used to load and merge workflows from files. If an option is marked as {parse.EMBED}, it can directly read the jsonable object embedded in the workflow configuration file.",
    f"A special task/argument [blue]{main._TEMPLATE}[/blue]/[blue]{main._DASH}{main._TEMPLATE}[/blue] [yellow]formatter file \[file ...][/yellow] can be used to load and merge workflows and replace the keys by the formatter.",
    f"A special argument [blue]{main._DASH}{main._FLAG}[/blue] \[flag ...] can be used to setup system level flags, which has the highest priority and will be available right after the argument parsing. The flags are mainly for internal use. For other purposes, use [blue]{main._DASH}{main.TaskOptions.setting.name}[/blue] instead.",
]

_WORKINPROGRESS = "[red]\[Work In Progress][/red]"
_DEPRECATED = "[orange1]\[Deprecated][/orange1]"
_TEST = "[yellow]\[Test][/yellow]"


def _print_mod(cat: str, imp: str, opts: list[str | dict], newline: str = "\n"):
    if cat is None:
        output = [f"[blue]{imp}[/blue]"]
    else:
        output = [f"[blue]{main._DASH}{cat}[/blue] [green]{imp}[/green]"]
    current = []
    for opt in opts + [None]:
        if (isinstance(opt, str) and opt.startswith(main._DASH)) or (opt is None):
            if current:
                output.append(indent(f"[yellow]{' '.join(current)}[/yellow]", _INDENT))
            current.clear()
        current.append(opt)
    return newline.join(output)


def __walk(cat: str, ctx: main._ModCtx):
    for pkg in walk_packages(
        "",
        Path(__file__).resolve().parents[3]
        / "/".join(EntryPoint._fetch_config(cat, ctx)),
        skip_private=True,
    ):
        yield pkg, ctx


def _walk_configs(cat: str, test: bool = False):
    yield from __walk(cat, main._ModCtx())
    if test:
        yield from __walk(cat, main._ModCtx(test=True))


class Main(main.Main):
    _no_state = True

    _keys = " ".join(f"{main._DASH}{k}" for k in EntryPoint._tasks)
    argparser = ArgParser(
        prog="help",
        description="Print help information.",
        workflow=[
            ("main", "[blue]task.help()[/blue] print help information"),
        ],
    )
    argparser.add_argument(
        "--all",
        action="store_true",
        help=f"list all available modules for [blue]{_keys}[/blue]",
    )
    argparser.add_argument(
        "--html", action="store_true", help='write "help.html" to output directory'
    )
    argparser.add_argument(
        "--filter",
        type=re.compile,
        help="a Python style regular expression to filter modules when [yellow]--all[/yellow] is set",
        default=".*",
    )
    argparser.add_argument(
        "--no-regular",
        action="store_true",
        help="skip regular tasks",
    )
    argparser.add_argument(
        "--wip",
        "--work-in-progress",
        action="store_true",
        help=f"list {_WORKINPROGRESS} tasks",
    )
    argparser.add_argument(
        "--deprecated",
        action="store_true",
        help=f"list {_DEPRECATED} tasks",
    )

    @classmethod
    def prelude(cls):
        cfg.Analysis.enable = False
        cfg.Monitor.enable = False
        cfg.Multiprocessing.context_library = "builtins"

    def __init__(self):
        super().__init__()
        from rich.console import Console

        self._console = Console(record=True, markup=True)

    def _print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)

    def _print_help(self, task: type[Task], depth: int = 1):
        self._print(indent(task.help(), _INDENT * depth))

    def _check_special(
        self,
        cls: type,
        depth: int = 0,
        force: bool = False,
        ctx: main._ModCtx = None,
    ):
        to_print = True
        labels = []
        if isinstance(cls, type):
            if issubclass(cls, WorkInProgress):
                labels.append(indent(_WORKINPROGRESS, _INDENT * depth))
                if not self.opts.wip:
                    to_print = False
            if issubclass(cls, Deprecated):
                labels.append(indent(_DEPRECATED, _INDENT * depth))
                if not self.opts.deprecated:
                    to_print = False
        if ctx is not None:
            if ctx.test:
                labels.append(indent(_TEST, _INDENT * depth))
                if not Flags.test:
                    to_print = False
        if self.opts.no_regular and len(labels) == 0:
            to_print = False
        if to_print or force:
            for label in labels:
                self._print(label)
        return to_print

    def run(self, parser: EntryPoint):
        import rich.terminal_theme as themes

        tasks = parser.args["main"]
        self._print("[orange3]\[Usage][/orange3]")
        self._print(
            " ".join(
                [
                    f"{parser.entrypoint} [blue]task[/blue] [yellow]\[args ...][/yellow]",
                    *(
                        f"[blue]{main._DASH}{k}[/blue] [green]module.class[/green] [yellow]\[args ...][/yellow]"
                        for k in parser._tasks
                    ),
                ]
            )
        )
        self._print(
            indent(
                f'[blue]task[/blue] = [blue]{"|".join(parser._mains)}[/blue]', _INDENT
            )
        )
        self._print(
            indent(
                f'[green]module.class[/green] = [purple]from[/purple] [green]{main._CLASSIFIER}.{main._CONFIG}.\[{"|".join(parser._tasks)}].module[/green] [purple]import[/purple] [green]class[/green]',
                _INDENT,
            )
        )
        self._print("\n[orange3]\[Notes][orange3]")
        for i, note in enumerate(_NOTES, 1):
            self._print(f"#{i}")
            self._print(indent(note, _INDENT))
        self._print("\n[orange3]\[Tasks][orange3]")
        for task in parser._mains:
            if self.opts.filter.fullmatch(task) is not None:
                _, cls, _ = parser._fetch_module(f"{task}.Main", main._MAIN)
                if self._check_special(cls):
                    self._print(f"[blue]{task}[/blue]")
                    self._print_help(cls)
        self._print(f"[blue]{main._DASH}{main._FLAG}[/blue]")
        for k in Flags.__annotations__:
            self._print(indent(f"[yellow]{k}[/yellow] = {getattr(Flags, k)}", _INDENT))
        if not self.opts.all:
            self._print("\n[orange3]\[Options][orange3]")
            self._print(_print_mod(None, "task", tasks[1]))
            for cat in parser._tasks:
                target = EntryPoint._tasks[cat]
                for imp, opts in parser.args[cat]:
                    mod, cls, (modname, clsname) = parser._fetch_module(imp, cat)
                    self._check_special(cls, force=True)
                    self._print(_print_mod(cat, imp, opts))
                    if mod is None:
                        self._print(
                            indent(f'[red]Module "{modname}" not found[/red]', _INDENT)
                        )
                    elif cls is None:
                        self._print(
                            f'[red]Class "{clsname}" not found in module "{modname}"[/red]'
                        )
                    else:
                        if not issubclass(cls, target):
                            self._print(
                                f'[red]Class "{clsname}" is not a subclass of {target.__name__}[/red]'
                            )
                        else:
                            self._print_help(cls)
        if self.opts.all:
            self._print("\n[orange3]\[Modules][/orange3]")
            for cat in parser._tasks:
                target = EntryPoint._tasks[cat]
                self._print(f"[blue]{main._DASH}{cat}[/blue]")
                for imp, ctx in _walk_configs(cat, Flags.test):
                    if imp:
                        imp = f"{imp}."
                    _imp = f"{imp}*"
                    mod, _, (modname, _) = parser._fetch_module(
                        _imp, cat, force_ctx=[ctx]
                    )
                    classes = {}
                    if mod is not None:
                        for name, obj in inspect.getmembers(mod, inspect.isclass):
                            if (
                                issubclass(obj, target)
                                and not main._is_private(name)
                                and obj.__module__.startswith(modname)
                            ):
                                fullname = f"{imp}{name}"
                                if self.opts.filter.fullmatch(fullname) is not None:
                                    classes[fullname] = obj
                    if classes:
                        for cls in classes:
                            if self._check_special(classes[cls], 1, ctx=ctx):
                                self._print(indent(f"[green]{cls}[/green]", _INDENT))
                                self._print_help(classes[cls], 2)
        if self.opts.html:
            with fsspec.open(cfg.IO.output / "help.html", "wt") as f:
                f.write(self._console.export_html(theme=themes.MONOKAI))
