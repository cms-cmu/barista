from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from rich import terminal_theme as themes
from rich._log_render import LogRender
from rich.console import Console
from rich.logging import RichHandler as _RichHandler
from rich.protocol import is_renderable
from rich.table import Table
from rich.text import Text, TextType

from ...utils import noop
from ..backends import Platform
from ..core import CLIENT_NAME_WIDTH
from ._redirect import MultiPlatformLogRecord


class ConsoleLogRender(LogRender):
    _url_pattern = re.compile(r"^([\w]+)://.*$")
    _https = frozenset({"https", "http"})

    @classmethod
    def _parse_link(cls, path: str, no: str):
        if path:
            if (match := cls._url_pattern.match(path)) is None:
                path = f"file://{path}"
            elif no and (match.groups()[0] in cls._https):
                no = f"L{no}"
        return path, no

    def __call__(
        self,
        console: Console,
        message: Text,
        log_time: datetime = None,
        time_format: str = None,
        level: TextType = "",
        path: str = None,
        line_no: int = None,
        link_path: str = None,
        name: str = None,
        args: tuple = None,
    ) -> Table:
        from rich.containers import Renderables

        output = Table.grid(padding=(0, 1))
        output.expand = True
        if self.show_time:
            output.add_column(style="log.time")
        if name is not None:
            output.add_column(style="repr.number", width=CLIENT_NAME_WIDTH + 2)
        if self.show_level:
            output.add_column(style="log.level", width=self.level_width)
        output.add_column(ratio=1, style="log.message", overflow="fold")
        if self.show_path and path:
            output.add_column(style="log.path")
        row = []
        if self.show_time:
            log_time = log_time or console.get_datetime()
            time_format = time_format or self.time_format
            if callable(time_format):
                log_time_display = time_format(log_time)
            else:
                log_time_display = Text(log_time.strftime(time_format))
            if log_time_display == self._last_time and self.omit_repeated_times:
                row.append(Text(" " * len(log_time_display)))
            else:
                row.append(log_time_display)
                self._last_time = log_time_display
        if name is not None:
            row.append(Text(f"[{name:>{CLIENT_NAME_WIDTH}}]"))
        if self.show_level:
            row.append(level)
        row.append(message)
        if self.show_path and path:
            link_path, line_no = self._parse_link(link_path, line_no)
            path_text = Text()
            path_text.append(path, style=f"link {link_path}" if link_path else "")
            if line_no:
                path_text.append(":")
                path_text.append(
                    f"{line_no}",
                    style=f"link {link_path}#{line_no}" if link_path else "",
                )
            row.append(path_text)
        output.add_row(*row)
        if args is not None:
            renderables = []
            for arg in args:
                if is_renderable(arg):
                    renderables.append(arg)
                else:
                    renderables.append(Text(repr(arg)))
            if renderables:
                output = Renderables([output, *renderables])
        return output


class ConsoleHandler(_RichHandler):
    __platform__ = Platform.console

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_render = ConsoleLogRender(
            show_time=True,
            show_level=True,
            show_path=True,
            time_format="[%x %X]",
            omit_repeated_times=True,
            level_width=None,
        )

    def render(self, *, record: MultiPlatformLogRecord, traceback, message_renderable):
        if isinstance(message_renderable, Platform):
            message = None
        else:
            message = message_renderable
        args = [*(record.args or ())]
        if traceback:
            args.append(traceback)
        log_renderable = self._log_render(
            self.console,
            message,
            log_time=datetime.fromtimestamp(record.created),
            time_format=None if self.formatter is None else self.formatter.datefmt,
            level=self.get_level_text(record),
            path=Path(record.pathname).name,
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
            name=record.name,
            args=args,
        )
        return log_renderable

    def render_message(self, record, message):
        if isinstance(message, Platform):
            return message
        return super().render_message(record, str(message))

    @classmethod
    def new(cls, console: Console):
        return cls(
            markup=True,
            rich_tracebacks=True,
            console=console,
        )


class ConsoleDump:
    console: Console

    @classmethod
    def init(cls):
        cls.console = Console(record=True, markup=True, file=noop)
        cls.handler = ConsoleHandler.new(cls.console)

    @classmethod
    def serialize(cls):
        return cls.console.export_html(theme=themes.MONOKAI).encode()
