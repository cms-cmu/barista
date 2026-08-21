from __future__ import annotations
import logging
import os
import sys

class CustomFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[1;36m',    # Bold Cyan
        'INFO': '\033[1;34m',     # Bold Blue (matching run_container)
        'WARNING': '\033[1;33m',  # Bold Yellow
        'ERROR': '\033[1;31m',    # Bold Red
        'CRITICAL': '\033[1;41m', # Bold Red Background
    }
    DIM = '\033[90m'              # Dim / Gray
    RESET = '\033[0m'

    def __init__(self, use_color: bool | None = None):
        super().__init__()
        if use_color is None:
            self.use_color = os.environ.get("NO_COLOR") not in ("1", "true", "True")
        else:
            self.use_color = use_color

    def format(self, record):
        asctime = self.formatTime(record, "%y/%m/%d %H:%M:%S")
        level_tag = f"[{record.levelname}]"

        if self.use_color:
            color = self.COLORS.get(record.levelname, '')
            colored_level = f"{color}{level_tag}{self.RESET}"
            source = f"{self.DIM}({record.filename}:{record.lineno}){self.RESET}"
            time_str = f"{self.DIM}[{asctime}]{self.RESET}"
        else:
            colored_level = level_tag
            source = f"({record.filename}:{record.lineno})"
            time_str = f"[{asctime}]"

        header = f"{colored_level} {time_str} {source}"
        message = record.getMessage()
        if '\n' in message:
            indent = " " * 7
            message = ("\n" + indent).join(message.splitlines())
        return f"{header} {message}"
