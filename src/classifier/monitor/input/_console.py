from threading import Lock

import readchar
from rich.text import Text


class Input:
    def __init__(self, title: str, multiline: bool = False, password: bool = False):
        self.text = ""

        self._title = title
        self._password = password
        self._multiline = multiline and not self._password

        self.__lock = Lock()
        self.__inputting = False
        self.__cursor = 0

    @property
    def is_inputting(self):
        return self.__inputting

    def get(self):
        self.text = ""
        self.__cursor = 0
        self.__inputting = True
        while True:
            try:
                self.__char(readchar.readkey())
            except KeyboardInterrupt:
                break
        self.__inputting = False
        return self

    def __insert(self, char: str):
        self.text = self.text[: self.__cursor] + char + self.text[self.__cursor :]

    def __char(self, char: str):
        with self.__lock:
            match char:
                case readchar.key.HOME:
                    self.__cursor = 0
                case readchar.key.END:
                    self.__cursor = len(self.text)
                case readchar.key.BACKSPACE:
                    if self.__cursor > 0:
                        self.text = (
                            self.text[: self.__cursor - 1] + self.text[self.__cursor :]
                        )
                        self.__cursor -= 1
                case readchar.key.DELETE:
                    if self.__cursor < len(self.text):
                        self.text = (
                            self.text[: self.__cursor] + self.text[self.__cursor + 1 :]
                        )
                case readchar.key.ENTER:
                    if self._multiline:
                        self.__insert("\n")
                        self.__cursor += 1
                    else:
                        raise KeyboardInterrupt
                case readchar.key.CTRL_C:
                    raise KeyboardInterrupt
                case readchar.key.LEFT:
                    self.__cursor -= 1
                case readchar.key.RIGHT:
                    ...
                case _:
                    if char.isprintable():
                        self.__insert(char)
                        self.__cursor += 1

            if self.__cursor > len(self.text):
                self.__cursor = len(self.text)
            if self.__cursor < 0:
                self.__cursor = 0

    def __rich_console__(self, console, options):
        if self.__inputting:
            with self.__lock:
                title = Text.from_markup(f"[yellow]{self._title}> [/yellow]")
                raw = self.text if not self._password else "*" * len(self.text)
                text = Text(raw[: self.__cursor], style="default", overflow="fold")
                if self.__cursor < len(self.text):
                    cursor = raw[self.__cursor]
                    if not cursor.isprintable():
                        cursor = " " + cursor
                    text.append(cursor, style="reverse")
                    text.append(raw[self.__cursor + 1 :])
                else:
                    text.append(" ", style="reverse")
                if self._multiline:
                    yield title
                    yield text
                else:
                    yield title + text
