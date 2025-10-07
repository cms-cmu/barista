import re
from collections import defaultdict
from pathlib import Path


class SimpleImporter:
    def __init__(self, file: str, cache: dict[str, dict[str, str]] = None):
        self._cache = defaultdict(dict, cache or {})
        self._file = file

    def _read(self, file: str, ext: str, **kwargs) -> str:
        if file not in self._cache[ext]:
            with open(Path(self._file).parent / ext / f"{file}.{ext}") as f:
                self._cache[ext][file] = f.read()
        content = self._cache[ext][file]
        if kwargs:
            kwargs = {"{{ " + k + " }}": v for k, v in kwargs.items()}
            content = re.sub(
                "|".join(re.escape(k) for k in kwargs),
                lambda match: kwargs.get(match.group(0)),
                content,
            )
        return content

    def js(self, file: str, **kwargs):
        return self._read(file, "js", **kwargs)

    def html(self, file: str, **kwargs):
        return self._read(file, "html", **kwargs)
