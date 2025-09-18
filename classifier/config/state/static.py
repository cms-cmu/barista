from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.storage.eos import EOS, PathLike


class GitRepo:
    user: str = "cms-cmu"
    repo: str = "coffea4bees"
    branch: str = "master"
    url: str = f"https://gitlab.cern.ch/{user}/{repo}/-/tree/{branch}/"

    _local: EOS = None
    _file: tuple[str, ...] = ("python", "classifier", "config", "state", "__init__.py")

    @classmethod
    def get_url(cls, path: PathLike) -> str:
        from src.storage.eos import EOS

        if cls._local is None:
            local = EOS(__file__)
            for i in range(len(cls._file)):
                if local.name != cls._file[-i - 1]:
                    i -= 1
                    break
                local = local.parent
            cls._local = local, cls.url + "".join(
                map(lambda x: x + "/", cls._file[: -i - 1])
            )
        local, url = cls._local
        path = EOS(path)
        if not path.isin(local):
            return str(path)
        path = path.relative_to(local)
        return f"{url}{path}"


class MonitorExtensions:
    backends: tuple[str, ...] = ("console",)
    components: tuple[str, ...] = ("logging", "usage", "progress")
