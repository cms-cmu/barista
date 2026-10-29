# TODO docstring
"""
An interface to operate on both local filesystem and EOS (or other XRootD supported system).

.. todo::
    - Use :func:`os.path.normpath`, :func:`glob.glob`
"""
from __future__ import annotations

import importlib
import os
import pickle
import re
import tempfile
from datetime import datetime
from pathlib import PurePosixPath as Path
from subprocess import PIPE, CalledProcessError, check_output
from typing import Any, Generator, Literal

from ..utils import arg_set
from ..utils.string import ensure
from ..utils.wrapper import retry

__all__ = ["EOS", "PathLike", "EOSError", "save", "load"]


def _is_missing(stderr) -> bool:
    """Whether an xrdfs error says the path is not there ([3011] / "No such file")."""
    msg = stderr.decode(errors="replace") if isinstance(stderr, bytes) else str(stderr)
    return "[3011]" in msg or "no such file" in msg.lower()


class EOSError(Exception):
    def __init__(self, cmd: list[str], stderr: bytes, *args):
        self.cmd, self.stderr = cmd, stderr
        msg = f'Operation failed\n  Command: {" ".join(cmd)}\n  Message: {stderr.decode()}'
        super().__init__(msg, *args)
        self.cmd = cmd
        self.stderr = stderr
        self._extra_args = args

    def __reduce__(self):
        # default Exception pickling calls cls(*self.args), which lacks `stderr`
        # and breaks un-serialization of results from worker processes
        return (self.__class__, (self.cmd, self.stderr, *self._extra_args))

    def __reduce__(self):
        """Make the exception survive being sent between processes.

        Exception's default pickling replays __init__ with self.args, which here holds only the
        rendered message -- one argument where __init__ wants cmd and stderr.  Unpickling then
        raised TypeError inside the worker-result reader, so a coffea run whose worker touched
        an unreachable EOS path died with "A result has failed to un-serialize" and the actual
        EOS message, the part that says what went wrong, was never shown (2026-10-15).
        """
        return (self.__class__, (self.cmd, self.stderr))


class EOS:
    _host_pattern = re.compile(r"^[\w]+://[^/]+")
    _slash_pattern = re.compile(r"(?<!:)/{2,}")

    run: bool = True
    allow_fail: bool = False
    client: Literal["eos", "xrdfs"] = "xrdfs"

    history: list[tuple[datetime, str, tuple[bool, bytes]]] = []

    def __init__(self, path: PathLike = None, host: str = ...):
        if path is None:
            self.path = None
        elif isinstance(path, EOS):
            default = path.host
            self.path = None if path.path is None else Path(path.path)
        else:
            default = ""
            if not isinstance(path, Path):
                if isinstance(path, os.PathLike):
                    path = os.fspath(path)
                match = self._host_pattern.match(path)
                if match is not None:
                    default = match.group(0)
                    path = path[len(default) :]
            self.path = Path(self._slash_pattern.sub("/", str(path)))
            if self.path == Path(os.devnull):
                self.path = None
        if self.path is None:
            self.host = ""
        else:
            self.host = arg_set(host, "", default)
            if self.host:
                self.host = ensure(self.host, __suffix="/")

    def _devnull(default=...):
        def wrapper(func):
            def method(self, *args, **kwargs):
                if self.path is None:
                    if default is ...:
                        return self
                    if isinstance(default, type):
                        return default()
                    return default
                return func(self, *args, **kwargs)

            return method

        return wrapper

    def _devnull_iter(func):
        def method(self, *args, **kwargs):
            if self.path is None:
                yield from ()
                return
            yield from func(self, *args, **kwargs)

        return method

    @property
    def as_local(self):
        return EOS(self.path, None)

    @property
    def is_local(self):
        return not self.host

    @property
    def is_null(self):
        return self.path is None

    @property
    @_devnull(False)
    def is_dir(self):
        if not self.is_local:
            raise NotImplementedError(
                f"`{EOS.is_dir.fget.__qualname__}` only works for local files"
            )  # TODO
        return self.path.is_dir()

    @property
    @_devnull(False)
    def is_file(self):
        if not self.is_local:
            raise NotImplementedError(
                f"`{EOS.is_file.fget.__qualname__}` only works for local files"
            )  # TODO
        return not self.is_dir

    @property
    @_devnull(True)
    def exists(self):
        if not self.is_local:
            # xrdfs answers a missing path with an error, which `call` raises; that is a
            # definite "no". Anything else (permission, network) still raises: we cannot tell.
            # Raising on [3011] crashed every fresh classifier training with an EOS output at
            # its checkpoint auto-resume check (2026-10-26).
            try:
                return self.call("ls", self.path)[0]
            except EOSError as e:
                if _is_missing(e.stderr):
                    return False
                raise
        # `Path` here is PurePosixPath, which has no .exists(): a pure path never touches a
        # filesystem.  Every local call raised AttributeError, which went unnoticed only
        # because nothing asked a local path whether it existed until mkdir did (2026-10-15).
        return os.path.exists(self.path)

    @classmethod
    @retry(5, delay = 2.0)
    def cmd(cls, *args) -> tuple[bool, bytes]:
        args = [str(arg) for arg in args if arg]
        if cls.run:
            try:
                output = (True, check_output(args, stderr=PIPE))
            except CalledProcessError as e:
                output = (False, e.stderr)
            except FileNotFoundError as e:
                if os.name != "posix":
                    output = (False, f'unsupported OS "{os.name.upper()}"'.encode())
                else:
                    output = (False, str(e).encode())
        else:
            output = (True, b"")
        cls.history.append((datetime.now(), " ".join(args), output))
        if not cls.allow_fail and not output[0]:
            raise EOSError(args, output[1])
        return output

    @classmethod
    def set_retry(cls, max: int = ..., delay: float = ...):
        cls.cmd.set(max=max, delay=delay)

    def call(self, executable: str, *args):
        eos = () if self.is_local else (self.client, self.host)
        return self.cmd(*eos, executable, *args)

    @_devnull(list)
    def ls(self):  # TODO test and improve
        files = self.call("ls", self.path)[1].decode().split("\n")
        if self.is_local or self.client == "eos":
            return [self / f for f in files if f]
        else:
            return [EOS(f, self.host) for f in files if f]

    @_devnull(False)
    def rm(self, recursive: bool = False):
        if not self.is_local and recursive and self.client == "xrdfs":
            raise NotImplementedError(
                f'`{self.rm.__qualname__}()` does not support recursive removal of remote files using "xrdfs" client'
            )  # TODO
        return self.call("rm", "-r" if recursive else "", self.path)[0]

    @_devnull()
    def mkdir(self, recursive: bool = False) -> EOS:
        """Create the directory, and treat one that is already there as success.

        `mkdir -p` is defined to be a no-op on an existing directory, but xrdfs does not honour
        that when the directory belongs to somebody else: it answers [3010] permission denied.
        So a job that created its output directory on one run failed on every run afterwards,
        against the directory it had made itself -- which is how SvB_friendtree_coffea4bees
        passed at 14:43 on 2026-10-15 and then failed six times in a row on the same commit.
        """
        try:
            if self.exists:
                return self
        except EOSError:
            pass            # cannot tell whether it is there; let mkdir answer, as before
        if self.call("mkdir", "-p" if recursive else "", self.path)[0]:
            return self

    @_devnull()
    def join(self, *other: str):
        if any(map(lambda x: x is None, other)):
            return EOS()
        return EOS(self.path.joinpath(*other), self.host)

    @_devnull_iter
    def walk(self) -> Generator[EOS, Any, None]:
        if not self.is_local:
            raise NotImplementedError(
                f"`{self.walk.__qualname__}()` only works for local files"
            )  # TODO
        if self.is_file:
            yield self
        else:
            for root, _, files in os.walk(self.path):
                root = EOS(root, self.host)
                for file in files:
                    yield root / file

    @_devnull_iter
    def scan(self) -> Generator[tuple[EOS, os.stat_result], Any, None]:
        if not self.is_local:
            raise NotImplementedError(
                f"`{self.scan.__qualname__}()` only works for local files"
            )  # TODO
        if self.is_file:
            yield self, self.stat()
        else:
            for entry in os.scandir(self.path):
                if entry.is_dir():
                    yield from EOS(entry.path).scan()
                else:
                    yield EOS(entry.path, self.host), entry.stat()

    def stat(self):
        if not self.is_local:
            raise NotImplementedError(
                f"`{self.stat.__qualname__}()` only works for local files"
            )  # TODO
        return self.path.stat()

    def isin(self, other: PathLike):
        other = EOS(other)
        if self.host != other.host:
            return False
        return self.common_base(self, other).path == other.path

    def relative_to(self, other: PathLike) -> str:
        other = EOS(other)
        if (self.path is None) and (other.path is None):
            return "."
        if (
            (self.path is not None)
            and (other.path is not None)
            and (self.host == other.host)
        ):
            return os.path.relpath(self.path, other.path)
        raise ValueError(
            f'Unable to determine the relative path between"{self}" and "{other}"'
        )

    @_devnull()
    def cd(self, relative: str):
        if relative is None:
            return EOS()
        return EOS(os.path.normpath(os.path.join(self, relative)), self.host)

    def copy_to(
        self,
        dst: PathLike,
        parents: bool = False,
        overwrite: bool = False,
        recursive: bool = False,
    ):
        return self.cp(self, dst, parents, overwrite, recursive)

    def move_to(
        self,
        dst: PathLike,
        parents: bool = False,
        overwrite: bool = False,
        recursive: bool = False,
    ):
        return self.mv(self, dst, parents, overwrite, recursive)

    @classmethod
    def cp(
        cls,
        src: PathLike,
        dst: PathLike,
        parents: bool = False,
        overwrite: bool = False,
        recursive: bool = False,
    ) -> EOS:
        src, dst = EOS(src), EOS(dst)
        if parents:
            dst.parent.mkdir(recursive=True)
        if src.is_local and dst.is_local:
            result = cls.cmd(
                "cp", "-r" if recursive else "", "-n" if not overwrite else "", src, dst
            )
        else:
            if recursive:
                raise NotImplementedError(
                    f"`{cls.cp.__qualname__}()` does not support recursive copying of remote files"
                )  # TODO
            result = cls.cmd("xrdcp", "-f" if overwrite else "", src, dst)
        if result[0]:
            return dst

    @classmethod
    def mv(
        cls,
        src: PathLike,
        dst: PathLike,
        parents: bool = False,
        overwrite: bool = False,
        recursive: bool = False,
    ) -> EOS:
        src, dst = EOS(src), EOS(dst)
        if (src.path is None) or (dst.path is None):
            return EOS()
        if src == dst:
            return dst
        if parents:
            dst.parent.mkdir(recursive=True)
        if src.host == dst.host:
            result = src.call(
                "mv",
                "-n" if not overwrite and src.client != "xrdfs" else "",
                src.path,
                dst.path,
            )[0]
        else:
            if recursive:
                raise NotImplementedError(
                    f"`{cls.mv.__qualname__}()` does not support recursive moving of remote files from different sites"
                )  # TODO
            result = cls.cp(src, dst, parents, overwrite, recursive)
            if result:
                result = src.rm()
        if result:
            return dst

    @property
    def name(self):
        return self.path.name

    @property
    def stem(self):
        return self.name.split(".")[0]

    @property
    def extension(self):
        exts = self.extensions
        if not exts:
            return ""
        return exts[-1]

    @property
    def extensions(self):
        return self.name.split(".")[1:]

    @property
    def suffix(self):
        return self.path.suffix

    @property
    def suffixes(self):
        return self.path.suffixes

    @property
    def parent(self):
        return EOS(self.path.parent, self.host)

    @property
    def parts(self):
        return self.path.parts

    def with_suffix(self, suffix: str):
        return EOS(self.path.with_suffix(suffix), self.host)

    def __hash__(self):
        return hash((self.host, self.path))

    def __eq__(self, other):
        if isinstance(other, EOS):
            return self.host == other.host and self.path == other.path
        elif isinstance(other, str | Path | None):
            return self == EOS(other)
        return NotImplemented

    @_devnull(os.devnull)
    def __str__(self):
        return self.host + str(self.path)

    def __repr__(self):
        return str(self)

    def __fspath__(self):
        return str(self)

    def __truediv__(self, other: str):
        return self.join(other)

    def local_temp(self, dir=None):
        fd, path = tempfile.mkstemp(suffix=f"_{self.name}", dir=dir)
        os.close(fd)
        return EOS(path)

    @classmethod
    def common_base(cls, *paths: PathLike):
        return EOS(os.path.commonpath(EOS(p, None) for p in paths))


PathLike = str | EOS | os.PathLike
"""
str, ~heptools.system.eos.EOS, ~os.PathLike: A str or path-like object with :meth:`__fspath__` method.
"""


def open_zip(
    algorithm: Literal["", "gzip", "bz2", "lzma"], file: PathLike, mode: str, **kwargs
):
    if not algorithm:
        return open(file, mode, **kwargs)
    module = importlib.import_module(algorithm)
    default = {}
    if algorithm in ["gzip", "bz2"]:
        default["compresslevel"] = 4
    return module.open(file, mode, **(default | kwargs))


def save(
    file: PathLike,
    obj,
    algorithm: Literal["", "gzip", "bz2", "lzma"] = "gzip",
    **kwargs,
):
    file = EOS(file)
    if file.is_local:
        pickle.dump(obj, open_zip(algorithm, file, "wb", **kwargs))
    else:
        raise NotImplementedError(
            f"`{save.__qualname__}()` does not support remote files"
        )  # TODO


def load(
    file: PathLike, algorithm: Literal["", "gzip", "bz2", "lzma"] = "gzip", **kwargs
):
    file = EOS(file)
    if file.is_local:
        return pickle.load(open_zip(algorithm, file, "rb", **kwargs))
    else:
        raise NotImplementedError(
            f"`{load.__qualname__}()` does not support remote files"
        )  # TODO
