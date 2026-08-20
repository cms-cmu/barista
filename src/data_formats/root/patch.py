import logging
import time
from types import MethodType
from typing import Sequence

from packaging import version


class _uproot_XRootDResource_open_retry:
    def __init__(
        self,
        max_retries: int,
        delay: Sequence[float],
        suppress_logging: bool = False,
    ):
        self._max = max_retries
        self._delay = delay
        self._log = not suppress_logging

    def __get__(self, obj, cls):
        if obj is None:
            return self
        return MethodType(self, obj)

    def __call__(self, obj):
        from uproot.extras import XRootD_client

        client = XRootD_client()

        for i in range(self._max):
            retry, delay = i + 1, 0
            if retry < self._max:
                delay = self._delay[i]

            obj._file = client.File()
            status, _ = obj._file.open(obj._file_path, timeout=obj._xrd_timeout())
            if not status.error:
                break
            else:
                try:
                    obj._file.close()
                except Exception:
                    pass
                try:
                    obj._xrd_error(status)
                except Exception:
                    if retry < self._max:
                        if self._log:
                            logging.warning(
                                f"Attempt {retry}/{self._max} failed while opening {obj._file_path}. Next attempt in {delay} seconds."
                            )
                    else:
                        raise
            time.sleep(delay)


def uproot_XRootD_retry(
    max_retries: int, delay: Sequence[float], suppress_logging: bool = False
):
    import uproot

    v = version.parse(uproot.__version__)
    if v >= version.parse("4.1.0"):
        from uproot.source.xrootd import XRootDResource

        XRootDResource._open = _uproot_XRootDResource_open_retry(
            max_retries, delay, suppress_logging
        )
