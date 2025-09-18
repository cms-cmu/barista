from typing import Callable, Protocol, TypeVar

from src.utils.wrapper import MethodDecorator
from classifier.task.state import GlobalSetting

from . import Monitor


class MonitorComponentConfig(Protocol):
    enable: bool


class MonitorComponentStatus(MethodDecorator):
    def __init__(
        self,
        func: Callable,
        dependencies: tuple[MonitorComponentConfig],
        default=None,
        is_callable: bool = False,
    ):
        super().__init__(func)
        self._cfgs = dependencies + (Monitor,)
        self._default = default
        self._is_callable = is_callable

    def __call__(self, *args, **kwargs):
        if all(cfg.enable for cfg in self._cfgs):
            return self._func(*args, **kwargs)
        if self._is_callable:
            return self._default()
        else:
            return self._default


_FuncT = TypeVar("_FuncT")


def check(
    *dependencies: MonitorComponentConfig,
    default=None,
    is_callable: bool = False,
) -> Callable[[_FuncT], _FuncT]:
    return lambda func: MonitorComponentStatus(func, dependencies, default, is_callable)


# backends
class Console(GlobalSetting):
    "Backend: console (rich)"

    enable: bool = True
    "enable the console backend"

    interval: float = 1.0
    "(seconds) interval to update the "
    fps: int = 10
    "frames per second"


class Web(GlobalSetting):  # TODO placeholder
    "Backend: web page (flask)"

    enable: bool = False
    "enable the web backend"


# functions
class Log(GlobalSetting):
    "Logging system"

    enable: bool = True
    "enable logging"
    file: str = "logs.html"
    "name of the log file"

    level: int = 20
    "logging level"
    forward_exception: bool = False
    "forward the uncaught exceptions to the monitor (set this to False or run a standalone monitor if some exceptions do not show up)"


class Progress(GlobalSetting):
    "Progress bars"

    enable: bool = True
    "enable progress bars"


class Usage(GlobalSetting):
    "Usage statistics"

    enable: bool = False
    "enable usage trackers (this will significantly slow down the program)"
    file: str = "usage.json"
    "name of the file to dump the raw usage data"

    interval: float = 1.0
    "(seconds) interval to update the usage"
    gpu: bool = True
    "track GPU usage"
    gpu_force_torch: bool = False
    "force to fetch GPU usage from pytorch instead of pynvml"


class Input(GlobalSetting):
    "Text input"

    enable: bool = True
    "enable text input"


class Notification(GlobalSetting):
    "Notification system"

    enable: bool = True
    "enable notification"


class Gmail(GlobalSetting):
    "Gmail notification via SMTP"

    enable: bool = True
    "enable Gmail notification"
    address: str = None
    "Gmail address"
    password: str = None
    """Google app password
    - NEVER pass this through the command line
    - option1: put the password in a local file with chmod 600
    - option2: leave this empty and an user input will be requested when needed
    - see https://support.google.com/accounts/answer/185833
    """

    smtp_server: str = "smtp.gmail.com"
    "SMTP server"
    smtp_port: int = 465
    "SMTP port"

    @classmethod
    def get__address(cls, value: str) -> str:
        if value is None:
            from classifier.monitor.input import RemoteInput

            value = RemoteInput.get("Gmail address")
            cls.address = value
        return value

    @classmethod
    def get__password(cls, value: str) -> str:
        if value is None:
            from classifier.monitor.input import RemoteInput

            value = RemoteInput.get("Google app password", password=True)
            cls.password = value
        return value
