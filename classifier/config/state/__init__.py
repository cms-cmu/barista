import getpass
import os
from datetime import datetime

from classifier.task import GlobalState
from classifier.task.state import _SET


class System(GlobalState):
    user_name: str = None
    main_task: str = None
    startup_time: datetime = datetime.fromtimestamp(0)
    in_singularity: bool = False

    @classmethod
    def _init(cls, main_task: str):
        try:
            cls.user_name = getpass.getuser()
        except Exception:
            cls.user_name = "unknown"
        cls.main_task = main_task
        cls.startup_time = datetime.now()
        cls.in_singularity = os.path.exists("/.singularity.d")

    @classmethod
    def run_time(cls):
        start = cls.startup_time
        if start.timestamp() > 0:
            return datetime.now() - start
        else:
            return start - start


class Flags(GlobalState):
    test: bool = False
    debug: bool = False

    def _set(self, *names: str):
        for name in names:
            if name in self.__annotations__:
                value = getattr(self, name)
                setter = f"{_SET}{name}"
                if not value and hasattr(self, setter):
                    getattr(self, setter)()
                setattr(self, name, True)

    @classmethod
    def set(cls, *names: str):
        cls._set(cls, *names)

    @classmethod
    def set__debug(cls):
        from ..setting.monitor import Log

        Log.level = 10
