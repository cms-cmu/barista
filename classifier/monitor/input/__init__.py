from __future__ import annotations

from typing import Optional

from classifier.config.setting import monitor as cfg

from ..backends import Platform
from ..core import MonitorProxy, Recorder, post_to_monitor


class RemoteInput(MonitorProxy):
    @classmethod
    @cfg.check(cfg.Input)
    def get(
        cls,
        title: str,
        multiline: bool = False,
        password: bool = False,
        platform: Platform = None,
    ) -> Optional[str]:
        return cls._get(
            title=title,
            multiline=multiline,
            password=password,
            platform=platform,
            node=Recorder.name(),
        )

    @post_to_monitor(wait_for_return=True, acquire_lock=True)
    def _get(
        cls,
        title: str,
        multiline: bool,
        password: bool,
        platform: Platform,
        node: str,
    ):
        if platform is None or Platform.console in platform:
            from ..backends.console import Dashboard
            from ._console import Input

            rich_input = Input(
                title=f"[cyan]\[{Recorder.registered(node)}][/cyan]{title}",
                multiline=multiline,
                password=password,
            )
            Dashboard.layout.add(rich_input)
            rich_input.get()
            Dashboard.layout.remove(rich_input)
            return rich_input.text
