from enum import Flag, auto


class Platform(Flag):
    console = auto()
    web = auto()
    all = console | web
