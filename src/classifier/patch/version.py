from packaging import version


def issue_version(module, fixed: str = None, introduced: str = None) -> bool:
    try:
        v = version.parse(module.__version__)
    except AttributeError:
        return True
    if fixed is None or v >= version.parse(fixed):
        return False
    if introduced is None or v < version.parse(introduced):
        return False
    return True
