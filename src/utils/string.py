import re


def ensure(__str: str, __prefix: str = None, __suffix: str = None):
    if __prefix is not None and not __str.startswith(__prefix):
        __str = __prefix + __str
    if __suffix is not None and not __str.endswith(__suffix):
        __str = __str + __suffix
    return __str


def format_stage_title(title: str) -> str:
    """Format underscore-delimited Stage_/Phase_/bkg_syst_ CLI titles into readable text."""
    if not title or " " in title or not re.match(r"^(?:Stage|Phase|bkg_syst)_", title):
        return title
    s = title.replace("SvB_MA", "\x00SVBMA\x00")
    s = re.sub(r"^bkg_syst_", "Stage_", s)
    s = re.sub(r"^(Stage|Phase)_([A-Za-z])_(\d+)_", r"\1 \2.\3 ", s)
    s = re.sub(r"^(Stage|Phase)_([A-Za-z0-9]+)_", r"\1 \2 ", s)
    return s.replace("_", " ").replace("\x00SVBMA\x00", "SvB_MA").strip()
