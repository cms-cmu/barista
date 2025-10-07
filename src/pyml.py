#!/usr/bin/env python
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.classifier.patch import patch_awkward_pandas
from src.classifier.task import EntryPoint

if __name__ == "__main__":
    # install patch
    patch_awkward_pandas()
    # run main
    main = EntryPoint()
    try:
        main.run()
    except KeyboardInterrupt:
        from src.classifier.sysutils import recursive_interrupt
        recursive_interrupt()
