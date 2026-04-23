#!/bin/sh
"exec" "python3" "-m" "src.pyml" "$@"
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
        from .classifier.sysutils import recursive_interrupt

        recursive_interrupt()
