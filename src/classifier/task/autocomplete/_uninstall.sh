#!/bin/bash
python -m src.classifier.task.autocomplete._bind exit
pkill -9 -u $USER -xf "python -m src.classifier.task.autocomplete._core"
complete -r "./pyml.py"