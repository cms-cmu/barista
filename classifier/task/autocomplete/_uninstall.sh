#!/bin/bash
python -m classifier.task.autocomplete._bind exit
pkill -9 -u $USER -xf "python -m classifier.task.autocomplete._core"
complete -r "./pyml.py"