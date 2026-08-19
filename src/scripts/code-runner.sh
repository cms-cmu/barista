#!/bin/bash

echo "############### Running runner.py unit tests"
python -m pytest src/tests/test_runner.py src/tests/test_dataset_resolution.py

