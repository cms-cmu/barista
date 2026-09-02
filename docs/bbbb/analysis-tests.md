#  tests

To run the cutflow_test

```
python analysis/tests/cutflow_test.py   --inputFile analysis/output/test.coffea --knownCounts analysis/tests/known_Counts.yml
```

For all Run2

```
python analysis/tests/cutflow_test.py   --inputFile analysis/output/histAll.coffea --knownCounts analysis/tests/histAllCounts.yml
```

# To update the cutflow numbers:

```
python     analysis/tests/dumpCutFlow.py --input [inputFileName] -o [outputFielName]
```

```
python analysis/tests/dumpCutFlow.py --input output/histAll.coffea -o analysis/tests/histAllCounts.yml
```

