# Tools in Barista

## How to compute integrated luminosity for data

The script `src/tools/compute_lumi_processes.py` reads a YAML file that is the output of the skimming steps, containing `lumis_processed` data (run numbers and lumi sections), converts it to the JSON format required by [brilcalc](https://cms-opendata-workshop.github.io/workshop-lesson-luminosity/03-calculating-luminosity/index.html), and automatically runs brilcalc to compute the integrated luminosity.

### Usage

**Basic usage** (prints luminosity to console):
```bash
./run_container brilcalc python src/tools/compute_lumi_processes.py -i <input.yml>
```

**Save output to file**:
```bash
./run_container brilcalc python src/tools/compute_lumi_processes.py -i <input.yml> -o <output.txt>
```

### Command-line Options

- `-i, --input`: **(Required)** Input YAML file with lumis_processed data
- `-o, --output`: **(Optional)** File to save brilcalc output. If not specified, output is printed to console
