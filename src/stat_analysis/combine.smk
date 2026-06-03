import os
import sys
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from src.stat_analysis.helpers import make_poi_maps, get_default_othersignals, get_grid_split_points, get_likelihood_scan_chunks


# Resolve combine container image dynamically based on CVMFS availability
COMBINE_IMAGE = "docker://gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0"
if os.path.exists("/cvmfs/unpacked.cern.ch"):
    COMBINE_IMAGE = f"/cvmfs/unpacked.cern.ch/{COMBINE_IMAGE.replace('docker://', '')}"


# Default target generation for transparent running
out_base = os.path.normpath(config.get("output_path", "output/v4_systematics_test/HH4b/"))
stat_only = config.get("stat_only", False)

targets = []
for channel, ch_config in config.get("channels", {}).items():
    signallabel = ch_config.get("signallabel")
    if not signallabel:
        continue
    # If run directly, the default path prefix is:
    base_prefix = f"{out_base}/workspace/datacard"
    targets.extend([
        f"{base_prefix}_limits__{signallabel}.json",
        f"significance__{base_prefix}__{signallabel}.log",
        f"{base_prefix}_likelihood_scan__{signallabel}.pdf",
        f"{base_prefix}_likelihood_scan__{signallabel}.png",
        f"{base_prefix}_postfit__{signallabel}.pdf",
        f"{base_prefix}_postfit__{signallabel}.png",
        f"{base_prefix}_fitDiagnostics_bonly__{signallabel}.root",
        f"{base_prefix}_fitDiagnostics_sb__{signallabel}.root",
        f"{base_prefix}_diffNuisances_bonly__{signallabel}.root",
        f"{base_prefix}_diffNuisances_sb__{signallabel}.root"
    ])
    if not stat_only:
        targets.extend([
            f"{base_prefix}_gof__{signallabel}.pdf",
            f"{base_prefix}_gof__{signallabel}.png",
            f"{base_prefix}_impacts__{signallabel}.pdf",
            f"{base_prefix}_impacts__{signallabel}.png",
            f"{base_prefix}_impacts_split__{signallabel}"
        ])

if targets:
    rule all:
        input: targets

def get_workspace_input(wildcards):
    default_input = f"{wildcards.path}.txt"
    if os.path.exists(default_input):
        return default_input
    basename = os.path.basename(wildcards.path)
    if basename.startswith("datacard__"):
        signallabel = basename.replace("datacard__", "")
    elif "__" in basename:
        signallabel = basename.split("__")[-1]
    else:
        signallabel = basename
    for channel, ch_config in config.get("channels", {}).items():
        if ch_config.get("signallabel") == signallabel:
            datacards_dir = os.path.join(os.path.dirname(out_base), "datacards", channel)
            input_datacard = os.path.join(datacards_dir, f"datacard__{channel}.txt")
            if os.path.exists(input_datacard):
                return input_datacard
    return default_input

def get_poi_maps_dynamic(wildcards):
    basename = os.path.basename(wildcards.path)
    if basename.startswith("datacard__"):
        signallabel = basename.replace("datacard__", "")
    elif "__" in basename:
        signallabel = basename.split("__")[-1]
    else:
        signallabel = basename
    for channel, ch_config in config.get("channels", {}).items():
        if ch_config.get("signallabel") == signallabel:
            signals = [ch_config["signallabel"]] + ch_config.get("othersignal", "").split()
            poi_ranges = config.get("poi_ranges", "1,-10,10")
            return make_poi_maps(signals=signals, poi_ranges=poi_ranges)
    return ""


ruleorder: likelihood_scan_snapshot > likelihood_scan_chunk > impacts_initial_fit > gof_data > gof_toys_chunk > fit_diagnostics_bonly > fit_diagnostics_sb > workspace

localrules: pdf_to_png, split_impacts, impacts_collect

rule workspace:
    input: get_workspace_input
    output: "{path}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        poi_maps = lambda wildcards: get_poi_maps_dynamic(wildcards),
        physics_model = lambda wildcards: config.get("physics_model", "HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel"),
        extra_t2w_args = lambda wildcards: config.get("extra_t2w_args", "--PO verbose")
    log: "output/logs/workspace_{path}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        set -o pipefail
        LOG=$(pwd)/{log}
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname {output})

        # Copy all input datacards and shape root files to workspace for self-containment atomically
        python3 -c '
import os, shutil, glob, tempfile
datacard = "{input}"
out_dir = os.path.dirname("{output}")
in_dir = os.path.dirname(datacard)
os.makedirs(out_dir, exist_ok=True)
for ext in ["*.txt", "*.root"]:
    for f in glob.glob(os.path.join(in_dir, ext)):
        dest = os.path.join(out_dir, os.path.basename(f))
        if os.path.exists(dest):
            if os.path.getsize(f) == os.path.getsize(dest):
                continue
        tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir)
        try:
            shutil.copy2(f, tmp_path)
            os.close(tmp_fd)
            os.replace(tmp_path, dest)
        except Exception:
            os.close(tmp_fd)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
'

        (
        echo "[$(date)] Starting workspace rule"
        cd $(dirname {output}) && \
            text2workspace.py $(basename {input}) \
            -P {params.physics_model} \
            {params.poi_maps} \
            {params.extra_t2w_args} \
            -o $(basename {output}) && \
            rootls $(basename {output})
        ) 2>&1 | tee {log}
        """

rule limits:
    input: "{path}__{signallabel}.root"
    output: 
        txt="{path}_limits__{signallabel}.txt",
        json="{path}_limits__{signallabel}.json"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120")
    log: "output/logs/limits_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        OUT_TXT=$(realpath {output.txt})
        OUT_JSON=$(realpath {output.json})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_TXT)
        mkdir -p $(dirname $OUT_JSON)
        (
        echo "[$(date)] Starting limits rule with signal {params.signallabel}"

        FREEZE_OPT=""
        if [ -n "{params.freeze_parameters}" ]; then
            formatted_params=$(echo "{params.freeze_parameters}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                FREEZE_OPT="--freezeParameters $formatted_params"
            fi
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                SET_ZERO_OPT="--setParameters $formatted_params"
            fi
        fi

        echo "[$(date)] Running AsymptoticLimits"
        cd $DATACARD_DIR && \
            combine -M AsymptoticLimits $(basename {input}) \
            -m {params.mass} \
            --redefineSignalPOIs r{params.signallabel} \
            $SET_ZERO_OPT \
            $FREEZE_OPT \
            -n _{params.signallabel} \
            > temp_limits.txt && \
        echo "[$(date)] Running CollectLimits" && \
            combineTool.py -M CollectLimits \
            higgsCombine_{params.signallabel}.AsymptoticLimits.mH{params.mass}.root \
            -o temp_limits.json && \
            mv temp_limits.txt $OUT_TXT && \
            mv temp_limits.json $OUT_JSON
        ) 2>&1 | tee {log}
        """

rule significance:
    input: "{path}__{signallabel}.root"
    output: "significance__{path}__{signallabel}.log"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120")
    log: "output/logs/significance_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname {output})
        OUT_FILE=$(realpath {output})
        (
        echo "[$(date)] Starting significance rule with signal {params.signallabel}"

        FREEZE_OPT=""
        if [ -n "{params.freeze_parameters}" ]; then
            formatted_params=$(echo "{params.freeze_parameters}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                FREEZE_OPT="--freezeParameters $formatted_params"
            fi
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                SET_ZERO_OPT="--setParameters $formatted_params"
            fi
        fi

        cd $DATACARD_DIR && \
            combine -M Significance $(basename {input}) \
            -m {params.mass} \
            $SET_ZERO_OPT \
            $FREEZE_OPT \
            --redefineSignalPOIs r{params.signallabel} \
            -n _{params.signallabel} > $OUT_FILE && \
            combine -M Significance $(basename {input}) \
            -m {params.mass} \
            --redefineSignalPOIs r{params.signallabel} \
            $SET_ZERO_OPT \
            $FREEZE_OPT \
            -n _{params.signallabel} \
            -t -1 --expectSignal=1 >> $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule likelihood_scan_snapshot:
    input: "{path}__{signallabel}.root"
    output: temp("{path}_likelihood_scan_snapshot__{signallabel}.root")
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120")
    log: "output/logs/likelihood_scan_snapshot_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        OUT_FILE=$(realpath {output})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_FILE)
        (
        echo "[$(date)] Starting likelihood_scan snapshot fit with signal {params.signallabel}"

        FREEZE_OPT=""
        if [ -n "{params.freeze_parameters}" ]; then
            formatted_params=$(echo "{params.freeze_parameters}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                FREEZE_OPT="--freezeParameters $formatted_params"
            fi
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                SET_ZERO_OPT="--setParameters $formatted_params"
            fi
        fi

        cd $DATACARD_DIR && \
            combine -M MultiDimFit -d $(basename {input}) \
            -m {params.mass} \
            -n _$(basename {input} .root)_snapshot \
            $SET_ZERO_OPT \
            $FREEZE_OPT \
            --saveWorkspace --robustFit 1
        ) 2>&1 | tee {log} && \
        cp $DATACARD_DIR/higgsCombine_$(basename {input} .root)_snapshot.MultiDimFit.mH{params.mass}.root $OUT_FILE
        """

rule likelihood_scan_chunk:
    input: "{path}_likelihood_scan_snapshot__{signallabel}.root"
    output: temp("{path}_likelihood_scan_chunk_{split_index}__{signallabel}.root")
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120"),
        points = lambda wildcards: config.get("likelihood_scan_points", "50"),
        r_min = lambda wildcards: config.get("r_min", "-10"),
        r_max = lambda wildcards: config.get("r_max", "10"),
        first_point = lambda wildcards: get_grid_split_points(wildcards, config)[0],
        last_point = lambda wildcards: get_grid_split_points(wildcards, config)[1]
    log: "output/logs/likelihood_scan_chunk_{split_index}_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        OUT_FILE=$(realpath {output})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_FILE)
        (
        echo "[$(date)] Starting likelihood_scan chunk {wildcards.split_index} with signal {params.signallabel}"

        FREEZE_OPT=""
        if [ -n "{params.freeze_parameters}" ]; then
            formatted_params=$(echo "{params.freeze_parameters}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                FREEZE_OPT="--freezeParameters $formatted_params"
            fi
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                SET_ZERO_OPT="--setParameters $formatted_params"
            fi
        fi

        cd $DATACARD_DIR && \
            combine -M MultiDimFit \
            -d $(basename {input}) \
            -n _$(basename {input} .root)_chunk_{wildcards.split_index} \
            -m {params.mass} \
            -P r{params.signallabel} \
            $SET_ZERO_OPT \
            $FREEZE_OPT \
            --snapshotName MultiDimFit --rMin {params.r_min} --rMax {params.r_max} --algo grid --points {params.points} --firstPoint {params.first_point} --lastPoint {params.last_point} --alignEdges 1
        ) 2>&1 | tee {log} && \
        cp $DATACARD_DIR/higgsCombine_$(basename {input} .root)_chunk_{wildcards.split_index}.MultiDimFit.mH{params.mass}.root $OUT_FILE
        """

rule likelihood_scan:
    input:
        lambda wildcards: get_likelihood_scan_chunks(wildcards, config)
    output: "{path}_likelihood_scan__{signallabel}.pdf"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        mass = lambda wildcards: config.get("mass", "120")
    log: "output/logs/likelihood_scan_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        OUT_FILE=$(realpath {output})
        DATACARD_DIR=$(dirname $OUT_FILE)
        mkdir -p $(dirname $LOG)
        mkdir -p $DATACARD_DIR
        (
        echo "[$(date)] Merging likelihood scan chunks and plotting"
        cd $DATACARD_DIR && \
            hadd -f higgsCombine_merged_{params.signallabel}.MultiDimFit.mH{params.mass}.root \
            $(for f in {input}; do basename $f; done) && \
            plot1DScan.py higgsCombine_merged_{params.signallabel}.MultiDimFit.mH{params.mass}.root \
            --POI r{params.signallabel} -o scan_plot && \
            mv scan_plot.pdf $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule impacts_initial_fit:
    input: "{path}__{signallabel}.root"
    output: "{path}_initialFit__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        set_parameters_ranges = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "125"),
        r_min = lambda wildcards: config.get("r_min", "-10"),
        r_max = lambda wildcards: config.get("r_max", "10"),
        stat_only = lambda wildcards: config.get("stat_only", False)
    log: "output/logs/impacts_initial_fit_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        OUT_FILE=$(realpath {output})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_FILE)
        (
        echo "[$(date)] Starting impacts_initial_fit rule with signal {params.signallabel}"

        # Check if running in stat_only mode
        if [ "{params.stat_only}" = "True" ] || [ "{params.stat_only}" = "1" ]; then
            echo "stat_only is enabled. Skipping impacts initial fit."
            echo "stat_only" > $OUT_FILE
            exit 0
        fi

        # Check if there are any nuisance parameters
        NUISANCES=$(find . -maxdepth 3 -name "*.txt" -exec grep -h "kmax" {{}} + 2>/dev/null | awk '{{print $2}}' | head -n 1)
        if [ "$NUISANCES" = "0" ] || [ -z "$NUISANCES" ]; then
            echo "no_nuisances" > $OUT_FILE
            exit 0
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                SET_ZERO_OPT="--setParameters $formatted_params"
            fi
        fi

        SET_RANGES_OPT=""
        if [ -n "{params.set_parameters_ranges}" ]; then
            formatted_params=$(echo "{params.set_parameters_ranges}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0,0/' | paste -sd: -)
            if [ -n "$formatted_params" ]; then
                SET_RANGES_OPT=":$formatted_params"
            fi
        fi

        echo "[$(date)] Running initial fit"
        cd $DATACARD_DIR && \
            combineTool.py -M Impacts -d $(basename {input}) \
            --doInitialFit --robustFit 1 -m {params.mass} \
            --redefineSignalPOIs r{params.signallabel} \
            --setParameterRanges r{params.signallabel}={params.r_min},{params.r_max}$SET_RANGES_OPT \
            $SET_ZERO_OPT \
            -n $(basename {input} .root)
        ) 2>&1 | tee {log} && \
        cp $DATACARD_DIR/higgsCombine_initialFit_$(basename {input} .root).MultiDimFit.mH{params.mass}.root $OUT_FILE
        """

rule impacts_do_fits:
    input:
        workspace = "{path}__{signallabel}.root",
        init_fit = "{path}_initialFit__{signallabel}.root"
    output:
        fits_dir = directory("{path}_impacts_fits__{signallabel}")
    container: config.get("combine_container", COMBINE_IMAGE)
    threads: int(config.get("impacts_parallel", "4"))
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        set_parameters_ranges = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "125"),
        r_min = lambda wildcards: config.get("r_min", "-10"),
        r_max = lambda wildcards: config.get("r_max", "10"),
        stat_only = lambda wildcards: config.get("stat_only", False)
    log: "output/logs/impacts_do_fits_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input.workspace}))
        OUT_DIR=$(realpath {output.fits_dir})
        mkdir -p $(dirname $LOG)
        mkdir -p $OUT_DIR
        (
        echo "[$(date)] Starting impacts_do_fits rule with signal {params.signallabel}"

        # Check if running in stat_only mode or no_nuisances mode
        if [ "{params.stat_only}" = "True" ] || [ "{params.stat_only}" = "1" ] || [ "$(cat {input.init_fit} 2>/dev/null)" = "stat_only" ]; then
            echo "stat_only" > $OUT_DIR/stat_only
            exit 0
        fi

        if [ -f {input.init_fit} ] && [ "$(cat {input.init_fit} 2>/dev/null)" = "no_nuisances" ]; then
            echo "no_nuisances" > $OUT_DIR/no_nuisances
            exit 0
        fi

        # Copy the initial fit root file back to DATACARD_DIR so combineTool can find it
        cp $(realpath {input.init_fit}) $DATACARD_DIR/higgsCombine_initialFit_$(basename {input.workspace} .root).MultiDimFit.mH{params.mass}.root

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                SET_ZERO_OPT="--setParameters $formatted_params"
            fi
        fi

        SET_RANGES_OPT=""
        if [ -n "{params.set_parameters_ranges}" ]; then
            formatted_params=$(echo "{params.set_parameters_ranges}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0,0/' | paste -sd: -)
            if [ -n "$formatted_params" ]; then
                SET_RANGES_OPT=":$formatted_params"
            fi
        fi

        echo "[$(date)] Running fits per systematic"
        cd $DATACARD_DIR && \
            combineTool.py -M Impacts -d $(basename {input.workspace}) \
            --doFits --robustFit 1 -m {params.mass} --parallel {threads} \
            --redefineSignalPOIs r{params.signallabel} \
            --setParameterRanges r{params.signallabel}={params.r_min},{params.r_max}$SET_RANGES_OPT \
            $SET_ZERO_OPT \
            -n $(basename {input.workspace} .root)
        ) 2>&1 | tee {log} && \
        cp $DATACARD_DIR/higgsCombine_paramFit_$(basename {input.workspace} .root)_*.root $OUT_DIR/ && \
        cp $(realpath {input.init_fit}) $OUT_DIR/
        """

rule impacts_collect:
    input:
        workspace = "{path}__{signallabel}.root",
        fits_done = "{path}_impacts_fits__{signallabel}"
    output:
        pdf = "{path}_impacts__{signallabel}.pdf"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        mass = lambda wildcards: config.get("mass", "125"),
        per_page = lambda wildcards: config.get("impacts_per_page", "20"),
        stat_only = lambda wildcards: config.get("stat_only", False)
    log: "output/logs/impacts_collect_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input.workspace}))
        OUT_FILE=$(realpath {output.pdf})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_FILE)
        (
        echo "[$(date)] Starting impacts_collect rule with signal {params.signallabel}"

        # Check if running in stat_only mode
        if [ "{params.stat_only}" = "True" ] || [ "{params.stat_only}" = "1" ] || [ -f {input.fits_done}/stat_only ]; then
            echo "stat_only is enabled. Creating dummy impacts plot."
            cat << 'EOF' > dummy_plot.py
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.text(0.5, 0.5, 'stat_only is enabled (Impacts not run)', ha='center', va='center', fontsize=14)
ax.axis('off')
plt.savefig(sys.argv[1])
EOF
            python3 dummy_plot.py $OUT_FILE
            rm dummy_plot.py
            exit 0
        fi

        if [ -f {input.fits_done}/no_nuisances ]; then
            cat << 'EOF' > dummy_plot.py
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.text(0.5, 0.5, 'No Nuisance Parameters in Datacard (Impacts not applicable)', ha='center', va='center', fontsize=14)
ax.axis('off')
plt.savefig(sys.argv[1])
EOF
            python3 dummy_plot.py $OUT_FILE
            rm dummy_plot.py
            exit 0
        fi

        # Copy files back from fits_done directory so combineTool can merge them
        cp {input.fits_done}/higgsCombine_initialFit_$(basename {input.workspace} .root).MultiDimFit.mH{params.mass}.root $DATACARD_DIR/
        cp {input.fits_done}/higgsCombine_paramFit_$(basename {input.workspace} .root)_*.root $DATACARD_DIR/

        echo "[$(date)] Running merging results"
        cd $DATACARD_DIR && \
            combineTool.py -M Impacts \
            -m {params.mass} -n $(basename {input.workspace} .root) \
            --redefineSignalPOIs r{params.signallabel} \
            -d $(basename {input.workspace}) \
            -o impacts_combine_$(basename {input.workspace} .root)_exp.json

        echo "[$(date)] Running creating pdf"
        cd $DATACARD_DIR && \
            plotImpacts.py -i impacts_combine_$(basename {input.workspace} .root)_exp.json \
            -o impacts_plot \
            --POI r{params.signallabel} \
            --per-page {params.per_page} --left-margin 0.3 --height 400 --label-size 0.04 && \
            mv impacts_plot.pdf $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule gof_data:
    input: "{path}__{signallabel}.root"
    output: "{path}_gof_data__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120"),
        gof_algo = lambda wildcards: config.get("gof_algo", "saturated"),
        stat_only = lambda wildcards: config.get("stat_only", False)
    log: "output/logs/gof_data_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        OUT_FILE=$(realpath {output})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_FILE)
        (
        echo "[$(date)] Starting gof_data rule with signal {params.signallabel}"

        if [ "{params.stat_only}" = "True" ] || [ "{params.stat_only}" = "1" ]; then
            echo "stat_only is enabled. Skipping GoF data fit."
            echo "stat_only" > $OUT_FILE
            exit 0
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                SET_ZERO_OPT="--setParameters $formatted_params"
            fi
        fi

        cd $DATACARD_DIR && \
            combine -M GoodnessOfFit $(basename {input}) \
            -m {params.mass} \
            --algo {params.gof_algo} \
            $SET_ZERO_OPT \
            -n _$(basename {input} .root)_{params.signallabel}_gof_data \
            2>&1 | tee gof_data_$(basename {input} .root)_{params.signallabel}.txt && \
            cp higgsCombine_$(basename {input} .root)_{params.signallabel}_gof_data.GoodnessOfFit.mH{params.mass}.root $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule gof_toys_chunk:
    input: "{path}__{signallabel}.root"
    output: "{path}_gof_toys_{split_index}__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120"),
        toys_per_job = lambda wildcards: config.get("toys_per_job", "50"),
        gof_algo = lambda wildcards: config.get("gof_algo", "saturated"),
        seed = lambda wildcards: int(wildcards.split_index) + 123456,
        stat_only = lambda wildcards: config.get("stat_only", False)
    log: "output/logs/gof_toys_chunk_{split_index}_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        OUT_FILE=$(realpath {output})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_FILE)
        (
        echo "[$(date)] Starting gof_toys_chunk {wildcards.split_index} rule with signal {params.signallabel}"

        if [ "{params.stat_only}" = "True" ] || [ "{params.stat_only}" = "1" ]; then
            echo "stat_only is enabled. Skipping GoF toys chunk."
            echo "stat_only" > $OUT_FILE
            exit 0
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                SET_ZERO_OPT="--setParameters $formatted_params"
            fi
        fi

        cd $DATACARD_DIR
        # Check if there are any nuisance parameters
        TOYS_OPT="--toysFrequentist"
        NUISANCES=$(find . -maxdepth 3 -name "*.txt" -exec grep -h "kmax" {{}} + 2>/dev/null | awk '{{print $2}}' | head -n 1)
        if [ "$NUISANCES" = "0" ] || [ -z "$NUISANCES" ]; then
            TOYS_OPT="--toysNoSystematics"
        fi

        combine -M GoodnessOfFit $(basename {input}) \
            -m {params.mass} \
            -t {params.toys_per_job} --algo {params.gof_algo} $TOYS_OPT \
            -s {params.seed} \
            $SET_ZERO_OPT \
            -n _$(basename {input} .root)_{params.signallabel}_gof_toys_{wildcards.split_index} \
            2>&1 | tee gof_toys_$(basename {input} .root)_{params.signallabel}_{wildcards.split_index}.txt && \
            cp higgsCombine_$(basename {input} .root)_{params.signallabel}_gof_toys_{wildcards.split_index}.GoodnessOfFit.mH{params.mass}.{params.seed}.root $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule gof:
    input:
        data = "{path}_gof_data__{signallabel}.root",
        toys = lambda wildcards: [f"{wildcards.path}_gof_toys_{i}__{wildcards.signallabel}.root" for i in range(int(config.get("num_toy_jobs", 10)))]
    output: "{path}_gof__{signallabel}.pdf"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        mass = lambda wildcards: config.get("mass", "120"),
        gof_algo = lambda wildcards: config.get("gof_algo", "saturated"),
        stat_only = lambda wildcards: config.get("stat_only", False)
    log: "output/logs/gof_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input.data}))
        OUT_FILE=$(realpath {output})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_FILE)
        (
        echo "[$(date)] Merging gof toys and plotting GoF saturated distribution"

        if [ "{params.stat_only}" = "True" ] || [ "{params.stat_only}" = "1" ] || [ "$(cat {input.data} 2>/dev/null)" = "stat_only" ]; then
            echo "stat_only is enabled. Creating dummy GoF plot."
            cat << 'EOF' > dummy_plot.py
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.text(0.5, 0.5, 'stat_only is enabled (GoF not run)', ha='center', va='center', fontsize=14)
ax.axis('off')
plt.savefig(sys.argv[1])
EOF
            python3 dummy_plot.py $OUT_FILE
            rm dummy_plot.py
            exit 0
        fi

        cd $DATACARD_DIR && \
            combineTool.py -M CollectGoodnessOfFit \
            --input $(basename {input.data}) $(for f in {input.toys}; do basename $f; done) \
            -o gof_$(basename {input.data} _gof_data__{params.signallabel}.root)_{params.signallabel}.json && \
            plotGof.py gof_$(basename {input.data} _gof_data__{params.signallabel}.root)_{params.signallabel}.json \
            --statistic {params.gof_algo} --mass {params.mass}.0 \
            --output gof_plot && \
            mv gof_plot.pdf $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule fit_diagnostics_bonly:
    input: "{path}__{signallabel}.root"
    output:
        bonly = "{path}_fitDiagnostics_bonly__{signallabel}.root",
        diff_bonly = "{path}_diffNuisances_bonly__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120")
    log: "output/logs/fit_diagnostics_bonly_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        OUT_BONLY=$(realpath {output.bonly})
        OUT_DIFF_BONLY=$(realpath {output.diff_bonly})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_BONLY)
        mkdir -p $(dirname $OUT_DIFF_BONLY)
        (
        echo "[$(date)] Starting fit_diagnostics_bonly rule with signal {params.signallabel}"

        # B-only parameters
        FREEZE_OPT_BONLY=""
        if [ -n "{params.freeze_parameters}" ]; then
            formatted_params=$(echo "{params.freeze_parameters} r{params.signallabel}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            FREEZE_OPT_BONLY="--freezeParameters $formatted_params"
        else
            FREEZE_OPT_BONLY="--freezeParameters r{params.signallabel}"
        fi

        SET_ZERO_OPT_BONLY=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero} r{params.signallabel}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            SET_ZERO_OPT_BONLY="--setParameters $formatted_params"
        else
            SET_ZERO_OPT_BONLY="--setParameters r{params.signallabel}=0"
        fi

        cd $DATACARD_DIR

        echo "[$(date)] Running FitDiagnostics B-only"
        combine -M FitDiagnostics $(basename {input}) \
            -m {params.mass} \
            --redefineSignalPOIs r{params.signallabel} \
            $SET_ZERO_OPT_BONLY \
            $FREEZE_OPT_BONLY \
            -n _$(basename {input} .root)_prefit_bonly \
            --saveShapes --saveWithUncertainties --plots

        echo "[$(date)] Running diffNuisances B-only"
        python3 $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py \
            -p r{params.signallabel} \
            -a fitDiagnostics_$(basename {input} .root)_prefit_bonly.root \
            -g diffNuisances_$(basename {input} .root)_prefit_bonly.root

        mkdir -p fitDiagnostics_bonly
        mv *prefit_bonly* fitDiagnostics_bonly/ 2>/dev/null || true
        ) 2>&1 | tee {log} && \
        cp $DATACARD_DIR/fitDiagnostics_bonly/fitDiagnostics_$(basename {input} .root)_prefit_bonly.root $OUT_BONLY && \
        cp $DATACARD_DIR/fitDiagnostics_bonly/diffNuisances_$(basename {input} .root)_prefit_bonly.root $OUT_DIFF_BONLY
        """

rule fit_diagnostics_sb:
    input: "{path}__{signallabel}.root"
    output:
        sb = "{path}_fitDiagnostics_sb__{signallabel}.root",
        diff_sb = "{path}_diffNuisances_sb__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120")
    log: "output/logs/fit_diagnostics_sb_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        OUT_SB=$(realpath {output.sb})
        OUT_DIFF_SB=$(realpath {output.diff_sb})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_SB)
        mkdir -p $(dirname $OUT_DIFF_SB)
        (
        echo "[$(date)] Starting fit_diagnostics_sb rule with signal {params.signallabel}"

        # S+B parameters
        FREEZE_OPT_SB=""
        if [ -n "{params.freeze_parameters}" ]; then
            formatted_params=$(echo "{params.freeze_parameters}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            FREEZE_OPT_SB="--freezeParameters $formatted_params"
        fi

        SET_ZERO_OPT_SB=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero} r{params.signallabel}" | tr ' ' '\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | sed 's/r{params.signallabel}=0/r{params.signallabel}=1/' | paste -sd, -)
            SET_ZERO_OPT_SB="--setParameters $formatted_params"
        else
            SET_ZERO_OPT_SB="--setParameters r{params.signallabel}=1"
        fi

        cd $DATACARD_DIR

        echo "[$(date)] Running FitDiagnostics S+B"
        combine -M FitDiagnostics $(basename {input}) \
            -m {params.mass} \
            --redefineSignalPOIs r{params.signallabel} \
            $SET_ZERO_OPT_SB \
            $FREEZE_OPT_SB \
            -n _$(basename {input} .root)_prefit_sb \
            --saveShapes --saveWithUncertainties --plots

        echo "[$(date)] Running diffNuisances S+B"
        python3 $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py \
            -p r{params.signallabel} \
            -a fitDiagnostics_$(basename {input} .root)_prefit_sb.root \
            -g diffNuisances_$(basename {input} .root)_prefit_sb.root

        mkdir -p fitDiagnostics_sb
        mv *prefit_sb* fitDiagnostics_sb/ 2>/dev/null || true
        ) 2>&1 | tee {log} && \
        cp $DATACARD_DIR/fitDiagnostics_sb/fitDiagnostics_$(basename {input} .root)_prefit_sb.root $OUT_SB && \
        cp $DATACARD_DIR/fitDiagnostics_sb/diffNuisances_$(basename {input} .root)_prefit_sb.root $OUT_DIFF_SB
        """

rule postfit:
    input:
        workspace = "{path}__{signallabel}.root",
        fit_result = "{path}_fitDiagnostics_bonly__{signallabel}.root"
    output: "{path}_postfit__{signallabel}.pdf"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        channel = "",
        signal = "",
        ylog = "",
        plot_script = config.get("postfit_plot_script", "src/stat_analysis/plots/make_postfit_plot.py"),
        metadata_template = lambda wildcards: config.get("metadata_template", "coffea4bees/stats_analysis/metadata/{channel}.yml")
    log: "output/logs/postfit_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input.workspace}))
        OUT_FILE=$(realpath {output})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_FILE)
        (
        # Run the plotting script from the Snakemake workspace root (not inside the datacard subfolder)
        METADATA_FILE=$(echo "{params.metadata_template}" | sed "s|{{channel}}|{params.channel}|g")
        python3 {params.plot_script} \
            -i {input.fit_result} \
            -o $DATACARD_DIR/plots/ \
            -c {params.channel} \
            -s {params.signal} \
            {params.ylog} \
            -m $METADATA_FILE
        ) 2>&1 | tee {log} && \
        cp $DATACARD_DIR/plots/postfitplots__{params.signallabel}__fit_s.pdf $OUT_FILE
        """

rule pdf_to_png:
    input: "{path}.pdf"
    output: "{path}.png"
    log: "output/logs/pdf_to_png_{path}.log"
    shell:
        """
        python3 src/plotting/pb_pdf_to_png.py {input} > {log} 2>&1
        """

rule split_impacts:
    input:
        pdf = "{path}_impacts__{signallabel}.pdf"
    output:
        dir = directory("{path}_impacts_split__{signallabel}")
    log: "output/logs/split_impacts_{path}__{signallabel}.log"
    shell:
        """
        LOG=$(pwd)/{log}
        mkdir -p $(dirname $LOG)
        (
        echo "[$(date)] Starting split_impacts rule"
        rm -rf {output.dir}
        mkdir -p {output.dir}
        pdfseparate {input.pdf} {output.dir}/page_%d.pdf
        for f in {output.dir}/*.pdf; do
            pdftocairo -singlefile -cropbox -png "$f" "${{f%.pdf}}"
        done
        ) > $LOG 2>&1
        """

