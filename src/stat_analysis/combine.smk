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
log_dir = os.path.join(config.get("output_path", "output"), "logs")
stat_only = config.get("stat_only", False)

targets = []
for channel, ch_config in config.get("channels", {}).items():
    signallabel = ch_config.get("signallabel")
    if not signallabel:
        continue
    # If run directly, the default path prefix is the channel out_base
    base_prefix = out_base
    targets.extend([
        f"{base_prefix}/limits/datacard_limits__{signallabel}.json",
        f"{base_prefix}/significance/datacard_significance__{signallabel}.log",
        f"{base_prefix}/likelihood_scan/datacard_likelihood_scan__{signallabel}.pdf",
        f"{base_prefix}/likelihood_scan/datacard_likelihood_scan__{signallabel}.png",
        f"{base_prefix}/postfit/datacard_postfit__{signallabel}.pdf",
        f"{base_prefix}/postfit/datacard_postfit__{signallabel}.png",
        f"{base_prefix}/postfit/datacard_fitDiagnostics_bonly__{signallabel}.root",
        f"{base_prefix}/postfit/datacard_fitDiagnostics_sb__{signallabel}.root",
        f"{base_prefix}/postfit/datacard_diffNuisances_bonly__{signallabel}.root",
        f"{base_prefix}/postfit/datacard_diffNuisances_sb__{signallabel}.root"
    ])
    if not stat_only:
        targets.extend([
            f"{base_prefix}/gof/datacard_gof__{signallabel}.pdf",
            f"{base_prefix}/gof/datacard_gof__{signallabel}.png",
            f"{base_prefix}/impacts/datacard_impacts__{signallabel}.pdf",
            f"{base_prefix}/impacts/datacard_impacts__{signallabel}.png",
            f"{base_prefix}/impacts/datacard_impacts_split__{signallabel}"
        ])

is_standalone = os.path.basename(workflow.main_snakefile) == "combine.smk"
if targets and is_standalone:
    rule all:
        input: targets

def get_channel_by_signal(wildcards):
    signallabel = wildcards.signallabel
    for channel, ch_config in config.get("channels", {}).items():
        if ch_config.get("signallabel") == signallabel or channel == signallabel:
            return channel
    return ""

def get_signal_by_channel(wildcards):
    channel = get_channel_by_signal(wildcards)
    return config.get("channels", {}).get(channel, {}).get("signal", "")

def get_workspace_input(wildcards):
    default_input = f"{wildcards.path}/workspace/datacard__{wildcards.signallabel}.txt"
    if os.path.exists(default_input):
        return default_input
    signallabel = wildcards.signallabel
    for channel, ch_config in config.get("channels", {}).items():
        if ch_config.get("signallabel") == signallabel or channel == signallabel:
            # Check if there is an explicit datacard name in cases config (for ZZ/ZH workflows)
            case_dc_name = f"datacard__{channel}"
            for case_key, case_info in config.get("cases", {}).items():
                if case_info.get("datacard"):
                    case_dc = os.path.basename(case_info["datacard"]).replace(".txt", "")
                    if case_key.upper() in channel.upper() or channel.upper() in case_key.upper():
                        case_dc_name = case_dc
                        break

            # If imported as a module, return the planned consolidated path to link the DAG.
            is_standalone = os.path.basename(workflow.main_snakefile) == "combine.smk"
            if not is_standalone:
                return os.path.join(wildcards.path, "datacards", f"{case_dc_name}.txt")

            # Check 1: new consolidated location (e.g. out_base/datacards/)
            for prefix in ["datacard__", "datacard_"]:
                path_to_check = os.path.join(wildcards.path, "datacards", f"{prefix}{channel}.txt")
                if os.path.exists(path_to_check):
                    return path_to_check
            # Check 2: workspace folder directly
            for prefix in ["datacard__", "datacard_"]:
                path_to_check = os.path.join(wildcards.path, "workspace", f"{prefix}{channel}.txt")
                if os.path.exists(path_to_check):
                    return path_to_check
            # Check 3: old location (2 levels up fallback)
            parent_dir = os.path.dirname(os.path.dirname(out_base))
            for prefix in ["datacard__", "datacard_"]:
                path_to_check = os.path.join(parent_dir, "datacards", channel, f"{prefix}{channel}.txt")
                if os.path.exists(path_to_check):
                    return path_to_check
            
            # Default fallback
            return os.path.join(wildcards.path, "datacards", f"{case_dc_name}.txt")
    return default_input

def get_poi_maps_dynamic(wildcards):
    signallabel = wildcards.signallabel
    for channel, ch_config in config.get("channels", {}).items():
        if ch_config.get("signallabel") == signallabel or channel == signallabel:
            actual_signal = ch_config.get("signallabel", channel)
            signals = [actual_signal] + ch_config.get("othersignal", "").split()
            poi_ranges = config.get("poi_ranges", "1,-10,10")
            return make_poi_maps(signals=signals, poi_ranges=poi_ranges)
    return ""


ruleorder: likelihood_scan_snapshot > likelihood_scan_chunk > impacts_initial_fit > gof_data > gof_toys_chunk > fit_diagnostics_bonly > fit_diagnostics_sb > workspace

localrules: pdf_to_png, split_impacts, impacts_collect

rule workspace:
    input: get_workspace_input
    output: "{path}/workspace/datacard__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        poi_maps = lambda wildcards: get_poi_maps_dynamic(wildcards),
        physics_model = lambda wildcards: config.get("physics_model", "HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel"),
        extra_t2w_args = lambda wildcards: config.get("extra_t2w_args", "--PO verbose")
    log: f"{log_dir}/workspace_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
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
    input: "{path}/workspace/datacard__{signallabel}.root"
    output: 
        txt="{path}/limits/datacard_limits__{signallabel}.txt",
        json="{path}/limits/datacard_limits__{signallabel}.json"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120")
    log: f"{log_dir}/limits_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        WORKSPACE_FILE=$(realpath {input})
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
        cd $(dirname $OUT_TXT) && \
            combine -M AsymptoticLimits $WORKSPACE_FILE \
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
    input: "{path}/workspace/datacard__{signallabel}.root"
    output: "{path}/significance/datacard_significance__{signallabel}.log"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120")
    log: f"{log_dir}/significance_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        WORKSPACE_FILE=$(realpath {input})
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

        cd $(dirname $OUT_FILE) && \
            combine -M Significance $WORKSPACE_FILE \
            -m {params.mass} \
            $SET_ZERO_OPT \
            $FREEZE_OPT \
            --redefineSignalPOIs r{params.signallabel} \
            -n _{params.signallabel} > $(basename {output}) && \
            combine -M Significance $WORKSPACE_FILE \
            -m {params.mass} \
            --redefineSignalPOIs r{params.signallabel} \
            $SET_ZERO_OPT \
            $FREEZE_OPT \
            -n _{params.signallabel} \
            -t -1 --expectSignal=1 >> $(basename {output})
        ) 2>&1 | tee {log}
        """

rule likelihood_scan_snapshot:
    input: "{path}/workspace/datacard__{signallabel}.root"
    output: temp("{path}/likelihood_scan/datacard_likelihood_scan_snapshot__{signallabel}.root")
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120")
    log: f"{log_dir}/likelihood_scan_snapshot_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        WORKSPACE_FILE=$(realpath {input})
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

        cd $(dirname $OUT_FILE) && \
            combine -M MultiDimFit -d $WORKSPACE_FILE \
            -m {params.mass} \
            -n _$(basename {input} .root)_snapshot \
            $SET_ZERO_OPT \
            $FREEZE_OPT \
            --saveWorkspace --robustFit 1 && \
            mv higgsCombine_$(basename {input} .root)_snapshot.MultiDimFit.mH{params.mass}.root $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule likelihood_scan_chunk:
    input: "{path}/likelihood_scan/datacard_likelihood_scan_snapshot__{signallabel}.root"
    output: temp("{path}/likelihood_scan/datacard_likelihood_scan_chunk_{split_index}__{signallabel}.root")
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
    log: f"{log_dir}/likelihood_scan_chunk_{{split_index}}_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        SNAPSHOT_FILE=$(realpath {input})
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

        cd $(dirname $OUT_FILE) && \
            combine -M MultiDimFit \
            -d $SNAPSHOT_FILE \
            -n _$(basename {input} .root)_chunk_{wildcards.split_index} \
            -m {params.mass} \
            -P r{params.signallabel} \
            $SET_ZERO_OPT \
            $FREEZE_OPT \
            --snapshotName MultiDimFit --rMin {params.r_min} --rMax {params.r_max} --algo grid --points {params.points} --firstPoint {params.first_point} --lastPoint {params.last_point} --alignEdges 1 && \
            mv higgsCombine_$(basename {input} .root)_chunk_{wildcards.split_index}.MultiDimFit.mH{params.mass}.root $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule likelihood_scan:
    input:
        lambda wildcards: get_likelihood_scan_chunks(wildcards, config)
    output: "{path}/likelihood_scan/datacard_likelihood_scan__{signallabel}.pdf"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        mass = lambda wildcards: config.get("mass", "120")
    log: f"{log_dir}/likelihood_scan_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        OUT_FILE=$(realpath {output})
        DATACARD_DIR=$(dirname $OUT_FILE)
        INPUT_FILES=""
        for f in {input}; do
            INPUT_FILES="$INPUT_FILES $(realpath $f)"
        done
        mkdir -p $(dirname $LOG)
        mkdir -p $DATACARD_DIR
        (
        echo "[$(date)] Merging likelihood scan chunks and plotting"
        cd $DATACARD_DIR && \
            hadd -f higgsCombine_merged_{params.signallabel}.MultiDimFit.mH{params.mass}.root \
            $INPUT_FILES && \
            plot1DScan.py higgsCombine_merged_{params.signallabel}.MultiDimFit.mH{params.mass}.root \
            --POI r{params.signallabel} -o scan_plot && \
            mv scan_plot.pdf $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule impacts_initial_fit:
    input: "{path}/workspace/datacard__{signallabel}.root"
    output: "{path}/impacts/datacard_initialFit__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        set_parameters_ranges = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "125"),
        r_min = lambda wildcards: config.get("r_min", "-10"),
        r_max = lambda wildcards: config.get("r_max", "10"),
        stat_only = lambda wildcards: config.get("stat_only", False)
    log: f"{log_dir}/impacts_initial_fit_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        WORKSPACE_FILE=$(realpath {input})
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
        cd $(dirname $OUT_FILE) && \
            combineTool.py -M Impacts -d $WORKSPACE_FILE \
            --doInitialFit --robustFit 1 -m {params.mass} \
            --redefineSignalPOIs r{params.signallabel} \
            --setParameterRanges r{params.signallabel}={params.r_min},{params.r_max}$SET_RANGES_OPT \
            $SET_ZERO_OPT \
            -n $(basename {input} .root) && \
            mv higgsCombine_initialFit_$(basename {input} .root).MultiDimFit.mH{params.mass}.root $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule impacts_do_fits:
    input:
        workspace = "{path}/workspace/datacard__{signallabel}.root",
        init_fit = "{path}/impacts/datacard_initialFit__{signallabel}.root"
    output:
        fits_dir = directory("{path}/impacts/datacard_impacts_fits__{signallabel}")
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
    log: f"{log_dir}/impacts_do_fits_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input.workspace}))
        WORKSPACE_FILE=$(realpath {input.workspace})
        INIT_FIT_FILE=$(realpath {input.init_fit})
        OUT_DIR=$(realpath {output.fits_dir})
        mkdir -p $(dirname $LOG)
        mkdir -p $OUT_DIR
        (
        echo "[$(date)] Starting impacts_do_fits rule with signal {params.signallabel}"

        # Check if running in stat_only mode or no_nuisances mode
        if [ "{params.stat_only}" = "True" ] || [ "{params.stat_only}" = "1" ] || [ "$(cat $INIT_FIT_FILE 2>/dev/null)" = "stat_only" ]; then
            echo "stat_only" > $OUT_DIR/stat_only
            exit 0
        fi

        if [ -f $INIT_FIT_FILE ] && [ "$(cat $INIT_FIT_FILE 2>/dev/null)" = "no_nuisances" ]; then
            echo "no_nuisances" > $OUT_DIR/no_nuisances
            exit 0
        fi

        # Copy the initial fit root file to OUT_DIR under the name combineTool expects
        cp $INIT_FIT_FILE $OUT_DIR/higgsCombine_initialFit_$(basename {input.workspace} .root).MultiDimFit.mH{params.mass}.root

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
        cd $OUT_DIR && \
            combineTool.py -M Impacts -d $WORKSPACE_FILE \
            --doFits --robustFit 1 -m {params.mass} --parallel {threads} \
            --redefineSignalPOIs r{params.signallabel} \
            --setParameterRanges r{params.signallabel}={params.r_min},{params.r_max}$SET_RANGES_OPT \
            $SET_ZERO_OPT \
            -n $(basename {input.workspace} .root) && \
            cp $INIT_FIT_FILE $OUT_DIR/
        ) 2>&1 | tee {log}
        """

rule impacts_collect:
    input:
        workspace = "{path}/workspace/datacard__{signallabel}.root",
        fits_done = "{path}/impacts/datacard_impacts_fits__{signallabel}"
    output:
        pdf = "{path}/impacts/datacard_impacts__{signallabel}.pdf"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        mass = lambda wildcards: config.get("mass", "125"),
        per_page = lambda wildcards: config.get("impacts_per_page", "20"),
        stat_only = lambda wildcards: config.get("stat_only", False)
    log: f"{log_dir}/impacts_collect_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input.workspace}))
        WORKSPACE_FILE=$(realpath {input.workspace})
        FITS_DONE_DIR=$(realpath {input.fits_done})
        OUT_FILE=$(realpath {output.pdf})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_FILE)
        (
        echo "[$(date)] Starting impacts_collect rule with signal {params.signallabel}"

        # Check if running in stat_only mode
        if [ "{params.stat_only}" = "True" ] || [ "{params.stat_only}" = "1" ] || [ -f $FITS_DONE_DIR/stat_only ]; then
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

        if [ -f $FITS_DONE_DIR/no_nuisances ]; then
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

        # Copy files from fits_done directory to the directory where we will collect them (impacts/)
        cd $(dirname $OUT_FILE) && \
            cp $FITS_DONE_DIR/higgsCombine_initialFit_$(basename {input.workspace} .root).MultiDimFit.mH{params.mass}.root ./ && \
            cp $FITS_DONE_DIR/higgsCombine_paramFit_$(basename {input.workspace} .root)_*.root ./

        echo "[$(date)] Running merging results"
        cd $(dirname $OUT_FILE) && \
            combineTool.py -M Impacts \
            -m {params.mass} -n $(basename {input.workspace} .root) \
            --redefineSignalPOIs r{params.signallabel} \
            -d $WORKSPACE_FILE \
            -o impacts_combine_$(basename {input.workspace} .root)_exp.json

        echo "[$(date)] Running creating pdf"
        cd $(dirname $OUT_FILE) && \
            plotImpacts.py -i impacts_combine_$(basename {input.workspace} .root)_exp.json \
            -o impacts_plot \
            --POI r{params.signallabel} \
            --per-page {params.per_page} --left-margin 0.3 --height 400 --label-size 0.04 && \
            mv impacts_plot.pdf $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule gof_data:
    input: "{path}/workspace/datacard__{signallabel}.root"
    output: "{path}/gof/datacard_gof_data__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120"),
        gof_algo = lambda wildcards: config.get("gof_algo", "saturated"),
        stat_only = lambda wildcards: config.get("stat_only", False)
    log: f"{log_dir}/gof_data_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        WORKSPACE_FILE=$(realpath {input})
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

        cd $(dirname $OUT_FILE) && \
            combine -M GoodnessOfFit $WORKSPACE_FILE \
            -m {params.mass} \
            --algo {params.gof_algo} \
            $SET_ZERO_OPT \
            -n _$(basename {input} .root)_{params.signallabel}_gof_data \
            2>&1 | tee gof_data_$(basename {input} .root)_{params.signallabel}.txt && \
            mv higgsCombine_$(basename {input} .root)_{params.signallabel}_gof_data.GoodnessOfFit.mH{params.mass}.root $(basename {output})
        ) 2>&1 | tee {log}
        """

rule gof_toys_chunk:
    input: "{path}/workspace/datacard__{signallabel}.root"
    output: "{path}/gof/datacard_gof_toys_{split_index}__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120"),
        toys_per_job = lambda wildcards: config.get("toys_per_job", "50"),
        gof_algo = lambda wildcards: config.get("gof_algo", "saturated"),
        seed = lambda wildcards: int(wildcards.split_index) + 123456,
        stat_only = lambda wildcards: config.get("stat_only", False)
    log: f"{log_dir}/gof_toys_chunk_{{split_index}}_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        WORKSPACE_FILE=$(realpath {input})
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

        cd $(dirname $OUT_FILE)
        # Check if there are any nuisance parameters
        TOYS_OPT="--toysFrequentist"
        NUISANCES=$(find $DATACARD_DIR -maxdepth 3 -name "*.txt" -exec grep -h "kmax" {{}} + 2>/dev/null | awk '{{print $2}}' | head -n 1)
        if [ "$NUISANCES" = "0" ] || [ -z "$NUISANCES" ]; then
            TOYS_OPT="--toysNoSystematics"
        fi

        combine -M GoodnessOfFit $WORKSPACE_FILE \
            -m {params.mass} \
            -t {params.toys_per_job} --algo {params.gof_algo} $TOYS_OPT \
            -s {params.seed} \
            $SET_ZERO_OPT \
            -n _$(basename {input} .root)_{params.signallabel}_gof_toys_{wildcards.split_index} \
            2>&1 | tee gof_toys_$(basename {input} .root)_{params.signallabel}_{wildcards.split_index}.txt && \
            mv higgsCombine_$(basename {input} .root)_{params.signallabel}_gof_toys_{wildcards.split_index}.GoodnessOfFit.mH{params.mass}.{params.seed}.root $(basename {output})
        ) 2>&1 | tee {log}
        """

rule gof:
    input:
        data = "{path}/gof/datacard_gof_data__{signallabel}.root",
        toys = lambda wildcards: [f"{wildcards.path}/gof/datacard_gof_toys_{i}__{wildcards.signallabel}.root" for i in range(int(config.get("num_toy_jobs", 10)))]
    output: "{path}/gof/datacard_gof__{signallabel}.pdf"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        mass = lambda wildcards: config.get("mass", "120"),
        gof_algo = lambda wildcards: config.get("gof_algo", "saturated"),
        stat_only = lambda wildcards: config.get("stat_only", False)
    log: f"{log_dir}/gof_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
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
    input: "{path}/workspace/datacard__{signallabel}.root"
    output:
        bonly = "{path}/postfit/datacard_fitDiagnostics_bonly__{signallabel}.root",
        diff_bonly = "{path}/postfit/datacard_diffNuisances_bonly__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120")
    log: f"{log_dir}/fit_diagnostics_bonly_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        WORKSPACE_FILE=$(realpath {input})
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

        cd $(dirname $OUT_BONLY)

        echo "[$(date)] Running FitDiagnostics B-only"
        combine -M FitDiagnostics $WORKSPACE_FILE \
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
            -g diffNuisances_$(basename {input} .root)_prefit_bonly.root \
            --skipFitB || touch diffNuisances_$(basename {input} .root)_prefit_bonly.root

        mkdir -p fitDiagnostics_bonly
        mv *prefit_bonly* fitDiagnostics_bonly/ 2>/dev/null || true
        mv fitDiagnostics_bonly/fitDiagnostics_$(basename {input} .root)_prefit_bonly.root $OUT_BONLY && \
        mv fitDiagnostics_bonly/diffNuisances_$(basename {input} .root)_prefit_bonly.root $OUT_DIFF_BONLY
        ) 2>&1 | tee {log}
        """

rule fit_diagnostics_sb:
    input: "{path}/workspace/datacard__{signallabel}.root"
    output:
        sb = "{path}/postfit/datacard_fitDiagnostics_sb__{signallabel}.root",
        diff_sb = "{path}/postfit/datacard_diffNuisances_sb__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120")
    log: f"{log_dir}/fit_diagnostics_sb_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        WORKSPACE_FILE=$(realpath {input})
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

        cd $(dirname $OUT_SB)

        echo "[$(date)] Running FitDiagnostics S+B"
        combine -M FitDiagnostics $WORKSPACE_FILE \
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
            -g diffNuisances_$(basename {input} .root)_prefit_sb.root || touch diffNuisances_$(basename {input} .root)_prefit_sb.root

        mkdir -p fitDiagnostics_sb
        mv *prefit_sb* fitDiagnostics_sb/ 2>/dev/null || true
        mv fitDiagnostics_sb/fitDiagnostics_$(basename {input} .root)_prefit_sb.root $OUT_SB && \
        mv fitDiagnostics_sb/diffNuisances_$(basename {input} .root)_prefit_sb.root $OUT_DIFF_SB
        ) 2>&1 | tee {log}
        """

rule postfit:
    input:
        workspace = "{path}/workspace/datacard__{signallabel}.root",
        fit_result = "{path}/postfit/datacard_fitDiagnostics_bonly__{signallabel}.root"
    output: "{path}/postfit/datacard_postfit__{signallabel}.pdf"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        channel = lambda wildcards: wildcards.path.rstrip('/').split('/')[-1],
        signal = "{signallabel}",
        ylog = lambda wildcards: "--log" if wildcards.path.rstrip('/').split('/')[-1] == "HH4b" else "",
        plot_script = config.get("postfit_plot_script", "src/stat_analysis/plots/make_postfit_plot.py"),
        metadata_template = lambda wildcards: config.get("metadata_template", "coffea4bees/stats_analysis/metadata/{channel}.yml")
    log: f"{log_dir}/postfit_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        OUT_FILE=$(realpath {output})
        OUT_DIR=$(realpath $(dirname {output}))
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_FILE)
        (
        # Run the plotting script from the Snakemake workspace root (not inside the datacard subfolder)
        METADATA_FILE=$(echo "{params.metadata_template}" | sed "s|{{channel}}|{params.channel}|g")
        python3 {params.plot_script} \
            -i {input.fit_result} \
            -o $OUT_DIR/plots/ \
            -c {params.channel} \
            -s {params.signal} \
            {params.ylog} \
            -m $METADATA_FILE && \
            cp $OUT_DIR/plots/postfitplots__{params.signallabel}__fit_s.pdf $OUT_FILE
        ) 2>&1 | tee {log}
        """

rule pdf_to_png:
    input: "{path}.pdf"
    output: "{path}.png"
    log: f"{log_dir}/pdf_to_png_{{path}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
        python3 src/plotting/pb_pdf_to_png.py {input} > {log} 2>&1
        """

rule split_impacts:
    input:
        pdf = "{path}/impacts/datacard_impacts__{signallabel}.pdf"
    output:
        dir = directory("{path}/impacts/datacard_impacts_split__{signallabel}")
    log: f"{log_dir}/split_impacts_{{path}}__{{signallabel}}.log"
    shell:
        """
        . /srv/apptainer_env.sh || true
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

