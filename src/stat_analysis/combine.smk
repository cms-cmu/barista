import os
import sys
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from src.stat_analysis.helpers import make_poi_maps, get_default_othersignals, get_grid_split_points, get_likelihood_scan_chunks


# Resolve combine container image dynamically based on CVMFS availability
COMBINE_IMAGE = "docker://gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0"
if os.path.exists("/cvmfs/unpacked.cern.ch"):
    COMBINE_IMAGE = f"/cvmfs/unpacked.cern.ch/{COMBINE_IMAGE.replace('docker://', '')}"



rule workspace:
    input: "{path}.txt"
    output: "{path}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        poi_maps = lambda wildcards: "",
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
        DATACARD_DIR=$(realpath $(dirname {input}))
        mkdir -p $(dirname $LOG)
        TMPOUT=$(mktemp /tmp/workspace_XXXXXX.root)
        (
        echo "[$(date)] Starting workspace rule"
        cd $DATACARD_DIR && \
            text2workspace.py $(basename {input}) \
            -P {params.physics_model} \
            {params.poi_maps} \
            {params.extra_t2w_args} \
            -o $TMPOUT && \
            rootls $TMPOUT
        ) 2>&1 | tee {log}
        test -s $TMPOUT || {{ echo "ERROR: workspace tmp output missing or empty" >&2; exit 1; }}
        cp $TMPOUT {output}
        rm -f $TMPOUT
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
        mkdir -p $(dirname $LOG)
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
            > $(basename {output.txt}) && \
        echo "[$(date)] Running CollectLimits" && \
            combineTool.py -M CollectLimits \
            higgsCombine_{params.signallabel}.AsymptoticLimits.mH{params.mass}.root \
            -o $(basename {output.json})
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
        mkdir -p $(dirname $LOG)
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
        ) 2>&1 | tee {log}
        cp $(dirname {input})/higgsCombine_$(basename {input} .root)_snapshot.MultiDimFit.mH{params.mass}.root {output}
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
        mkdir -p $(dirname $LOG)
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
        ) 2>&1 | tee {log}
        cp $(dirname {input})/higgsCombine_$(basename {input} .root)_chunk_{wildcards.split_index}.MultiDimFit.mH{params.mass}.root {output}
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
        DATACARD_DIR=$(realpath $(dirname {output}))
        mkdir -p $(dirname $LOG)
        (
        echo "[$(date)] Merging likelihood scan chunks and plotting"
        cd $DATACARD_DIR && \
            hadd -f higgsCombine_merged_{params.signallabel}.MultiDimFit.mH{params.mass}.root \
            $(for f in {input}; do basename $f; done) && \
            plot1DScan.py higgsCombine_merged_{params.signallabel}.MultiDimFit.mH{params.mass}.root \
            --POI r{params.signallabel} -o $(basename {output} .pdf)
        ) 2>&1 | tee {log}
        """



rule impacts:
    input: "{path}__{signallabel}.root"
    output: "{path}_impacts__{signallabel}.pdf"
    container: config.get("combine_container", COMBINE_IMAGE)
    threads: int(config.get("impacts_parallel", "4"))
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        set_parameters_ranges = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "125"),
        parallel = lambda wildcards: config.get("impacts_parallel", "4"),
        per_page = lambda wildcards: config.get("impacts_per_page", "20"),
        r_min = lambda wildcards: config.get("r_min", "-10"),
        r_max = lambda wildcards: config.get("r_max", "10")
    log: "output/logs/impacts_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        mkdir -p $(dirname $LOG)
        (
        echo "[$(date)] Starting impacts rule with signal {params.signallabel}"

        # Check if there are any nuisance parameters
        NUISANCES=$(grep -h "kmax" $DATACARD_DIR/*.txt 2>/dev/null | awk '{{print $2}}' | head -n 1)
        if [ "$NUISANCES" = "0" ] || [ -z "$NUISANCES" ]; then
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
            python3 dummy_plot.py "$DATACARD_DIR/$(basename {{output}})"
            rm dummy_plot.py
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

        cd $DATACARD_DIR && \
            combineTool.py -M Impacts -d $(basename {input}) \
            --doInitialFit --robustFit 1 -m {params.mass} \
            --redefineSignalPOIs r{params.signallabel} \
            --setParameterRanges r{params.signallabel}={params.r_min},{params.r_max}$SET_RANGES_OPT \
            $SET_ZERO_OPT \
            -n $(basename {input} .root) && \
            combineTool.py -M Impacts -d $(basename {input}) \
            --doFits --robustFit 1 -m {params.mass} --parallel {threads} \
            --redefineSignalPOIs r{params.signallabel} \
            --setParameterRanges r{params.signallabel}={params.r_min},{params.r_max}$SET_RANGES_OPT \
            $SET_ZERO_OPT \
            -n $(basename {input} .root) && \
            combineTool.py -M Impacts \
            -m {params.mass} -n $(basename {input} .root) \
            --redefineSignalPOIs r{params.signallabel} \
            -d $(basename {input}) \
            -o impacts_combine_$(basename {input} .root)_exp.json && \
            plotImpacts.py -i impacts_combine_$(basename {input} .root)_exp.json \
            -o $(basename {output} .pdf) \
            --POI r{params.signallabel} \
            --per-page {params.per_page} --left-margin 0.3 --height 400 --label-size 0.04
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
        gof_algo = lambda wildcards: config.get("gof_algo", "saturated")
    log: "output/logs/gof_data_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        mkdir -p $(dirname $LOG)
        (
        echo "[$(date)] Starting gof_data rule with signal {params.signallabel}"

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
            cp higgsCombine_$(basename {input} .root)_{params.signallabel}_gof_data.GoodnessOfFit.mH{params.mass}.root $(basename {output})
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
        seed = lambda wildcards: int(wildcards.split_index) + 123456
    log: "output/logs/gof_toys_chunk_{split_index}_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        mkdir -p $(dirname $LOG)
        (
        echo "[$(date)] Starting gof_toys_chunk {wildcards.split_index} rule with signal {params.signallabel}"

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
        NUISANCES=$(grep -h "kmax" $DATACARD_DIR/*.txt 2>/dev/null | awk '{{print $2}}' | head -n 1)
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
            cp higgsCombine_$(basename {input} .root)_{params.signallabel}_gof_toys_{wildcards.split_index}.GoodnessOfFit.mH{params.mass}.{params.seed}.root $(basename {output})
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
        gof_algo = lambda wildcards: config.get("gof_algo", "saturated")
    log: "output/logs/gof_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input.data}))
        mkdir -p $(dirname $LOG)
        (
        echo "[$(date)] Merging gof toys and plotting GoF saturated distribution"
        cd $DATACARD_DIR && \
            combineTool.py -M CollectGoodnessOfFit \
            --input $(basename {input.data}) $(for f in {input.toys}; do basename $f; done) \
            -o gof_$(basename {input.data} _gof_data__{params.signallabel}.root)_{params.signallabel}.json && \
            plotGof.py gof_$(basename {input.data} _gof_data__{params.signallabel}.root)_{params.signallabel}.json \
            --statistic {params.gof_algo} --mass {params.mass}.0 \
            --output $(basename {output} .pdf)
        ) 2>&1 | tee {log}
        """

rule fit_diagnostics:
    input: "{path}__{signallabel}.root"
    output: "{path}_fitDiagnostics_bonly__{signallabel}.root"
    container: config.get("combine_container", COMBINE_IMAGE)
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = lambda wildcards: get_default_othersignals(wildcards, config),
        freeze_parameters = lambda wildcards: get_default_othersignals(wildcards, config),
        mass = lambda wildcards: config.get("mass", "120")
    log: "output/logs/fit_diagnostics_{path}__{signallabel}.log"
    shell:
        """
        if [ "${{SLURM_PROCID:-0}}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input}))
        mkdir -p $(dirname $LOG)
        (
        echo "[$(date)] Starting fit_diagnostics rule with signal {params.signallabel}"

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
            combine -M FitDiagnostics $(basename {input}) \
            -m {params.mass} \
            --redefineSignalPOIs r{params.signallabel} \
            $SET_ZERO_OPT \
            $FREEZE_OPT \
            -n _$(basename {input} .root)_prefit_bonly \
            --saveShapes --saveWithUncertainties --plots
        mkdir -p fitDiagnostics_bonly
        mv *th1x* fitDiagnostics_bonly/ 2>/dev/null || true
        mv covariance* fitDiagnostics_bonly/ 2>/dev/null || true
        ) 2>&1 | tee {log}
        cp $DATACARD_DIR/fitDiagnostics_$(basename {input} .root)_prefit_bonly.root {output}
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
        mkdir -p $(dirname $LOG)
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
        ) 2>&1 | tee {log}
        cp $DATACARD_DIR/plots/postfitplots__{params.signallabel}__fit_s.pdf {output}
        """
