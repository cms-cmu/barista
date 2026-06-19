#!/usr/bin/env python3
import os
import sys
import re
import subprocess
import glob
import getpass
from snakemake.utils import read_job_properties

TEMPLATES = {
    "workspace": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        set -o pipefail
        LOG=$(pwd)/{log}
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname {output})

        python3 -c '
import os, shutil, glob
datacard = "{input}"
out_dir = os.path.dirname("{output}")
in_dir = os.path.dirname(datacard)
for ext in ["*.txt", "*.root"]:
    for f in glob.glob(os.path.join(in_dir, ext)):
        shutil.copy(f, os.path.join(out_dir, os.path.basename(f)))
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
    """,
    "limits": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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
            formatted_params=$(echo "{params.freeze_parameters}" | tr ' ' '\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                FREEZE_OPT="--freezeParameters $formatted_params"
            fi
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
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
    """,
    "significance": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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
            formatted_params=$(echo "{params.freeze_parameters}" | tr ' ' '\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                FREEZE_OPT="--freezeParameters $formatted_params"
            fi
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
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
    """,
    "likelihood_scan_snapshot": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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
            formatted_params=$(echo "{params.freeze_parameters}" | tr ' ' '\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                FREEZE_OPT="--freezeParameters $formatted_params"
            fi
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
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
    """,
    "likelihood_scan_chunk": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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
            formatted_params=$(echo "{params.freeze_parameters}" | tr ' ' '\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                FREEZE_OPT="--freezeParameters $formatted_params"
            fi
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
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
    """,
    "likelihood_scan": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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
    """,
    "impacts_initial_fit": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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

        # Check if there are any nuisance parameters
        if [ "{has_nuisances}" = "0" ]; then
            echo "no_nuisances" > $OUT_FILE
            exit 0
        fi

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\\\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                SET_ZERO_OPT="--setParameters $formatted_params"
            fi
        fi

        SET_RANGES_OPT=""
        if [ -n "{params.set_parameters_ranges}" ]; then
            formatted_params=$(echo "{params.set_parameters_ranges}" | tr ' ' '\\\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0,0/' | paste -sd: -)
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
    """,
    "impacts_do_fits": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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

        if [ -f {input.init_fit} ] && [ "$(cat {input.init_fit} 2>/dev/null)" = "no_nuisances" ]; then
            echo "no_nuisances" > $OUT_DIR/no_nuisances
            exit 0
        fi

        # Copy the initial fit root file back to DATACARD_DIR so combineTool can find it
        cp $(realpath {input.init_fit}) $DATACARD_DIR/higgsCombine_initialFit_$(basename {input.workspace} .root).MultiDimFit.mH{params.mass}.root

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\\\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                SET_ZERO_OPT="--setParameters $formatted_params"
            fi
        fi

        SET_RANGES_OPT=""
        if [ -n "{params.set_parameters_ranges}" ]; then
            formatted_params=$(echo "{params.set_parameters_ranges}" | tr ' ' '\\\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0,0/' | paste -sd: -)
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
    """,
    "gof_data": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
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
        echo "[$(date)] Completed gof_data rule with signal {params.signallabel}"
        ) 2>&1 | tee {log}
    """,
    "gof_toys_chunk": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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

        SET_ZERO_OPT=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero}" | tr ' ' '\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            if [ -n "$formatted_params" ]; then
                SET_ZERO_OPT="--setParameters $formatted_params"
            fi
        fi

        cd $DATACARD_DIR
        # Check if there are any nuisance parameters
        TOYS_OPT="--toysFrequentist"
        if [ "{has_nuisances}" = "0" ]; then
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
        echo "[$(date)] Completed gof_toys_chunk {wildcards.split_index} rule with signal {params.signallabel}"
        ) 2>&1 | tee {log}
    """,
    "gof": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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
        cd $DATACARD_DIR && \
            combineTool.py -M CollectGoodnessOfFit \
            --input $(basename {input.data}) $(for f in {input.toys}; do basename $f; done) \
            -o gof_$(basename {input.data} _gof_data__{params.signallabel}.root)_{params.signallabel}.json && \
            plotGof.py gof_$(basename {input.data} _gof_data__{params.signallabel}.root)_{params.signallabel}.json \
            --statistic {params.gof_algo} --mass {params.mass}.0 \
            --output gof_plot && \
            mv gof_plot.pdf $OUT_FILE
        echo "[$(date)] Completed gof rule"
        ) 2>&1 | tee {log}
    """,
    "fit_diagnostics_bonly": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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
            formatted_params=$(echo "{params.freeze_parameters} r{params.signallabel}" | tr ' ' '\\\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            FREEZE_OPT_BONLY="--freezeParameters $formatted_params"
        else
            FREEZE_OPT_BONLY="--freezeParameters r{params.signallabel}"
        fi

        SET_ZERO_OPT_BONLY=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero} r{params.signallabel}" | tr ' ' '\\\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | paste -sd, -)
            SET_ZERO_OPT_BONLY="--setParameters $formatted_params"
        else
            SET_ZERO_OPT_BONLY="--setParameters r{params.signallabel}=0"
        fi

        cd $DATACARD_DIR

        echo "[$(date)] Running FitDiagnostics B-only"
        combine -M FitDiagnostics $(basename {input}) \\
            -m {params.mass} \\
            --redefineSignalPOIs r{params.signallabel} \\
            $SET_ZERO_OPT_BONLY \\
            $FREEZE_OPT_BONLY \\
            -n _$(basename {input} .root)_prefit_bonly \\
            --saveShapes --saveWithUncertainties --plots

        echo "[$(date)] Running diffNuisances B-only"
        python3 $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py \\
            -p r{params.signallabel} \\
            -a fitDiagnostics_$(basename {input} .root)_prefit_bonly.root \\
            -g diffNuisances_$(basename {input} .root)_prefit_bonly.root

        mkdir -p fitDiagnostics_bonly
        mv *prefit_bonly* fitDiagnostics_bonly/ 2>/dev/null || true
        ) 2>&1 | tee {log} && \\
        cp $DATACARD_DIR/fitDiagnostics_bonly/fitDiagnostics_$(basename {input} .root)_prefit_bonly.root $OUT_BONLY && \\
        cp $DATACARD_DIR/fitDiagnostics_bonly/diffNuisances_$(basename {input} .root)_prefit_bonly.root $OUT_DIFF_BONLY
    """,
    "fit_diagnostics_sb": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
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
            formatted_params=$(echo "{params.freeze_parameters}" | tr ' ' '\\\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | paste -sd, -)
            FREEZE_OPT_SB="--freezeParameters $formatted_params"
        fi

        SET_ZERO_OPT_SB=""
        if [ -n "{params.set_parameters_zero}" ]; then
            formatted_params=$(echo "{params.set_parameters_zero} r{params.signallabel}" | tr ' ' '\\\\n' | sed '/^$/d' | sed 's/^r//' | sed 's/^/r/' | sed 's/$/=0/' | sed 's/r{params.signallabel}=0/r{params.signallabel}=1/' | paste -sd, -)
            SET_ZERO_OPT_SB="--setParameters $formatted_params"
        else
            SET_ZERO_OPT_SB="--setParameters r{params.signallabel}=1"
        fi

        cd $DATACARD_DIR

        echo "[$(date)] Running FitDiagnostics S+B"
        combine -M FitDiagnostics $(basename {input}) \\
            -m {params.mass} \\
            --redefineSignalPOIs r{params.signallabel} \\
            $SET_ZERO_OPT_SB \\
            $FREEZE_OPT_SB \\
            -n _$(basename {input} .root)_prefit_sb \\
            --saveShapes --saveWithUncertainties --plots

        echo "[$(date)] Running diffNuisances S+B"
        python3 $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py \\
            -p r{params.signallabel} \\
            -a fitDiagnostics_$(basename {input} .root)_prefit_sb.root \\
            -g diffNuisances_$(basename {input} .root)_prefit_sb.root

        mkdir -p fitDiagnostics_sb
        mv *prefit_sb* fitDiagnostics_sb/ 2>/dev/null || true
        ) 2>&1 | tee {log} && \\
        cp $DATACARD_DIR/fitDiagnostics_sb/fitDiagnostics_$(basename {input} .root)_prefit_sb.root $OUT_SB && \\
        cp $DATACARD_DIR/fitDiagnostics_sb/diffNuisances_$(basename {input} .root)_prefit_sb.root $OUT_DIFF_SB
    """,
    "postfit": """
        if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
            echo "Skipping duplicate Slurm task (SLURM_PROCID=$SLURM_PROCID)"
            exit 0
        fi
        LOG=$(pwd)/{log}
        DATACARD_DIR=$(realpath $(dirname {input.workspace}))
        OUT_FILE=$(realpath {output})
        mkdir -p $(dirname $LOG)
        mkdir -p $(dirname $OUT_FILE)
        (
        # Run postfit plot script from Snakemake workspace root (plot script path is relative to workspace root)
        METADATA_FILE=$(echo "{params.metadata_template}" | sed "s|{channel}|{params.channel}|g")
        python3 {params.plot_script} \\
            -i {input.fit_result} \\
            -o $DATACARD_DIR/plots/ \\
            -c {params.channel} \\
            -s {params.signal} \\
            {params.ylog} \\
            -m $METADATA_FILE
        ) 2>&1 | tee {log} && \\
        cp $DATACARD_DIR/plots/postfitplots__{params.signallabel}__fit_s.pdf $OUT_FILE
    """
}

def parse_datacard_shapes(datacard_path):
    """
    Parse a datacard text file to find references to shape root files.
    Returns a list of resolved shape root file paths.
    """
    if not os.path.exists(datacard_path):
        return []
    shape_files = set()
    datacard_dir = os.path.dirname(datacard_path)
    with open(datacard_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Look for lines starting with "shapes"
            if line.startswith('shapes'):
                tokens = line.split()
                for token in tokens:
                    if token.endswith('.root'):
                        # Resolve path relative to datacard directory
                        full_path = os.path.join(datacard_dir, token)
                        if os.path.exists(full_path):
                            shape_files.add(os.path.normpath(full_path))
                        # Also check if it matches wildcard/is a basename and exists directly
                        elif os.path.exists(token):
                            shape_files.add(os.path.normpath(token))
    return list(shape_files)

def format_template(template, job_properties):
    """
    Interpolates Snakemake job properties into the rule shell command template.
    """
    inputs = job_properties.get("input", [])
    outputs = job_properties.get("output", [])
    params = job_properties.get("params", {})
    wildcards = job_properties.get("wildcards", {})
    log = job_properties.get("log", [""])[0]
    threads = job_properties.get("threads", 1)
    
    # Determine if there are nuisance parameters (for gof and impacts dummy fallback)
    has_nuisances = "1"
    txt_files = []
    if inputs:
        inp_list = inputs if isinstance(inputs, list) else list(inputs.values())
        if inp_list:
            input_dir = os.path.dirname(inp_list[0])
            txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
            
    if not txt_files:
        output_base = get_output_base_dir(job_properties)
        txt_files = glob.glob(os.path.join(output_base, "..", "**", "*.txt"), recursive=True)
        
    if txt_files:
        for txt_file in txt_files:
            try:
                with open(txt_file, "r") as f:
                    content = f.read()
                    if "kmax 0" in content:
                        has_nuisances = "0"
                        break
            except Exception:
                pass

    fmt_dict = {
        "log": log,
        "threads": threads,
        "has_nuisances": has_nuisances,
    }
    
    # Add input mappings
    if isinstance(inputs, list):
        for idx, inp in enumerate(inputs):
            fmt_dict[f"input_{idx}"] = inp
        if len(inputs) == 1:
            fmt_dict["input"] = inputs[0]
        elif len(inputs) > 1:
            fmt_dict["input"] = " ".join(inputs) # Join all inputs with space
    elif isinstance(inputs, dict):
        for key, val in inputs.items():
            if isinstance(val, (list, tuple)):
                fmt_dict[f"input.{key}"] = " ".join(val)
            else:
                fmt_dict[f"input.{key}"] = val
            
    # Add output mappings
    if isinstance(outputs, list):
        for idx, out in enumerate(outputs):
            fmt_dict[f"output_{idx}"] = out
        if len(outputs) == 1:
            fmt_dict["output"] = outputs[0]
        elif len(outputs) > 1:
            fmt_dict["output"] = outputs[0] # Default fallback
    elif isinstance(outputs, dict):
        for key, val in outputs.items():
            fmt_dict[f"output.{key}"] = val
            
    # Add wildcards mappings
    for key, val in wildcards.items():
        fmt_dict[f"wildcards.{key}"] = val
        
    # Add params mappings
    for key, val in params.items():
        if isinstance(val, (list, tuple)):
            fmt_dict[f"params.{key}"] = " ".join(map(str, val))
        else:
            fmt_dict[f"params.{key}"] = val
        
    # Specific rule parameter overrides for Snakemake dict-like access
    rule = job_properties["rule"]
    if rule == "likelihood_grid_scan":
        fmt_dict["input.workspace"] = inputs[0]
        fmt_dict["input.init_fit"] = inputs[1]
    elif rule == "gof_collect":
        fmt_dict["input.workspace"] = inputs[0]
        fmt_dict["input.data_root"] = inputs[1]
        fmt_dict["input.toy_roots"] = " ".join(inputs[2:])
    elif rule == "gof":
        fmt_dict["input.data"] = inputs[0]
        fmt_dict["input.toys"] = " ".join(inputs[1:])
    elif rule == "postfit":
        fmt_dict["input.workspace"] = inputs[0]
        fmt_dict["input.fit_result"] = inputs[1]
    elif rule == "impacts_do_fits":
        fmt_dict["input.workspace"] = inputs[0]
        fmt_dict["input.init_fit"] = inputs[1]
    elif rule == "impacts_collect":
        fmt_dict["input.workspace"] = inputs[0]
        fmt_dict["input.fits_done"] = inputs[1]
        
    if rule == "limits":
        fmt_dict["output.txt"] = outputs[0]
        fmt_dict["output.json"] = outputs[1]
    elif rule == "fit_diagnostics_bonly":
        fmt_dict["output.bonly"] = outputs[0]
        fmt_dict["output.diff_bonly"] = outputs[1]
    elif rule == "fit_diagnostics_sb":
        fmt_dict["output.sb"] = outputs[0]
        fmt_dict["output.diff_sb"] = outputs[1]

        
    # Formatting using regex to handle dotted keys (e.g. {input.workspace})
    def replace_match(match):
        key = match.group(1)
        return str(fmt_dict.get(key, match.group(0)))
        
    pattern = re.compile(r'(?<!{){([^{}]+)}(?!})')
    
    cmd = template
    for _ in range(3):
        cmd = pattern.sub(replace_match, cmd)
        
    return cmd

def get_output_base_dir(job_properties):
    """
    Dynamically extract the base output directory (e.g. output/v4_systematics_test/HH4b)
    from log or output paths in job_properties.
    """
    log = job_properties.get("log", [""])[0]
    if log:
        parts = log.split("/")
        if "logs" in parts:
            idx = parts.index("logs")
            return "/".join(parts[:idx])
            
    outputs = job_properties.get("output", [])
    for out in outputs:
        # Check if the output has subdirectories like limits, workspace, gof, etc.
        parts = out.split("/")
        for subd in ["workspace", "limits", "significance", "likelihood_scan", "gof", "impacts", "postfit"]:
            if subd in parts:
                idx = parts.index(subd)
                return "/".join(parts[:idx])
    return "output"

def main():
    jobscript = sys.argv[-1]
    job_properties = read_job_properties(jobscript)
    

    rule = job_properties.get("rule", "unknown")
    jobid = job_properties.get("jobid", 0)

    # Determine if this rule should run on HTCondor or locally
    config = job_properties.get("config", {})
    run_on_condor_val = config.get("run_on_condor", False)
    run_on_condor = str(run_on_condor_val).lower() in ("true", "1", "yes")

    CONDOR_ELIGIBLE_RULES = {
        "likelihood_scan_snapshot",
        "likelihood_scan_chunk",
        "likelihood_scan",
        "impacts_initial_fit",
        "impacts_do_fits",
        "gof_data",
        "gof_toys_chunk",
        "gof",
        "fit_diagnostics_bonly",
        "fit_diagnostics_sb"
    }

    if not (run_on_condor and rule in CONDOR_ELIGIBLE_RULES):
        print(f"[submit_wrapper] Running rule '{rule}' locally on login node (run_on_condor={run_on_condor})...", file=sys.stderr)
        try:
            result = subprocess.run(["/bin/bash", jobscript], stdout=sys.stderr)
            print(f"local_job_{rule}_{jobid}")
            sys.exit(result.returncode)
        except Exception as e:
            print(f"[submit_wrapper] Local execution of rule '{rule}' failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Track inputs/outputs
    raw_inputs = job_properties.get("input", [])
    inputs = list(raw_inputs.values()) if isinstance(raw_inputs, dict) else list(raw_inputs)
    raw_outputs = job_properties.get("output", [])
    outputs = list(raw_outputs.values()) if isinstance(raw_outputs, dict) else list(raw_outputs)
    log = job_properties.get("log", [""])[0]
    
    # Route Condor logs to output log directory dynamically
    output_base_dir = get_output_base_dir(job_properties)
    if log:
        log_dir = os.path.join(os.path.dirname(log), "condor")
    else:
        log_dir = os.path.join(output_base_dir, "logs", "condor")
        
    # Create necessary directories
    os.makedirs(log_dir, exist_ok=True)
    job_dir = os.path.join(output_base_dir, "condor_jobs")
    os.makedirs(job_dir, exist_ok=True)
    
    if log:
        outputs.append(log)
    params = job_properties.get("params", {})
    threads = job_properties.get("threads", 1)
    resources = job_properties.get("resources", {})
    
    # We do NOT need to transfer the Snakemake jobscript since we run raw command directly!
    transfer_inputs = []
    
    # For relative directory mapping on worker node
    input_rel_paths = []
    output_rel_paths = []
    
    # 1. Process inputs
    for inp in inputs:
        inp_norm = os.path.normpath(inp)
        if os.path.exists(inp_norm):
            transfer_inputs.append(os.path.realpath(inp_norm))
            input_rel_paths.append(inp_norm)
            
            # Special case: if this is the workspace rule, we parse the datacard for shapes
            if rule == "workspace" and inp_norm.endswith(".txt"):
                shapes = parse_datacard_shapes(inp_norm)
                for shape in shapes:
                    real_shape = os.path.realpath(shape)
                    if real_shape not in transfer_inputs:
                        transfer_inputs.append(real_shape)
                        input_rel_paths.append(shape)

    # 2. Process custom parameters (e.g. plot_script, metadata_template)
    # For postfit_plot rule
    if rule in ["postfit_plot", "postfit"]:
        plot_script = params.get("plot_script")
        if plot_script and os.path.exists(plot_script):
            plot_script_norm = os.path.normpath(plot_script)
            transfer_inputs.append(os.path.realpath(plot_script_norm))
            input_rel_paths.append(plot_script_norm)
            
            # Transfer cmsstyle.py if it exists in the same directory
            plot_dir = os.path.dirname(plot_script_norm)
            cmsstyle_path = os.path.join(plot_dir, "cmsstyle.py")
            if os.path.exists(cmsstyle_path):
                transfer_inputs.append(os.path.realpath(cmsstyle_path))
                input_rel_paths.append(cmsstyle_path)
            
        metadata_template = params.get("metadata_template")
        channel = params.get("channel")
        if metadata_template and channel:
            metadata_file = metadata_template.replace("{channel}", channel)
            metadata_file_norm = os.path.normpath(metadata_file)
            if os.path.exists(metadata_file_norm):
                transfer_inputs.append(os.path.realpath(metadata_file_norm))
                input_rel_paths.append(metadata_file_norm)

    # 3. Process impacts_collect rule inputs (transfer fit root files to execute node)
    if rule == "impacts_collect":
        workspace_path = os.path.normpath(inputs[0])
        workspace_dir = os.path.dirname(workspace_path)
        # Find all paramFit and initialFit root files
        extra_inputs = glob.glob(os.path.join(workspace_dir, "higgsCombine_paramFit_*.root"))
        extra_inputs += glob.glob(os.path.join(workspace_dir, "higgsCombine_initialFit_*.root"))
        for extra in extra_inputs:
            extra_norm = os.path.normpath(extra)
            real_extra = os.path.realpath(extra_norm)
            if real_extra not in transfer_inputs:
                transfer_inputs.append(real_extra)
                input_rel_paths.append(extra_norm)

    # Make transfer inputs unique while preserving order
    seen = set()
    transfer_inputs = [x for x in transfer_inputs if not (x in seen or seen.add(x))]
    
    # 4. Process outputs
    for out in outputs:
        out_norm = os.path.normpath(out)
        output_rel_paths.append(out_norm)
        
    output_transfers = list(output_rel_paths)
    # Special case for impacts_do_fits: copy back the whole directory to get the fit root files
    if rule == "impacts_do_fits":
        workspace_path = os.path.normpath(inputs[0])
        workspace_dir = os.path.dirname(workspace_path)
        if workspace_dir and workspace_dir not in output_transfers:
            output_transfers.append(workspace_dir)

    # Directories to create on execute node
    execute_dirs = set()
    for rel_path in input_rel_paths + output_rel_paths:
        dirname = os.path.dirname(rel_path)
        if dirname:
            execute_dirs.add(dirname)
            
    # Sorted by depth so parents are created first
    execute_dirs = sorted(list(execute_dirs), key=len)
    
    # Unique names for the wrapper script and JDL
    wrapper_path = os.path.join(job_dir, f"job_{rule}_{jobid}.sh")
    jdl_path = os.path.join(job_dir, f"job_{rule}_{jobid}.jdl")
    
    # Format the rule template shell command
    rule_template = TEMPLATES.get(rule)
    if not rule_template:
        print(f"Error: No shell template found for rule {rule}", file=sys.stderr)
        sys.exit(1)
        
    rule_command = format_template(rule_template, job_properties)
    
    # Write wrapper script
    with open(wrapper_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -eo pipefail\n")
        f.write("echo '=================== HOST INFO =================== '\n")
        f.write("uname -a\n")
        f.write("pwd\n")
        f.write("ls -la\n")
        f.write("echo '================================================= '\n\n")
        
        # Set up CMSSW environment inside container on worker node
        f.write("# Set up CMSSW environment\n")
        f.write("if [ -f /cvmfs/cms.cern.ch/cmsset_default.sh ]; then\n")
        f.write("    echo '[$(date)] Sourcing CMSSW environment...'\n")
        f.write("    source /cvmfs/cms.cern.ch/cmsset_default.sh\n")
        f.write("    cd /home/cmsusr/CMSSW_14_1_0_pre4/\n")
        f.write("    eval $(scramv1 runtime -sh)\n")
        f.write("    cd - > /dev/null\n")
        f.write("fi\n\n")
        
        # Create directories
        if execute_dirs:
            f.write("# Recreate directories\n")
            for d in execute_dirs:
                f.write(f"mkdir -p {d}\n")
            f.write("\n")
            
        # Move transferred inputs to their relative directories
        if input_rel_paths:
            f.write("# Move transferred input files to relative paths\n")
            for path in input_rel_paths:
                basename = os.path.basename(path)
                f.write(f"if [ -f {basename} ]; then\n")
                f.write(f"    mv {basename} {path}\n")
                f.write(f"fi\n")
            f.write("\n")
            
        # Execute the raw rule command
        f.write("# Run the actual rule command\n")
        f.write(rule_command)
        f.write("\n")
        f.write("echo '=================== POST-RUN DIR LISTING =================== '\n")
        f.write("ls -la\n")
        f.write("if [ -d output/v4_systematics_test/datacards/HHbb ]; then\n")
        f.write("    ls -la output/v4_systematics_test/datacards/HHbb\n")
        f.write("fi\n")
        f.write("echo '=========================================================== '\n")
        
    os.chmod(wrapper_path, 0o755)
    
    # Write JDL file
    memory = resources.get("mem_mb", 4000)
    transfer_inputs_str = ", ".join(transfer_inputs)
    transfer_outputs_str = ", ".join(output_transfers)
    
    jdl_content = f"""universe = vanilla
executable = {wrapper_path}
arguments = 
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
preserve_relative_paths = True
"""
    if transfer_inputs:
        jdl_content += f"transfer_input_files = {transfer_inputs_str}\n"
        
    if output_transfers:
        jdl_content += f"transfer_output_files = {transfer_outputs_str}\n"
        
    jdl_content += f"""output = {log_dir}/job_{rule}_{jobid}_$(Cluster).out
error = {log_dir}/job_{rule}_{jobid}_$(Cluster).err
log = {log_dir}/job_{rule}_{jobid}_$(Cluster).log
request_cpus = {threads}
request_memory = {memory}
requirements = TARGET.HasSingularity
+HasSingularity = True
+SingularityImage = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0"
queue
"""
    with open(jdl_path, "w") as f:
        f.write(jdl_content)
        
    # Submit job to HTCondor
    try:
        cmd = f"condor_submit {jdl_path}"
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_str = result.stdout.decode()
        
        # Parse cluster ID from output
        m = re.search(r"submitted to cluster (\d+)\.", stdout_str)
        if m:
            cluster_id = m.group(1)
            # Print the cluster ID to stdout
            print(cluster_id)
            print(f"Submitted rule {rule} job {jobid} to HTCondor, Cluster ID: {cluster_id}", file=sys.stderr)
        else:
            print(f"Error parsing cluster ID from condor_submit output: {stdout_str}", file=sys.stderr)
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"condor_submit failed with exit code {e.returncode}", file=sys.stderr)
        print(f"stdout: {e.stdout.decode()}", file=sys.stderr)
        print(f"stderr: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
