#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

display_section_header "THIS IS A FULL TEST OF THE SKIMMER"
display_section_header "THE RESULTS MIGHT NOT BE RELEVANT"
display_section_header "AND IT IS RUNNING MANY STEPS"

# Setup proxy if needed
setup_proxy 

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE_DIR/skimmer_basic_test"
create_output_directory "$OUTPUT_DIR"

display_section_header "Changing metadata"
SKIM_CONFIG="src/skimmer/tests/modify_branches_skimmer.yml"
sed -e "s|base_.*|base_path: $OUTPUT_DIR|" $SKIM_CONFIG > ${OUTPUT_DIR}/modify_branches_skimmer.yml
cat ${OUTPUT_DIR}/modify_branches_skimmer.yml; echo

display_section_header "Changing datasets"
nanoAOD_file="root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/GluGluToHHTo4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2810000/94DA5440-3B94-354B-A25B-78518A52D2D1.root"

echo """
datasets:
    GluGluToHHTo4B_cHHH1:
        UL18:
            nanoAOD:
                - "${nanoAOD_file}"
            picoAOD:
                count: 1000000
                files:
                    - root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v1p1/GluGluToHHTo4B_cHHH1_UL18/picoAOD.chunk0.root
                outliers:
                    - 334929
                saved_events: 199653
                sumw: 26757.179077148438
                sumw2: 919.0958442687988
                sumw2_diff: 1192990.2674102783
                sumw2_raw: 1193909.4220537543
                sumw_diff: 1453.1521577835083
                sumw_raw: 28210.3267947
                total_events: 1000000
        xs: 0.03077*0.5824**2
""" > ${OUTPUT_DIR}/datasets_HH4b.yml
cat ${OUTPUT_DIR}/datasets_HH4b.yml; echo

echo """
luminosities:
  UL18: 59.8e3
""" > ${OUTPUT_DIR}/luminosities_HH4b.yml
cat ${OUTPUT_DIR}/luminosities_HH4b.yml; echo

echo """
triggers:
  UL18: 
    - PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5
""" > ${OUTPUT_DIR}/triggers_HH4b.yml
cat ${OUTPUT_DIR}/triggers_HH4b.yml; echo

display_section_header "Skimming"
cmd=(python runner.py -s \
    -p src/skimmer/tests/modify_branches.py \
    -c ${OUTPUT_DIR}/modify_branches_skimmer.yml \
    -y UL18 -d GluGluToHHTo4B_cHHH1 \
    -op ${OUTPUT_DIR} \
    --luminosities ${OUTPUT_DIR}/luminosities_HH4b.yml \
    --triggers ${OUTPUT_DIR}/triggers_HH4b.yml \
    -o picoAOD_modify_branches.yml \
    -m ${OUTPUT_DIR}/datasets_HH4b.yml \
    -t --debug)
run_command "${cmd[@]}"
ls -R $OUTPUT_DIR

display_section_header "Merging skimmer outputs datasets"
cmd=(python src/utils/merge_yaml_datasets.py \
    -m ${OUTPUT_DIR}/datasets_HH4b.yml \
    -f ${OUTPUT_DIR}/picoAOD_modify_branches.yml \
    -o ${OUTPUT_DIR}/picoAOD_modify_branches.yml)
run_command "${cmd[@]}"

display_section_header "Running analysis on skimmer output"
cmd=(python runner.py \
    -p src/skimmer/tests/modify_branches.py \
    -c src/skimmer/tests/modify_branches_analysis.yml \
    -y UL18 -d GluGluToHHTo4B_cHHH1 \
    -op ${OUTPUT_DIR} \
    --luminosities ${OUTPUT_DIR}/luminosities_HH4b.yml \
    --triggers ${OUTPUT_DIR}/triggers_HH4b.yml \
    -o modify_branches.coffea \
    -m ${OUTPUT_DIR}/picoAOD_modify_branches.yml \
    -t --debug)
run_command "${cmd[@]}"
ls -R ${OUTPUT_DIR}