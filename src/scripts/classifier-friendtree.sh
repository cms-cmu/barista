#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Create output directory
JOB="classifier_friendtree"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Input Datasets"
DATASETS=${DATASET:-"coffea4bees/metadata/datasets_HH4b.yml"}
echo "Using datasets file: $DATASETS"

display_section_header "Modifying config"
JOB_CONFIG=$OUTPUT_DIR/classifier_friendtree.yml
sed -e "s|make_.*|make_classifier_input: $OUTPUT_DIR|" \
    coffea4bees/analysis/metadata/HH4b_classifier_inputs.yml > $JOB_CONFIG
cat $JOB_CONFIG; echo

echo "############### Running test processor"
python runner.py -t -o classifier_friendtree.yml -d data GluGluToHHTo4B_cHHH1 -p coffea4bees/analysis/processors/processor_HH4b.py -y UL18 -op $OUTPUT_DIR -c $JOB_CONFIG -m $DATASETS
ls $OUTPUT_DIR

