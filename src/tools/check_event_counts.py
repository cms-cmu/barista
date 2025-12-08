import yaml
import argparse
import json
from collections import defaultdict


def get_events_in_yaml(file_name):

    with open(file_name, 'r') as file:
        yml_data = yaml.full_load(file)

    total_events_in_yaml = defaultdict(list)

    for _period in yml_data:

        total_events_in_yaml[_period] = yml_data[_period]["total_events"]

    return total_events_in_yaml



def get_events_in_das(dataset_names):

    counts_in_das = defaultdict(list)

    for _dataset in dataset_names:
        # Open and read a JSON file
        with open(f"coffea4bees/skimmer/metadata/das_summary_{_dataset}.json", 'r') as file:
            json_data = json.load(file)

        counts_in_das[_dataset] = json_data[0]["summary"][0]["nevents"]

    return counts_in_das

def run(yaml_file_name):
    print()
    counts_in_yaml = get_events_in_yaml(yaml_file_name)
    counts_in_das  = get_events_in_das(counts_in_yaml.keys())

    for _dataset in counts_in_yaml.keys():
        count_diff = counts_in_yaml[_dataset] - counts_in_das[_dataset]
        if count_diff:
            print(f"\t{_dataset} count difference = {count_diff} out of {counts_in_das}")
        else:
            print(f"\t{_dataset} ... all good! ")
    print()

if __name__ == "__main__":

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(description='Run coffea processor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-y', '--yaml_file', default="coffea4bees/skimmer/metadata/picoaod_datasets_data_2023_BPix.yml", help='skimmer output yaml.')
    args = parser.parse_args()

    # ASssuem that
    # py coffea4bees/skimmer/metadata/get_das_info.py -d python/python/metadata/datasets_HH4b_Run3.yml

    run(yaml_file_name = args.yaml_file)
