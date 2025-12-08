import yaml
import argparse
import json
from collections import defaultdict



def convert_range_to_list(input_range):
    return list(range(input_range[0], input_range[1] + 1))



def get_LBs_in_json(file_name):

    # Open and read a JSON file
    with open(file_name, 'r') as file:
        json_data = json.load(file)


    run_list = list(json_data.keys())
    #run_list = run_list[0:2]

    expanded_LBs_in_json = defaultdict(list)

    for _run in run_list:
        #print(f"Run: {_run} --> LB ranges {json_data[_run]}")
        for _lb_range in json_data[_run]:
            #print(_lb_range)
            expanded_LBs_in_json[_run] += convert_range_to_list(_lb_range)

            #print(f" expanded_LBs_in_json is now {expanded_LBs_in_json[_run]}")

    return expanded_LBs_in_json


def get_LBs_in_yaml(file_name):

    with open(file_name, 'r') as file:
        yml_data = yaml.full_load(file)

    LBs_in_yaml = defaultdict(list)

    for _period in yml_data:
        _lumis_processed = yml_data[_period]["lumis_processed"]
        for _run, _LBs in _lumis_processed.items():
            LBs_in_yaml[str(_run)] = _LBs

    return LBs_in_yaml

def run(json_file_name, yaml_file_name):

    LBs_in_json = get_LBs_in_json(json_file_name)
    LBs_in_yaml = get_LBs_in_yaml(yaml_file_name)



    #
    # Check all runs seen
    #
    if set(LBs_in_json.keys()) == set(LBs_in_yaml.keys()):
        print("All runs seen.")
    else:
        # OK to have runs missing in the JSON
        # missing_in_json = set(LBs_in_yaml.keys()) - set(LBs_in_json.keys())
        # print(f"Runs Missing in JSON {missing_in_json}.")

        missing_in_yaml = set(LBs_in_json.keys()) - set(LBs_in_yaml.keys())
        print(f"Runs Missing in YAML  {missing_in_yaml}.")

    runs_in_both = list(set(LBs_in_json.keys()) & set(LBs_in_yaml.keys()))
    #runs_in_both = runs_in_both[0:2]

    total_json_lbs = 0
    missing_lbs = 0

    for _r in runs_in_both:

        total_json_lbs += len(LBs_in_json[_r])

        missing_in_yaml = set(LBs_in_json[_r]) - set(LBs_in_yaml[_r])
        missing_lbs += len(missing_in_yaml)

        if len(missing_in_yaml):
            print(_r)
            print(f"\t Missing LBS {missing_in_yaml}")


    print(f" \n Have {missing_lbs} missing LBs out of {total_json_lbs} total {round(missing_lbs/total_json_lbs,5)}")

if __name__ == "__main__":

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(description='Run coffea processor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-j', '--json_file', default="base_class/data/goldenJSON/Cert_Collisions2023_366442_370790_Golden.json", help='golden json.')
    parser.add_argument('-y', '--yaml_file', default="coffea4bees/skimmer/metadata/picoaod_datasets_data_2023_BPix.yml", help='skimmer output yaml.')
    #parser.add_argument('-o', '--output_file', dest="output_file", default="datasets_HH4b_merged.yml", help='Output file.')
    #parser.add_argument('-f', '--files_to_add', nargs='+', dest="files_to_add", default=["datasets_HH4b.yml"], help='Files to add.')
    args = parser.parse_args()


    run(json_file_name = args.json_file,
        yaml_file_name = args.yaml_file,
        )
