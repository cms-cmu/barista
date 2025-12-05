import subprocess

import yaml
import argparse

def get_das_summary(dataset, output_path, output_file):

    # Define the command
    command = [
        "dasgoclient",
        f"--query=summary dataset={dataset}",
        "-json"
    ]

    # Output file
    output_file = f"{output_path}/das_summary_{output_file}.json"

    # Execute the command and handle errors
    try:
        with open(output_file, 'w') as f:
            subprocess.run(command, stdout=f, check=True)
        #print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")



if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(description='Run coffea processor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--datasets', default="datasets_HH4b.yml", help='Main datasets file.')
    parser.add_argument('-o', '--output_path', default="coffea4bees/skimmer/metadata/", help='Main datasets file.')
    args = parser.parse_args()

    yaml_data = yaml.full_load(open(args.datasets, 'r'))
    dataset_dict = yaml_data["datasets"]



    for _process in dataset_dict:
        for _year in dataset_dict[_process]:
            dataset_year_dict = dataset_dict[_process][_year]["nanoAOD"]
            for _era, _dataset in dataset_year_dict.items():
                print("Doing...", _dataset)
                #print(f"{_process}_{_year}{ _era}")
                get_das_summary(_dataset, args.output_path, f"{_process}_{_year}{ _era}")
