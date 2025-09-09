import argparse

from ..storage.eos import EOS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        type=EOS,
        action="extend",
        nargs="+",
        default=[],
        help="remote files to be fixed",
    )
    args = parser.parse_args()

    files: list[EOS] = args.file
    for file in files:
        tmp = file.local_temp(".")
        file.move_to(tmp, overwrite=True)
        tmp.move_to(file)
