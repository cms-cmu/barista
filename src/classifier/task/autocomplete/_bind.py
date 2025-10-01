import os
import sys
import time
from multiprocessing.connection import Client

ADDRESS = "/tmp/pyml-autocomplete-server.sock"
_EXIT = "exit"
_WAIT = "wait"

_WAIT_INTERVAL = 0.1  # seconds
_WAIT_TIMEOUT = 10  # seconds

# Exit codes
NOSERVER = 254
FILEDIR = 255

_CONN_ERR = (FileNotFoundError, ConnectionRefusedError)

def is_exit(args: list[str]):
    return len(args) == 1 and args[0] == _EXIT


def is_wait(args: list[str]):
    return len(args) >= 1 and args[0] == _WAIT


def pipe_client():
    args = sys.argv[1:]
    if is_wait(args):
        start = time.time()
        while (time.time() - start) < _WAIT_TIMEOUT:
            try:
                client = Client(ADDRESS)
                break
            except _CONN_ERR:
                time.sleep(_WAIT_INTERVAL)
    else:
        try:
            client = Client(ADDRESS)
        except _CONN_ERR:
            sys.exit(NOSERVER)
    client.send(args)
    if is_exit(args):
        try:
            os.remove(ADDRESS)
        except FileNotFoundError:
            ...
    try:
        message = client.recv()
        if isinstance(message, list):
            print("\n".join(message))
        elif isinstance(message, int):
            sys.exit(message)
    except EOFError:
        return


if __name__ == "__main__":
    pipe_client()
