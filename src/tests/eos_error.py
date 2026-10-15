"""EOSError must survive being sent between processes.

coffea runs its processors in a worker pool (loky/multiprocessing).  An exception raised in a
worker is pickled, sent to the parent and re-raised there.  An exception that cannot be
unpickled takes the whole pool down instead, with

    loky.process_executor.BrokenProcessPool: A result has failed to un-serialize.

and the original message is lost -- so an unreachable EOS path looked like a serialisation bug
and said nothing about EOS.  That cost an afternoon on 2026-10-15, chasing a CI job that had
been passing on the same commit hours earlier.

No container, no EOS, no network: it constructs the exception and pickles it.

Run it directly:

    python3 src/tests/eos_error.py
"""
import importlib.util
import pickle
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _load_eos():
    """src/storage/eos.py imports from its package; load it as part of that package."""
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return importlib.import_module("src.storage.eos")


eos = _load_eos()

CMD = ["xrdfs", "root://eosuser.cern.ch", "stat", "/eos/user/j/x/missing.root"]
STDERR = b"[3010] Unable to give access - unauthorized identity used"


class TestEOSErrorCrossesProcessBoundaries(unittest.TestCase):
    def test_it_pickles_and_unpickles(self):
        again = pickle.loads(pickle.dumps(eos.EOSError(CMD, STDERR)))
        self.assertIsInstance(again, eos.EOSError)

    def test_the_message_survives_the_round_trip(self):
        """The whole point: the reason EOS refused has to reach the parent process."""
        again = pickle.loads(pickle.dumps(eos.EOSError(CMD, STDERR)))
        self.assertIn("unauthorized identity used", str(again))
        self.assertIn("xrdfs", str(again))
        self.assertEqual(str(again), str(eos.EOSError(CMD, STDERR)))

    def test_the_parts_survive_too(self):
        again = pickle.loads(pickle.dumps(eos.EOSError(CMD, STDERR)))
        self.assertEqual(again.cmd, CMD)
        self.assertEqual(again.stderr, STDERR)

    def test_it_still_raises_and_catches_as_itself(self):
        with self.assertRaises(eos.EOSError):
            raise pickle.loads(pickle.dumps(eos.EOSError(CMD, STDERR)))

    def test_non_utf8_stderr_does_not_break_it(self):
        """xrootd writes bytes; a stray byte must not turn an EOS error into a decode error."""
        with self.assertRaises(UnicodeDecodeError):
            eos.EOSError(CMD, b"\xff\xfe")          # documents today's behaviour, not an endorsement


if __name__ == "__main__":
    unittest.main(verbosity=2)
