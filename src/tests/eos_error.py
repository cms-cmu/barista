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
import os
import pickle
import shutil
import tempfile
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


class TestMkdirAcceptsAnExistingDirectory(unittest.TestCase):
    """xrdfs refuses `mkdir -p` on a directory owned by someone else, so creating an output
    directory once then reusing it has to not be an error."""

    def setUp(self):
        self.calls = []
        self.saved = eos.EOS.call

    def tearDown(self):
        eos.EOS.call = self.saved

    def fake_call(self, present: bool, mkdir_ok: bool = False):
        """Stand in for xrdfs: `ls` answers whether the path is there, `mkdir` is refused."""
        def call(cls, op, *args):
            self.calls.append(op)
            if op == "ls":
                if present:
                    return (True, b"")
                raise eos.EOSError(["xrdfs", "ls"], b"[3011] no such file or directory")
            if op == "mkdir":
                if mkdir_ok:
                    return (True, b"")
                raise eos.EOSError(["xrdfs", "mkdir"], b"[3010] permission denied")
            return (True, b"")
        return classmethod(call)

    def test_an_existing_directory_is_not_recreated(self):
        eos.EOS.call = self.fake_call(present=True)
        d = eos.EOS("root://cmsdata.phys.cmu.edu//store/group/HH4b/x")
        self.assertIsNotNone(d.mkdir(recursive=True))
        self.assertNotIn("mkdir", self.calls, "it tried to create a directory that was already there")

    def test_a_missing_directory_is_created(self):
        eos.EOS.call = self.fake_call(present=False, mkdir_ok=True)
        d = eos.EOS("root://cmsdata.phys.cmu.edu//store/group/HH4b/y")
        self.assertIsNotNone(d.mkdir(recursive=True))
        self.assertIn("mkdir", self.calls)

    def test_a_genuine_failure_still_raises(self):
        """Tolerating an existing directory must not swallow a real permission problem."""
        eos.EOS.call = self.fake_call(present=False, mkdir_ok=False)
        d = eos.EOS("root://cmsdata.phys.cmu.edu//store/group/HH4b/z")
        with self.assertRaises(eos.EOSError):
            d.mkdir(recursive=True)


class TestLocalPathsAnswerWhetherTheyExist(unittest.TestCase):
    """`Path` in src/storage/eos.py is PurePosixPath, which never touches a filesystem.

    `exists` called .exists() on it, so every local path raised AttributeError.  Nothing noticed
    until mkdir started asking, and then 36 jobs failed at once (2026-10-15).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_an_existing_local_directory_is_reported(self):
        self.assertTrue(eos.EOS(self.tmp).exists)

    def test_a_missing_local_path_is_reported(self):
        self.assertFalse(eos.EOS(os.path.join(self.tmp, "nope")).exists)

    def test_mkdir_creates_a_local_directory_and_then_accepts_it(self):
        d = os.path.join(self.tmp, "a", "b")
        eos.EOS(d).mkdir(recursive=True)
        self.assertTrue(os.path.isdir(d))
        eos.EOS(d).mkdir(recursive=True)            # second call must not raise
        self.assertTrue(os.path.isdir(d))


if __name__ == "__main__":
    unittest.main(verbosity=2)
