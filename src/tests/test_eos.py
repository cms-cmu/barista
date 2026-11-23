"""Unit tests for src.storage.eos path parsing and local operations."""

import os
import tempfile
from pathlib import PurePosixPath

from src.storage.eos import EOS, _is_missing


def test_local_path_properties():
    p = EOS("/tmp/foo//bar/file.root")
    assert p.is_local
    assert not p.is_null
    assert p.host == ""
    assert p.path == PurePosixPath("/tmp/foo/bar/file.root")
    assert str(p) == "/tmp/foo/bar/file.root"
    assert p.name == "file.root"
    assert p.stem == "file"
    assert p.suffix == ".root"


def test_remote_xrootd_path_parsing():
    p = EOS("root://cmseos.fnal.gov//store/user/algomez/test.root")
    assert not p.is_local
    assert not p.is_null
    assert p.host == "root://cmseos.fnal.gov/"
    assert p.path == PurePosixPath("/store/user/algomez/test.root")
    assert str(p) == "root://cmseos.fnal.gov//store/user/algomez/test.root"
    local = p.as_local
    assert local.is_local
    assert str(local) == "/store/user/algomez/test.root"


def test_null_and_devnull_paths():
    for arg in (None, os.devnull):
        p = EOS(arg)
        assert p.is_null
        assert p.is_local
        assert p.host == ""
        assert p.exists is True
        assert p.is_dir is False
        assert p.is_file is False


def test_path_joining_and_equality():
    base = EOS("root://eosuser.cern.ch//eos/user/a/algomez")
    child = base / "sub" / "out.coffea"
    assert str(child) == "root://eosuser.cern.ch//eos/user/a/algomez/sub/out.coffea"
    assert child == EOS("root://eosuser.cern.ch//eos/user/a/algomez/sub/out.coffea")
    assert child.parent == base / "sub"


def test_local_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = EOS(tmpdir)
        assert d.exists

        f = d / "sample.txt"
        assert not f.exists
        with open(str(f), "w") as fp:
            fp.write("hello")
        assert f.exists


def test_is_missing_helper():
    assert _is_missing(b"[3011] Unable to stat /eos/missing")
    assert _is_missing("ERROR: No such file or directory")
    assert not _is_missing(b"[3010] Unable to give access - permission denied")
