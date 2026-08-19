import os
import pytest
from unittest.mock import patch
from src.runner.dataset import expand_directory_files, list_of_files


def test_expand_local_directory(tmp_path):
    # Create test directory structure with subdirectories 0000, 0001
    d0 = tmp_path / "0000"
    d1 = tmp_path / "0001"
    d0.mkdir()
    d1.mkdir()

    (d0 / "file_2.root").touch()
    (d0 / "file_10.root").touch()
    (d0 / "file_1.root").touch()
    (d1 / "file_20.root").touch()
    (d1 / "ignore.txt").touch()

    # Test discovery from directory path
    res = expand_directory_files(str(tmp_path))
    assert len(res) == 4
    basenames = [os.path.basename(f) for f in res]
    # Check natural sorting
    assert basenames == ["file_1.root", "file_2.root", "file_10.root", "file_20.root"]


def test_expand_xrootd_directory():
    mock_output = (
        "/store/user/test/0000/ntuple_merged_2.root\n"
        "/store/user/test/0000/ntuple_merged_10.root\n"
        "/store/user/test/0000/ntuple_merged_1.root\n"
        "/store/user/test/0001/ntuple_merged_20.root\n"
        "/store/user/test/log.tar.gz\n"
    )
    with patch("subprocess.check_output", return_value=mock_output):
        res = expand_directory_files("root://cmsxrootd.fnal.gov//store/user/test/")
        assert len(res) == 4
        assert res[0] == "root://cmsxrootd.fnal.gov//store/user/test/0000/ntuple_merged_1.root"
        assert res[1] == "root://cmsxrootd.fnal.gov//store/user/test/0000/ntuple_merged_2.root"
        assert res[2] == "root://cmsxrootd.fnal.gov//store/user/test/0000/ntuple_merged_10.root"
        assert res[3] == "root://cmsxrootd.fnal.gov//store/user/test/0001/ntuple_merged_20.root"


def test_list_of_files_integration():
    mock_output = (
        "/store/user/test/ntuple_merged_1.root\n"
        "/store/user/test/ntuple_merged_2.root\n"
        "/store/user/test/ntuple_merged_3.root\n"
    )
    with patch("subprocess.check_output", return_value=mock_output):
        # Full directory string
        files = list_of_files("root://cmsxrootd.fnal.gov//store/user/test/", test=False)
        assert len(files) == 3

        # Test limit
        test_files = list_of_files("root://cmsxrootd.fnal.gov//store/user/test/", test=True, test_files=2)
        assert len(test_files) == 2


def test_yaml_dataset_definitions():
    import yaml
    for fname, datasets in [
        ("coffea4bees/metadata/datasets/ZZ4b.yml", ["ZZ4b"]),
        ("coffea4bees/metadata/datasets/ZH4b.yml", ["ZH4b", "ggZH4b"])
    ]:
        with open(fname) as f:
            db = yaml.safe_load(f)

        for ds in datasets:
            assert ds in db
            for era in ["2022_preEE", "2022_postEE", "2023_preBPIX", "2023_postBPIX"]:
                assert era in db[ds]
                assert "nanoAOD" in db[ds][era]
                assert db[ds][era]["nanoAOD"].startswith("root://cmsxrootd.fnal.gov//store/user/ekoenig/Run3_private_production/")
