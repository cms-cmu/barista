"""Dataset-name parsing used by the cutflow tools (src/tools/cutflow_table.py)."""
import pytest

from src.tools.cutflow_table import parse_dataset_name


@pytest.mark.parametrize(
    "dataset, expected",
    [
        # Run 2
        ("data_UL16_preVFPC", ("data", "UL16_preVFP", "UL16_preVFPC")),
        ("data_UL17C", ("data", "UL17", "UL17C")),
        ("ttHbb_UL16_preVFP", ("ttHbb", "UL16_preVFP", "UL16_preVFP")),
        ("TTToHadronic_UL18", ("TTToHadronic", "UL18", "UL18")),
        ("TTbar_from_d3_UL17D", ("TTbar_from_d3", "UL17", "UL17D")),
        # Run 3, single underscore (cutflow dumps)
        ("data_2022_preEED", ("data", "2022_preEE", "2022_preEE_D")),
        ("data_2022_EEG", ("data", "2022_EE", "2022_EE_G")),
        ("data_2022_EE_G", ("data", "2022_EE", "2022_EE_G")),
        ("TTToSemiLeptonic_stitched_2022_preEE", ("TTToSemiLeptonic_stitched", "2022_preEE", "2022_preEE")),
        ("TTToSemiLeptonic_stitched_2022_EE", ("TTToSemiLeptonic_stitched", "2022_EE", "2022_EE")),
        ("TTbar_from_d3_2022_EEE", ("TTbar_from_d3", "2022_EE", "2022_EE_E")),
        ("data_2023_preBPixC1", ("data", "2023_preBPix", "2023_preBPix_C1")),
        ("data_2023_BPixD", ("data", "2023_BPix", "2023_BPix_D")),
        ("data_2024", ("data", "2024", "2024")),
        # Run 3, double underscore
        ("TTToHadronic__2022_preEE", ("TTToHadronic", "2022_preEE", "2022_preEE")),
        ("data__2022_preEE_B", ("data", "2022_preEE", "2022_preEE_B")),
        ("data__2022_EE_G", ("data", "2022_EE", "2022_EE_G")),
    ],
)
def test_parse_dataset_name(dataset, expected):
    assert parse_dataset_name(dataset) == expected


def test_merge_ttbar():
    assert parse_dataset_name("TTToSemiLeptonic_stitched_2022_EE", merge_ttbar=True)[0] == "TTbar"
    assert parse_dataset_name("TTTo2L2Nu_UL18", merge_ttbar=True)[0] == "TTbar"
    assert parse_dataset_name("data_2022_EEG", merge_ttbar=True)[0] == "data"
