import unittest
from unittest.mock import patch, MagicMock
import argparse
import sys
import os

from runner import (
    make_parser,
    setup_config_defaults,
    get_dataset_type,
    find_matching_dataset,
    calculate_cross_section,
    setup_schema,
    setup_pico_base_name,
    apply_storage_remap,
    find_free_port,
    setup_local_cluster,
    setup_shared_dask_client
)


class TestRunner(unittest.TestCase):

    def test_make_parser_defaults(self):
        """Test that the parser configures correct defaults."""
        parser = make_parser()
        # Parse an empty argument list to check defaults
        args = parser.parse_args([])
        self.assertFalse(args.shared_dask)
        self.assertFalse(args.condor)
        self.assertFalse(args.slurm)
        self.assertFalse(args.debug)
        self.assertEqual(args.processor, "coffea4bees/analysis/processors/processor_HH4b.py")
        self.assertEqual(args.configs, "coffea4bees/analysis/metadata/HH4b.yml")
        self.assertEqual(args.idle_timeout, 600)

    def test_make_parser_flags(self):
        """Test that command line flags are parsed correctly."""
        parser = make_parser()
        args = parser.parse_args([
            "--shared-dask",
            "--condor",
            "--debug",
            "-y", "UL18", "UL17",
            "-d", "GluGluToHHTo4B_cHHH1"
        ])
        self.assertTrue(args.shared_dask)
        self.assertTrue(args.condor)
        self.assertTrue(args.debug)
        self.assertEqual(args.years, ["UL18", "UL17"])
        self.assertEqual(args.datasets, ["GluGluToHHTo4B_cHHH1"])

    def test_make_parser_invalid_args(self):
        """Test that invalid argument combinations cause SystemExit."""
        parser = make_parser()
        # Suppress stderr to keep test output clean
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stderr', devnull):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--invalid-flag-xyz"])

    def test_setup_config_defaults_standalone(self):
        """Test that setup_config_defaults sets correct workers for standalone mode."""
        args = MagicMock()
        args.shared_dask = False
        args.worker_memory = None
        args.slurm_qos = None
        args.test = False
        
        config_runner = {}
        setup_config_defaults(config_runner, args)
        self.assertEqual(config_runner["max_workers"], 400)
        self.assertEqual(config_runner["workers"], 2)

    def test_setup_config_defaults_shared(self):
        """Test that setup_config_defaults sets correct workers for shared mode."""
        args = MagicMock()
        args.shared_dask = True
        args.worker_memory = None
        args.slurm_qos = None
        args.test = False
        
        config_runner = {}
        setup_config_defaults(config_runner, args)
        self.assertEqual(config_runner["max_workers"], 1000)
        self.assertEqual(config_runner["workers"], 2)

    def test_setup_config_defaults_custom(self):
        """Test that setup_config_defaults respects user-specified worker limits."""
        args = MagicMock()
        args.shared_dask = True
        args.worker_memory = "8GB"
        args.slurm_qos = "cpu_medium"
        args.test = False
        
        config_runner = {"max_workers": 50, "worker_memory": "4GB", "slurm_qos": "cpu_light"}
        setup_config_defaults(config_runner, args)
        self.assertEqual(config_runner["max_workers"], 50)  # should respect config value
        self.assertEqual(config_runner["worker_memory"], "4GB")
        self.assertEqual(config_runner["slurm_qos"], "cpu_light")

    def test_get_dataset_type(self):
        """Test dataset type classification."""
        self.assertEqual(get_dataset_type("data"), "data")
        self.assertEqual(get_dataset_type("data__Run2018A"), "data")
        self.assertEqual(get_dataset_type("GluGluToHHTo4B_cHHH1"), "mc")
        self.assertEqual(get_dataset_type("TTToSemiLeptonic"), "mc")
        self.assertEqual(get_dataset_type("mixeddata"), "mixed_data")
        self.assertEqual(get_dataset_type("datamixed"), "data_mixed")
        self.assertEqual(get_dataset_type("data_3b_for_mixed"), "data_for_mix")
        self.assertEqual(get_dataset_type("TTToSemiLeptonic_for_mixed"), "tt_for_mixed")

    def test_apply_storage_remap(self):
        """Test apply_storage_remap utility."""
        remaps = [
            {"from": "root://cms-xrd-global.cern.ch/", "to": "root://cmsxrootd.fnal.gov/"}
        ]
        obj = {
            "files": [
                "root://cms-xrd-global.cern.ch//store/mc/RunII/file1.root",
                "root://other-site.ch//store/mc/file2.root"
            ]
        }
        res = apply_storage_remap(obj, remaps)
        self.assertEqual(res["files"][0], "root://cmsxrootd.fnal.gov//store/mc/RunII/file1.root")
        self.assertEqual(res["files"][1], "root://other-site.ch//store/mc/file2.root")

    def test_find_matching_dataset(self):
        """Test finding matching dataset keys in metadata."""
        metadata = {
            "datasets": {
                "GluGluToHHTo4B_cHHH1": {"xs": 0.01},
                "TTToSemiLeptonic": {"xs": 365.3}
            }
        }
        matched = find_matching_dataset("GluGluToHHTo4B_cHHH1", metadata)
        self.assertEqual(matched, "GluGluToHHTo4B_cHHH1")
        
        # Test substring matching
        matched_substring = find_matching_dataset("TTToSemi", metadata)
        self.assertEqual(matched_substring, "TTToSemiLeptonic")

    def test_calculate_cross_section(self):
        """Test cross section calculation formula."""
        metadata = {
            "datasets": {
                "GluGluToHHTo4B_cHHH1": {"xs": 0.01}
            }
        }
        xs = calculate_cross_section("GluGluToHHTo4B_cHHH1", "mc", metadata)
        self.assertEqual(xs, 0.01)
        
        # Data cross-section is always 1.0
        xs_data = calculate_cross_section("data", "data", metadata)
        self.assertEqual(xs_data, 1.0)

    def test_setup_schema(self):
        """Test mapping string schema to actual Coffea classes."""
        from coffea.nanoevents.schemas import NanoAODSchema, PFNanoAODSchema
        
        config = {"schema": "NanoAODSchema"}
        setup_schema(config)
        self.assertEqual(config["schema"], NanoAODSchema)
        
        config_base = {"schema": "PFNanoAODSchema"}
        setup_schema(config_base)
        self.assertEqual(config_base["schema"], PFNanoAODSchema)

    def test_setup_pico_base_name(self):
        """Test setup_pico_base_name outputs correct string format."""
        configs = {
            "runner": {
                "class_name": "SubSampler"
            }
        }
        base_name = setup_pico_base_name(configs)
        self.assertEqual(base_name, "picoAOD_PSData")

    def test_find_free_port(self):
        """Test find_free_port returns a valid integer port."""
        port = find_free_port(10200)
        self.assertIsInstance(port, int)
        self.assertTrue(1024 <= port <= 65535)

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_setup_local_cluster(self, mock_client, mock_cluster):
        """Test Dask setup_local_cluster orchestration using mocks."""
        config_runner = {
            "dashboard_address": 12345,
            "workers": 4,
            "worker_memory": "4GB"
        }
        client, cluster = setup_local_cluster(config_runner)
        mock_cluster.assert_called_once_with(
            n_workers=4,
            memory_limit="4GB",
            threads_per_worker=1,
            dashboard_address=":12345",
            scheduler_port=8786
        )
        mock_client.assert_called_once_with(mock_cluster.return_value)

    @patch("os.path.exists")
    @patch("builtins.open")
    @patch("distributed.Client")
    def test_setup_shared_dask_client(self, mock_client, mock_open, mock_exists):
        """Test setup_shared_dask_client correctly loads JSON and connects."""
        mock_exists.return_value = True
        
        # Mock reading the scheduler JSON file containing the address
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = '{"address": "tcp://127.0.0.1:8786"}'
        mock_open.return_value = mock_file
        
        args = MagicMock()
        args.start_cluster_daemon = False
        args.scheduler_address = None
        
        config_runner = {}
        client, cluster = setup_shared_dask_client(args, config_runner)
        
        mock_client.assert_called_once_with("tcp://127.0.0.1:8786", timeout="5s")
        self.assertIsNone(cluster)


if __name__ == '__main__':
    unittest.main()
