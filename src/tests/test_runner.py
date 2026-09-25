import unittest
from unittest.mock import patch, MagicMock
import argparse
import sys
import os

import tempfile
import tarfile

from runner import (
    WorkerInitializer,
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
from src.runner import cluster as cluster_mod
from src.runner.cluster import (
    detect_condor_site,
    eos_xrootd_url,
    setup_condor_cluster,
    setup_lxplus_condor_cluster,
    create_code_tarball,
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
            "-d", "GluGluToHHTo4B_cHHH1",
            "--not-do-proxy",
            "--run-performance"
        ])
        self.assertTrue(args.shared_dask)
        self.assertTrue(args.condor)
        self.assertTrue(args.debug)
        self.assertEqual(args.years, ["UL18", "UL17"])
        self.assertEqual(args.datasets, ["GluGluToHHTo4B_cHHH1"])
        self.assertTrue(args.not_do_proxy)
        self.assertTrue(args.run_performance)

    def test_make_parser_invalid_args(self):
        """Test that invalid argument combinations cause SystemExit."""
        parser = make_parser()
        # Suppress stderr to keep test output clean
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stderr', devnull):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--invalid-flag-xyz"])

    @patch("sys.argv", ["runner.py", "-c", "coffea4bees/analysis/metadata/HH4b.yml"])
    def test_parse_args_with_c_option(self):
        """Test that the parser does not treat a file passed to -c as a positional YAML config."""
        from src.runner.cli import parse_args
        args = parse_args()
        self.assertEqual(args.configs, "coffea4bees/analysis/metadata/HH4b.yml")
        self.assertIsNone(args.job_yaml_path)

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
        
        mock_client.assert_called_once_with("tcp://127.0.0.1:8786", timeout="30s")
        self.assertIsNone(cluster)


class TestCondorSites(unittest.TestCase):
    """HTCondor site selection and the lxplus (dask_lxplus) backend."""

    LXPLUS_CONFIG = {
        'condor_cores': 2, 'worker_memory': '4GB', 'dashboard_address': 10200,
        'min_workers': 1, 'max_workers': 5,
        'lxplus_job_flavour': 'workday', 'lxplus_disk_per_worker': '10GB', 'lxplus_death_timeout': 3600,
        'lxplus_scheduler_port': 8786, 'lxplus_batch_name': 'barista-dask', 'lxplus_worker_image': None,
        'lxplus_eos_scratch': None, 'lxplus_send_credential': True,
    }

    def test_detect_condor_site_precedence(self):
        with patch.dict(os.environ, {"BARISTA_SITE": "lxplus"}):
            self.assertEqual(detect_condor_site({"condor_site": "lpc"}), "lpc")
            self.assertEqual(detect_condor_site({}), "lxplus")
        with patch.dict(os.environ, {"BARISTA_SITE": "lpc_gpu"}):
            self.assertEqual(detect_condor_site({}), "lpc")
        with patch.dict(os.environ, {"BARISTA_SITE": ""}):
            with patch("socket.gethostname", return_value="lxplus954.cern.ch"):
                self.assertEqual(detect_condor_site({}), "lxplus")
            with patch("socket.gethostname", return_value="cmslpc307.fnal.gov"):
                self.assertEqual(detect_condor_site({}), "lpc")
            with patch("socket.gethostname", return_value="falcon.phys.cmu.edu"):
                self.assertEqual(detect_condor_site(None), "lpc")

    def test_setup_condor_cluster_dispatch(self):
        with patch.object(cluster_mod, "setup_lxplus_condor_cluster") as lx, \
             patch.object(cluster_mod, "setup_lpc_condor_cluster") as lpc:
            setup_condor_cluster({}, "/tmp/code.tar.gz", proxy_path="/tmp/proxy", site="lxplus")
            lx.assert_called_once_with({}, "/tmp/code.tar.gz", "/tmp/proxy")
            setup_condor_cluster({}, "/tmp/code.tar.gz", site="lpc")
            lpc.assert_called_once_with({}, "/tmp/code.tar.gz")
        with self.assertRaises(ValueError):
            setup_condor_cluster({}, "/tmp/code.tar.gz", site="bogus")

    def test_eos_xrootd_url(self):
        self.assertEqual(eos_xrootd_url("/eos/cms/store/group/x"), "root://eoscms.cern.ch//eos/cms/store/group/x")
        self.assertEqual(eos_xrootd_url("/eos/user/m/me/x"), "root://eosuser.cern.ch//eos/user/m/me/x")
        self.assertIsNone(eos_xrootd_url("/tmp/x"))

    def _run_lxplus_setup(self, config, tarball, proxy, env):
        fake_cls = MagicMock(name="CernCluster")
        fake_mod = MagicMock(CernCluster=fake_cls)
        with patch.dict(sys.modules, {"dask_lxplus": fake_mod}), \
             patch("dask.distributed.Client") as mock_client, \
             patch("src.runner.cluster._port_is_free", return_value=True), \
             patch.dict(os.environ, env):
            client, cluster, log_dir = setup_lxplus_condor_cluster(config, tarball, proxy)
        return fake_cls, mock_client, cluster, log_dir

    def test_setup_lxplus_condor_cluster(self):
        with tempfile.TemporaryDirectory() as tmp:
            tarball = os.path.join(tmp, "code_barista.tar.gz")
            proxy = os.path.join(tmp, "x509_proxy")
            for f in (tarball, proxy):
                open(f, "w").close()
            config = dict(self.LXPLUS_CONFIG, worker_log_directory=os.path.join(tmp, "logs"))
            fake_cls, mock_client, cluster, log_dir = self._run_lxplus_setup(
                config, tarball, proxy, {"WORKER_IMAGE": "/cvmfs/unpacked.cern.ch/barista:test"})

            kwargs = fake_cls.call_args.kwargs
            self.assertEqual(kwargs["container_runtime"], "singularity")
            self.assertEqual(kwargs["worker_image"], "/cvmfs/unpacked.cern.ch/barista:test")
            self.assertEqual(kwargs["cores"], 2)
            self.assertEqual(kwargs["processes"], 1)
            self.assertEqual(kwargs["disk"], "10GB")
            self.assertEqual(kwargs["scheduler_options"]["port"], 8786)  # the configured port, never a fallback
            self.assertEqual(kwargs["log_directory"], os.path.join(tmp, "logs"))
            self.assertIn("export X509_USER_PROXY=${X509_USER_PROXY:-$PWD/x509_proxy}", kwargs["job_script_prologue"])
            directives = kwargs["job_extra_directives"]
            self.assertEqual(directives["+JobFlavour"], '"workday"')
            self.assertEqual(directives["MY.SendCredential"], "True")
            self.assertEqual(directives["x509userproxy"], proxy)
            self.assertEqual(directives["transfer_input_files"], tarball)
            self.assertEqual(directives["leave_in_queue"], "False")
            self.assertEqual(directives["transfer_executable"], "False")
            self.assertNotIn("output_destination", directives)  # log dir is not on EOS
            fake_cls.return_value.adapt.assert_called_once_with(minimum=1, maximum=5)
            mock_client.assert_called_once_with(fake_cls.return_value)
            self.assertEqual(log_dir, os.path.join(tmp, "logs"))
            self.assertEqual(cluster.barista_cleanup_paths, [])

    def test_setup_lxplus_condor_cluster_eos_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tarball = os.path.join(tmp, "code_barista.tar.gz")
            open(tarball, "w").close()
            config = dict(self.LXPLUS_CONFIG, worker_log_directory="/eos/cms/store/group/test/logs",
                          lxplus_send_credential=False)
            with patch("os.makedirs"):
                fake_cls, _, _, log_dir = self._run_lxplus_setup(config, tarball, None, {"WORKER_IMAGE": "/cvmfs/x"})
            directives = fake_cls.call_args.kwargs["job_extra_directives"]
            self.assertEqual(directives["output_destination"], "root://eoscms.cern.ch//eos/cms/store/group/test/logs/")
            self.assertEqual(directives["Output"], "worker-$(ClusterId).$(ProcId).out")
            self.assertEqual(directives["MY.XRDCP_CREATE_DIR"], "True")
            self.assertNotIn("MY.SendCredential", directives)
            self.assertNotIn("x509userproxy", directives)
            self.assertEqual(log_dir, "/eos/cms/store/group/test/logs")

    def test_wait_for_lxplus_port_errors_when_busy(self):
        with patch("src.runner.cluster._port_is_free", return_value=False), patch("time.sleep"):
            with self.assertRaises(RuntimeError):
                cluster_mod._wait_for_lxplus_port(8786, timeout=0)

    def test_setup_lxplus_condor_cluster_requires_dask_lxplus(self):
        with patch.dict(sys.modules, {"dask_lxplus": None}):
            with self.assertRaises(ImportError):
                setup_lxplus_condor_cluster(dict(self.LXPLUS_CONFIG), "/tmp/none.tar.gz")

    def test_create_code_tarball_excludes_vcs_and_bytecode(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                os.makedirs("pkg/.git/objects")
                os.makedirs("pkg/__pycache__")
                open("pkg/.git/objects/blob", "w").close()
                open("pkg/__pycache__/mod.cpython-312.pyc", "w").close()
                open("pkg/mod.py", "w").close()
                tarball, temp_dir = create_code_tarball(["pkg"], tmpdir=os.path.join(tmp, "scratch"))
                with tarfile.open(tarball) as tar:
                    names = tar.getnames()
            finally:
                os.chdir(cwd)
            self.assertIn("pkg/mod.py", names)
            self.assertFalse(any(".git" in n or "__pycache__" in n for n in names), names)

    def test_setup_config_defaults_lxplus_keys(self):
        args = MagicMock(shared_dask=False, worker_memory=None, slurm_qos=None, test=False)
        config_runner = {}
        setup_config_defaults(config_runner, args)
        self.assertIsNone(config_runner["condor_site"])
        self.assertEqual(config_runner["lxplus_job_flavour"], "workday")
        self.assertEqual(config_runner["lxplus_disk_per_worker"], "10GB")
        self.assertIn("{user}", config_runner["lxplus_eos_scratch"])
        self.assertTrue(config_runner["lxplus_send_credential"])

    def test_make_parser_condor_site(self):
        parser = make_parser()
        self.assertIsNone(parser.parse_args([]).condor_site)
        self.assertEqual(parser.parse_args(["--condor", "--condor-site", "lxplus"]).condor_site, "lxplus")
        with open(os.devnull, 'w') as devnull, patch('sys.stderr', devnull):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--condor-site", "falcon"])

    def test_worker_initializer_absolutizes_relative_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                open("x509_proxy", "w").close()
                with patch.dict(os.environ, {"X509_USER_PROXY": "x509_proxy"}):
                    WorkerInitializer().setup(worker=None)
                    self.assertEqual(os.environ["X509_USER_PROXY"], os.path.join(os.path.realpath(tmp), "x509_proxy"))
            finally:
                os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
