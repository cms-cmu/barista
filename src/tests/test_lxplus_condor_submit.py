"""Unit tests for software/snakemake/scripts/lxplus_condor_submit.py (Snakemake cluster-generic submit on lxplus)."""
import importlib.util
import os
import tempfile
import unittest
import unittest.mock

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "..", "software", "snakemake", "scripts", "lxplus_condor_submit.py")

spec = importlib.util.spec_from_file_location("lxplus_condor_submit", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestLxplusCondorSubmit(unittest.TestCase):

    def test_job_flavour(self):
        self.assertEqual(mod.job_flavour(None), "workday")
        self.assertEqual(mod.job_flavour(15), "espresso")
        self.assertEqual(mod.job_flavour(60), "microcentury")
        self.assertEqual(mod.job_flavour(480), "workday")
        self.assertEqual(mod.job_flavour(481), "tomorrow")
        self.assertEqual(mod.job_flavour("4320"), "testmatch")
        self.assertEqual(mod.job_flavour(10**6), "nextweek")
        self.assertEqual(mod.job_flavour("not-a-number"), "workday")

    def test_n_gpus(self):
        self.assertEqual(mod.n_gpus({}), 0)
        self.assertEqual(mod.n_gpus({"gres": "mps:50"}), 1)
        self.assertEqual(mod.n_gpus({"gres": "gpu:2"}), 2)
        self.assertEqual(mod.n_gpus({"gpus": 3}), 3)
        self.assertEqual(mod.n_gpus({"request_gpus": "1"}), 1)

    def test_should_submit(self):
        self.assertTrue(mod.should_submit("train", {}, {}))
        self.assertTrue(mod.should_submit("evaluate_all_svb", {}, {}))
        self.assertTrue(mod.should_submit("custom", {"gres": "mps:50"}, {}))
        self.assertFalse(mod.should_submit("plot_inputs_raw", {"mem_mb": 16000}, {}))
        self.assertTrue(mod.should_submit("plot_inputs_raw", {}, {"lxplus_condor_rules": "plot_inputs_raw,foo"}))
        self.assertTrue(mod.should_submit("anything", {}, {"lxplus_condor_all": "True"}))
        with unittest.mock.patch.dict(os.environ, {"LXPLUS_CONDOR_RULES": "analyze"}):
            self.assertTrue(mod.should_submit("analyze", {}, {}))

    def test_build_jdl_gpu_rule(self):
        jdl = mod.build_jdl("/work/.snakemake/tmp.x/snakejob.train.3.sh", "train", 3, 8,
                            {"runtime": 480, "mem_mb": 64000, "gres": "mps:50"}, gpus=1, workdir="/work")
        self.assertIn("executable = /work/.snakemake/tmp.x/snakejob.train.3.sh", jdl)
        self.assertIn("initialdir = /work", jdl)
        self.assertIn("MY.SendCredential = True", jdl)
        self.assertIn('+JobFlavour = "workday"', jdl)
        self.assertIn("request_cpus = 8", jdl)
        self.assertIn("request_memory = 64000MB", jdl)
        self.assertIn("request_gpus = 1", jdl)
        self.assertIn('requirements = !regexp("MIG", TARGET.GPUs_DeviceName)', jdl)
        self.assertIn("log = /work/condor_logs/job_train_3_$(Cluster).log", jdl)
        self.assertTrue(jdl.rstrip().endswith("queue"))

    def test_build_jdl_cpu_rule(self):
        jdl = mod.build_jdl("/work/js.sh", "analyze", 7, 4, {"runtime": 60, "mem_mb": 16000}, gpus=0, workdir="/work")
        self.assertNotIn("request_gpus", jdl)
        self.assertNotIn("requirements", jdl)
        self.assertIn('+JobFlavour = "microcentury"', jdl)

    def test_read_job_properties_fallback(self):
        props = {"rule": "train", "jobid": 5, "threads": 8, "resources": {"gres": "mps:50", "runtime": 480}, "config": {}}
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/sh\n# properties = " + __import__("json").dumps(props) + "\necho hi\n")
            path = f.name
        try:
            read = mod.read_job_properties(path)
        finally:
            os.remove(path)
        self.assertEqual(read["rule"], "train")
        self.assertEqual(read["resources"]["gres"], "mps:50")


if __name__ == "__main__":
    unittest.main()
