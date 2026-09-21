"""Unit tests for src/tools/roast.py, the production-run orchestrator.

Runs anywhere python3 and bash are available: no ssh, no cluster, no network, no
container.  Almost everything roast does on a remote host is bash it generates
from f-strings with embedded awk, so these tests cover the two failure modes that
are invisible until a production run is already going wrong:

  * the generated bash is malformed (quoting/escaping), and
  * the snakemake command line loses a flag that makes a run reproducible
    (--configfile, --config roast_id) or recoverable (--rerun-incomplete).

Run it directly:

    python3 src/tests/roast.py
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
BASH = shutil.which("bash")


def _load_roast():
    """roast.py is a CLI script, not an importable package module."""
    path = REPO / "src" / "tools" / "roast.py"
    spec = importlib.util.spec_from_file_location("roast_tool", path)
    mod = importlib.util.module_from_spec(spec)
    os.chdir(REPO)          # roast resolves the barista root from cwd at import time
    spec.loader.exec_module(mod)
    return mod


roast = _load_roast()

FAKE_ID = "citest_20260101_aaaaaaa-bbbbbbb"
CFG = {
    "hosts": {
        "cmslpc": {"ssh": "u@cmslpc307.fnal.gov", "prod_root": "~/nobackup/HH4b/prod",
                   "reference": "~/nobackup/HH4b/Run3/barista", "cores": 8},
        "falcon": {"ssh": "u@falcon.phys.cmu.edu", "prod_root": "~/work/prod",
                   "reference": "~/work/barista", "cores": 4},
    },
    "cernbox": {"eos_path": "/eos/user/u/user/www/HH4b/prod", "url": "https://user.web.cern.ch/user/HH4b/prod"},
    "eos": {"url": "root://cmseos.fnal.gov", "path": "/store/user/u/HH4b_prod"},
}
GITLAB = "ssh://git@gitlab.cern.ch:7999/cms-cmu/barista.git"


def fake_roast(host="cmslpc", step_name="B"):
    return {
        "id": FAKE_ID, "label": "citest", "created": "2026-01-01 00:00:00", "owner": "CI",
        "barista": {"sha": "a" * 40, "origin": GITLAB, "web": "https://gitlab.cern.ch/cms-cmu/barista"},
        "coffea4bees": {"sha": "b" * 40, "origin": GITLAB, "web": "https://gitlab.cern.ch/cms-cmu/coffea4bees"},
        "config": {"source": "coffea4bees/workflows/config/nominal_run2.yml",
                   "captured": "config.yml", "sha256": "c" * 64},
        "steps": [{"name": step_name, "host": host,
                   "snakefile": "coffea4bees/workflows/Snakefile_PhaseB.smk", "targets": "", "extra": ""}],
        "hosts": {host: {"checkout": f"~/prod/{FAKE_ID}/barista", "ssh": CFG["hosts"][host]["ssh"]}},
        "history": [],
    }


def bash_ok(script, name):
    """bash -n: parse without executing."""
    res = subprocess.run([BASH, "-n"], input=script, text=True, capture_output=True)
    return res.returncode == 0, f"{name}: {res.stderr.strip()}"


@unittest.skipUnless((REPO / "coffea4bees" / "workflows").is_dir(),
                     "coffea4bees not checked out (CI clones it in the setup stage; this job needs nothing)")
class TestPhaseTable(unittest.TestCase):
    """roast --phases maps to real Snakefiles; a workflow rename must not silently break it.

    coffea4bees lives in its own repo and is cloned into the workspace by the `setup` stage,
    so this class is skipped wherever it is absent and does its work in a full checkout
    (your working tree, or any job that has the clone)."""

    def test_every_phase_snakefile_exists(self):
        for phase, (host, smk) in roast.PHASES.items():
            with self.subTest(phase=phase):
                self.assertTrue((REPO / smk).is_file(), f"phase {phase}: {smk} missing")
                self.assertIn(host, CFG["hosts"], f"phase {phase}: unknown host {host}")

    def test_phase_hosts_match_the_documented_split(self):
        # coffea4bees/workflows/README.md: A, B, E, F on cmslpc; C, D on falcon (GPU).
        for phase in "ABEF":
            self.assertEqual(roast.PHASES[phase][0], "cmslpc")
        for phase in "CD":
            self.assertEqual(roast.PHASES[phase][0], "falcon")


@unittest.skipIf(BASH is None, "bash not available")
class TestGeneratedBashParses(unittest.TestCase):
    """Every script roast pipes to a host must be valid bash."""

    @classmethod
    def setUpClass(cls):
        # _run_script reads the captured config to redirect --test outputs
        cls.rdir = roast.roast_dir(FAKE_ID)
        cls.rdir.mkdir(parents=True, exist_ok=True)
        (cls.rdir / "config.yml").write_text('label: "CI"\noutput_path: "output/CI/"\n')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.rdir, ignore_errors=True)

    def test_run_script(self):
        r = fake_roast()
        for resume in (False, True):
            for extra in ("", "-n", "--config test=true", "--resources gres=mps:25"):
                with self.subTest(resume=resume, extra=extra):
                    s = roast._run_script(CFG, r, r["steps"][0], "~/prod/x/barista", 8, extra, resume)
                    ok, msg = bash_ok(s, "run script")
                    self.assertTrue(ok, msg)

    def test_copy_script(self):
        r = fake_roast()
        for kind in ("publish", "archive"):
            for dry in (False, True):
                with self.subTest(kind=kind, dry=dry):
                    ps = roast.copy_settings(kind, CFG, r)
                    s = roast._copy_script(r, "~/prod/x/barista", "root://eosuser.cern.ch",
                                           "/eos/user/u/user/www/HH4b/prod/" + FAKE_ID, ps,
                                           dry_run=dry, htaccess=(kind == "publish"))
                    ok, msg = bash_ok(s, f"{kind} script")
                    self.assertTrue(ok, msg)

    def test_status_script(self):
        for host in ("cmslpc", "falcon"):
            with self.subTest(host=host):
                r = fake_roast(host=host)
                s = roast._status_script(r, "~/prod/x/barista", r["steps"], host)
                ok, msg = bash_ok(s, f"status script ({host})")
                self.assertTrue(ok, msg)

    def test_checkout_and_cleanup_scripts(self):
        r = fake_roast()
        hc = CFG["hosts"]["cmslpc"]
        for name, s in (("checkout", roast._checkout_script(hc, r, "~/prod/x/barista")),
                        ("finish checkout", roast._finish_checkout_script(r, "~/prod/x/barista")),
                        ("eos rm tree", roast._eos_rm_tree_script("root://cmseos.fnal.gov", "/store/user/u/x"))):
            with self.subTest(script=name):
                ok, msg = bash_ok(s, name)
                self.assertTrue(ok, msg)


class TestRunScriptFlags(unittest.TestCase):
    """The snakemake command line is the contract between a roast and its results."""

    @classmethod
    def setUpClass(cls):
        cls.rdir = roast.roast_dir(FAKE_ID)
        cls.rdir.mkdir(parents=True, exist_ok=True)
        (cls.rdir / "config.yml").write_text('label: "CI"\noutput_path: "output/CI/"\n')
        cls.r = fake_roast()
        cls.step = cls.r["steps"][0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.rdir, ignore_errors=True)

    def script(self, extra="", resume=False):
        return roast._run_script(CFG, self.r, self.step, "~/prod/x/barista", 8, extra, resume)

    def test_pins_the_captured_config_and_run_id(self):
        s = self.script()
        self.assertIn(f"--configfile roasts/{FAKE_ID}/config.yml", s)
        self.assertIn(f"--config roast_id={FAKE_ID}", s)   # resolves {roast_id} in the workflow config
        self.assertIn("--printshellcmds", s)               # so logs record the commands that ran

    def test_submit_does_not_unlock_or_rerun(self):
        s = self.script()
        self.assertNotIn("--unlock", s)
        self.assertNotIn("--rerun-incomplete", s)

    def test_resume_unlocks_and_reruns_incomplete(self):
        s = self.script(resume=True)
        self.assertIn("--unlock", s)
        self.assertIn("--rerun-incomplete", s)

    def test_body_is_wrapped_in_main(self):
        # Bash reads a script by byte offset: if the file is regenerated mid-run, an
        # unwrapped body resumes at a stale offset and executes fragments.
        s = self.script()
        self.assertIn("main() {", s)
        self.assertTrue(s.rstrip().endswith('main "$@"'), "launcher must call main() at the end")

    def test_records_exit_code_and_keeps_a_failed_step_open(self):
        s = self.script()
        self.assertIn("logs/B.exit", s)
        self.assertIn("PIPESTATUS", s)
        self.assertIn("exec bash", s)

    def test_seeds_the_grid_proxy(self):
        self.assertIn("proxy/x509_proxy", self.script())


class TestCopySettings(unittest.TestCase):
    def test_defaults(self):
        r = fake_roast()
        pub = roast.copy_settings("publish", CFG, r)
        arc = roast.copy_settings("archive", CFG, r)
        self.assertIn("*.png", pub["include"])
        self.assertIn("*.coffea", arc["include"])
        self.assertEqual(arc["max_mb"], 0)             # no size cap on archived products

    def test_user_config_then_roast_rules_win(self):
        r = fake_roast()
        cfg = dict(CFG, archive={"max_mb": 10})
        self.assertEqual(roast.copy_settings("archive", cfg, r)["max_mb"], 10)
        r["archive_rules"] = {"max_mb": 99}
        self.assertEqual(roast.copy_settings("archive", cfg, r)["max_mb"], 99)

    def test_test_slices_are_excluded_unless_asked_for(self):
        r = fake_roast()
        self.assertTrue(any("_test" in e for e in roast.copy_settings("publish", CFG, r)["exclude"]))
        self.assertFalse(any("_test" in e for e in roast.copy_settings("publish", CFG, r, include_test=True)["exclude"]))

    def test_archive_keeps_only_merged_products(self):
        excl = roast.copy_settings("archive", CFG, fake_roast())["exclude"]
        self.assertTrue(any("singlefiles" in e for e in excl))
        self.assertTrue(any("classifier_inputs_dataset_" in e for e in excl))

    def test_publish_skips_dask_reports_and_per_job_logs(self):
        ps = roast.copy_settings("publish", CFG, fake_roast())
        self.assertTrue(any("dask-report" in e for e in ps["exclude"]))
        self.assertTrue(any(e.startswith("output/") and e.endswith("logs") for e in ps["exclude"]))


class TestHelpers(unittest.TestCase):
    def test_rq_expands_a_leading_tilde_on_the_remote(self):
        self.assertEqual(roast.rq("~/work/x"), '"$HOME/work/x"')
        self.assertEqual(roast.rq("/abs/path"), "/abs/path")

    @unittest.skipIf(BASH is None, "bash not available")
    def test_rq_keeps_substitutions_inert(self):
        # ~ must expand on the remote, but nothing else in the path may be evaluated.
        quoted = roast.rq("~/a$(whoami)`id`")
        out = subprocess.run([BASH, "-c", f"printf %s {quoted}"], text=True, capture_output=True).stdout
        self.assertEqual(out, str(Path.home()) + "/a$(whoami)`id`")

    def test_gitlab_web_url_from_ssh_remote(self):
        self.assertEqual(roast.gitlab_web(GITLAB), "https://gitlab.cern.ch/cms-cmu/barista")
        self.assertEqual(roast.gitlab_web("git@github.com:o/r.git"), "https://github.com/o/r")

    def test_resolve_ssh_follows_the_pin_file(self):
        hc = dict(CFG["hosts"]["cmslpc"])
        self.assertEqual(roast.resolve_ssh(hc), "u@cmslpc307.fnal.gov")
        with tempfile.NamedTemporaryFile("w", suffix=".host", delete=False) as f:
            f.write("cmslpc999\n")
            hc["host_file"] = f.name
        try:
            self.assertEqual(roast.resolve_ssh(hc), "u@cmslpc999.fnal.gov")
        finally:
            os.unlink(f.name)


class TestCommandLine(unittest.TestCase):
    """Argparse wiring: every subcommand must parse its own flags.

    A greedy positional (nargs=REMAINDER) once swallowed a sibling option, so the
    subcommand refused a flag it declares.  --help exercises the parser cheaply.
    """

    SUBCOMMANDS = ["init", "new", "checkout", "submit", "resume", "status", "attach", "proxy",
                   "publish", "archive", "pourover", "pull", "index", "ls", "show", "rm"]

    def run_roast(self, *args):
        return subprocess.run([sys.executable, str(REPO / "src" / "tools" / "roast.py"), *args],
                              text=True, capture_output=True, cwd=REPO)

    def test_top_level_help(self):
        res = self.run_roast("--help")
        self.assertEqual(res.returncode, 0, res.stderr)
        for sub in self.SUBCOMMANDS:
            self.assertIn(sub, res.stdout, f"{sub} missing from the top-level help")

    def test_each_subcommand_help(self):
        for sub in self.SUBCOMMANDS:
            with self.subTest(subcommand=sub):
                res = self.run_roast(sub, "--help")
                self.assertEqual(res.returncode, 0, f"{sub} --help: {res.stderr.strip()}")
                self.assertIn("usage:", res.stdout)

    def test_unknown_subcommand_fails_cleanly(self):
        res = self.run_roast("frappuccino")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("invalid choice", res.stderr)


@unittest.skipIf(BASH is None or shutil.which("tac") is None, "needs bash and tac (GNU coreutils)")
class TestStatusScriptBehaviour(unittest.TestCase):
    """The status script runs on the host; execute it here against a fake checkout.

    "running" must mean a live snakemake driver, not merely "no exit file yet":
    a driver killed by a node reboot or by oomd used to read as running forever.
    """

    def classify(self, log=None, exit_code=None):
        r = fake_roast()
        with tempfile.TemporaryDirectory() as d:
            if log is not None:
                (Path(d) / "logs").mkdir()
                (Path(d) / "logs" / "B.log").write_text(log)
                if exit_code is not None:
                    (Path(d) / "logs" / "B.exit").write_text(f"{exit_code}\n")
            script = roast._status_script(r, d, r["steps"], "cmslpc")
            res = subprocess.run([BASH], input=script, text=True, capture_output=True, cwd=d)
            for line in res.stdout.splitlines():
                if line.startswith("STEP|"):
                    return line.split("|")[2]
        return None

    def test_not_started(self):
        self.assertEqual(self.classify(), "not started")

    def test_finished_reports_its_exit_code(self):
        self.assertEqual(self.classify(log="=== roast x step B start now ===\ndone\n", exit_code=0), "exit=0")
        self.assertEqual(self.classify(log="=== roast x step B start now ===\nboom\n", exit_code=1), "exit=1")

    def test_dead_driver_after_a_failed_job_is_an_error_not_running(self):
        log = ("=== roast x step B start now ===\n"
               "Error in rule analysis_dataset:\n"
               "WorkflowError:\nAt least one job did not complete successfully.\n")
        self.assertEqual(self.classify(log=log), "error")

    def test_dead_driver_with_no_error_is_stalled(self):
        self.assertEqual(self.classify(log="=== roast x step B start now ===\nquietly gone\n"), "stalled")

    def test_only_the_latest_run_segment_is_inspected(self):
        # An old failure followed by a clean restart must not colour the current run.
        log = ("=== roast x step B start old ===\nWorkflowError:\n"
               "=== roast x step B exit 1 old ===\n"
               "=== roast x step B start now ===\nall good\n")
        self.assertEqual(self.classify(log=log), "stalled")


if __name__ == "__main__":
    unittest.main(verbosity=2)
