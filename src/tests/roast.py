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
import json
import os
import re
import shlex
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

    def test_targets_come_before_the_config_option(self):
        # `--config` takes every argument after it: a target placed after `--config roast_id=...`
        # is parsed as a malformed name=value entry and snakemake exits before running anything.
        import argparse
        step = {**self.step, "targets": "all_M1"}
        s = roast._run_script(CFG, self.r, step, "~/prod/x/barista", 8, "-n", False, targets="all_M2 out/x.yml")
        cmd = next(l for l in s.splitlines() if "snakemake -s" in l)
        argv = shlex.split(cmd.split("snakemake", 1)[1].split("2>&1")[0].split("|")[0])
        p = argparse.ArgumentParser()        # the shape of snakemake's own parser for these options
        p.add_argument("targets", nargs="*")
        p.add_argument("-s"); p.add_argument("--configfile"); p.add_argument("--cores"); p.add_argument("--jobs")
        p.add_argument("--printshellcmds", action="store_true"); p.add_argument("-n", action="store_true")
        p.add_argument("--config", nargs="*")
        ns = p.parse_args(argv)
        self.assertEqual(ns.targets, ["all_M1", "all_M2", "out/x.yml"])
        self.assertEqual(ns.config, [f"roast_id={FAKE_ID}"])
        self.assertTrue(ns.n)


class TestRoastSsh(unittest.TestCase):
    """A roast follows the node it was placed on, not whatever the config says today."""

    def test_recorded_node_wins_over_the_configured_target(self):
        r = fake_roast()
        r["hosts"]["cmslpc"]["ssh"] = "u@cmslpc361.fnal.gov"
        cfg = json.loads(json.dumps(CFG))
        cfg["hosts"]["cmslpc"]["ssh"] = "u@cmslpc-el9.fnal.gov"   # the round-robin gateway
        self.assertEqual(roast.roast_ssh(cfg, r, "cmslpc"), "u@cmslpc361.fnal.gov")

    def test_falls_back_to_the_config_before_checkout(self):
        r = fake_roast()
        r["hosts"] = {}
        self.assertEqual(roast.roast_ssh(CFG, r, "cmslpc"), CFG["hosts"]["cmslpc"]["ssh"])

    def test_checkout_asks_the_far_side_which_node_it_is(self):
        script = roast._checkout_script(CFG["hosts"]["cmslpc"], fake_roast(), "~/prod/x/barista")
        self.assertIn("hostname -f", script)


class TestConcurrentRoasts(unittest.TestCase):
    """Several roasts can be in flight at once, so nothing may be named per label alone.

    submit closes a tmux window of the name it is about to use, so two roasts sharing a
    window name would let one kill the other's running step.
    """

    def window(self, label, date, shas, step="D"):
        r = fake_roast(step_name=step)
        r.update(label=label, created=f"{date} 00:00:00", id=f"{label}_{date.replace('-', '')}_{shas}")
        return roast._window_name(r, r["steps"][0])

    def test_same_label_and_day_different_code_are_distinct(self):
        a = self.window("nominal_run3", "2026-09-22", "3f9e199-1e0504f")
        b = self.window("nominal_run3", "2026-09-22", "aaaaaaa-bbbbbbb")
        self.assertNotEqual(a, b)

    def test_steps_of_one_roast_are_distinct(self):
        r = fake_roast()
        c = dict(r["steps"][0], name="C")
        d = dict(r["steps"][0], name="D")
        self.assertNotEqual(roast._window_name(r, c), roast._window_name(r, d))

    def test_name_is_tmux_safe(self):
        w = self.window("nominal_run3", "2026-09-22", "3f9e199-1e0504f")
        self.assertNotIn(" ", w)
        self.assertNotIn(":", w)     # tmux target syntax is session:window
        self.assertLessEqual(len(w), 40)


try:
    import yaml
except ImportError:          # the CI image is bare python; only the layered-config path needs PyYAML
    yaml = None


class TestCaptureConfig(unittest.TestCase):
    """`roast new` captures the config the steps run from.  A config with `base:` holds only its
    differences and must be captured merged, self-contained, so later edits to the base cannot
    change what an existing roast runs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = roast.ROOT
        roast.ROOT = self.tmp            # base: paths resolve against the barista root

    def tearDown(self):
        roast.ROOT = self.root
        shutil.rmtree(self.tmp)

    def write(self, name, text):
        p = self.tmp / name
        p.write_text(text)
        return p

    def test_plain_config_is_copied_verbatim(self):
        src = self.write("plain.yml", "# comment kept\nlabel: x\n")
        self.assertEqual(roast.capture_config(src, self.tmp / "out.yml"), [])
        self.assertEqual((self.tmp / "out.yml").read_text(), src.read_text())

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_layered_config_is_merged_over_its_base_chain(self):
        self.write("root.yml", "label: a\noutput_path: output/a/\nlst: [1, 2]\n"
                               "analysis_config:\n  processor: p.py\n  config:\n    run_SvB: true\n    tight: true\n")
        self.write("mid.yml", "base: root.yml\nlst: [3]\n")
        src = self.write("top.yml", "base: mid.yml\nlabel: b\noutput_path: output/b/\n"
                                    "analysis_config:\n  config:\n    cand: q.yml\n")
        chain = roast.capture_config(src, self.tmp / "out.yml")
        self.assertEqual([b["source"] for b in chain], ["root.yml", "mid.yml"])
        text = (self.tmp / "out.yml").read_text()
        got = yaml.safe_load(text)
        self.assertNotIn("base", got)
        self.assertEqual(got["label"], "b")
        self.assertEqual(got["lst"], [3])                           # lists replace, like snakemake
        self.assertEqual(got["analysis_config"]["processor"], "p.py")
        self.assertEqual(got["analysis_config"]["config"],
                         {"run_SvB": True, "tight": True, "cand": "q.yml"})  # nested siblings survive
        # roast status/submit read output_path back out of the captured text with this regex
        m = re.search(r'^output_path:\s*["\']?([^"\'\s#]+)', text, re.M)
        self.assertEqual(m.group(1), "output/b/")

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_placeholders_survive_the_round_trip(self):
        self.write("root.yml", 'handoff:\n  eos_base: "root://x//{roast_id}/handoff"\n'
                               "fvt:\n  workflow_overrides:\n    '--friends \"\"': a/{roast_id}.json@@HCR_input\n")
        src = self.write("top.yml", "base: root.yml\nlabel: b\n")
        roast.capture_config(src, self.tmp / "out.yml")
        got = yaml.safe_load((self.tmp / "out.yml").read_text())
        self.assertEqual(got["handoff"]["eos_base"], "root://x//{roast_id}/handoff")
        self.assertEqual(got["fvt"]["workflow_overrides"]['--friends ""'], "a/{roast_id}.json@@HCR_input")

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_base_cycle_is_an_error(self):
        self.write("a.yml", "base: b.yml\n")
        src = self.write("b.yml", "base: a.yml\n")
        with self.assertRaises(SystemExit):
            roast.capture_config(src, self.tmp / "out.yml")


class TestCheckInputs(unittest.TestCase):
    """A dependent roast (e.g. mixed-data production reading the nominal's FvT) names what it reads
    from other roasts under `inputs:`.  `roast new` must refuse a URL that is not inside a named
    upstream roast's EOS area -- a hand-run product or another production's would otherwise be
    read silently."""

    UP = "nominal_run3_20260922_3f9e199-1e0504f"
    AREA = f"root://cmseos.fnal.gov//store/user/u/HH4b_prod/{UP}"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.roasts = roast.ROASTS
        roast.ROASTS = self.tmp
        self.add_roast(self.UP, archived=True)

    def tearDown(self):
        roast.ROASTS = self.roasts
        shutil.rmtree(self.tmp)

    def add_roast(self, rid, archived):
        r = fake_roast()
        r["id"] = rid
        if archived:
            r["archive"] = {"eos": f"root://cmseos.fnal.gov//store/user/u/HH4b_prod/{rid}", "ok": True}
        (self.tmp / rid).mkdir()
        (self.tmp / rid / "roast.json").write_text(json.dumps(r))

    def check(self, inputs):
        return roast.check_inputs(CFG, {"label": "x", "inputs": inputs})

    def test_no_inputs_block(self):
        self.assertIsNone(roast.check_inputs(CFG, {"label": "x"}))

    def test_urls_inside_the_upstream_area_are_recorded(self):
        fvt = f"{self.AREA}/friend/FvT_nominal/result.json@@analysis.0.merged"
        rec = self.check({"upstream_roasts": self.UP, "FvT": fvt, "hemilib": {"registry": f"{self.AREA}/hemilib/h.yml"}})
        self.assertEqual(rec["refs"]["FvT"], {"url": fvt, "roast": self.UP})
        self.assertIn("hemilib.registry", rec["refs"])                       # nested keys, dotted
        self.assertEqual(rec["upstream"][0]["id"], self.UP)
        self.assertEqual(rec["upstream"][0]["barista"], "a" * 40)            # upstream code pinned in the record
        self.assertTrue(rec["upstream"][0]["archived"])

    def test_slash_differences_do_not_matter(self):
        url = f"root://cmseos.fnal.gov/store/user/u/HH4b_prod/{self.UP}//friend/x.json"
        self.assertIn("x", self.check({"upstream_roasts": [self.UP], "x": url})["refs"])

    def test_url_outside_every_upstream_is_an_error(self):
        handrun = "root://cmseos.fnal.gov//store/user/u/HH4b_Run3_v2/friend/FvT/result.json"
        with self.assertRaises(SystemExit):
            self.check({"upstream_roasts": self.UP, "FvT": handrun})

    def test_a_longer_id_is_not_inside_a_shorter_one(self):
        with self.assertRaises(SystemExit):
            self.check({"upstream_roasts": self.UP, "x": f"{self.AREA}_rerun/friend/x.json"})

    def test_local_paths_and_placeholders_are_errors(self):
        for bad in ("coffea4bees/metadata/friends/x.json", f"{self.AREA}/{{roast_id}}/x", 3):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                self.check({"upstream_roasts": self.UP, "x": bad})

    def test_upstream_must_be_named_and_exist(self):
        with self.assertRaises(SystemExit):
            self.check({"x": f"{self.AREA}/x"})
        with self.assertRaises(SystemExit):
            self.check({"upstream_roasts": "no_such_roast_20260101_aaaaaaa-bbbbbbb", "x": f"{self.AREA}/x"})

    def test_unarchived_upstream_uses_its_eos_namespace(self):
        rid = "mixeddata_run3_20260925_ccccccc-ddddddd"
        self.add_roast(rid, archived=False)
        url = f"root://cmseos.fnal.gov//store/user/u/HH4b_prod/{rid}/hemilib/h.yml"
        rec = self.check({"upstream_roasts": [self.UP, rid], "hemilib": url})
        self.assertEqual(rec["refs"]["hemilib"]["roast"], rid)
        self.assertFalse(rec["upstream"][1]["archived"])


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


class TestStatusGrouping(unittest.TestCase):
    """Slurm jobs are reported under the step that submitted them.

    The step comes from whichever logs/<step>.log announced the job id, so two phases that
    both run a rule called `train` keep their own attempts instead of one hiding the other.
    """

    OUT = "\n".join([
        "STEP|C|exit=0|tmux=0|5 of 5 steps (100%) done|done",
        "STEP|D|running|tmux=1||Job 1 submitted",
        "SLURM|42722|D|rule_train|RUNNING|18:46|8:00:00|rogue02|8|62G|gres/mps:50|loss=0.47",
        "SLURMDONE|42442|C|rule_train|COMPLETED|04:35:59|",
        "SLURMDONE|42717|D|rule_train|FAILED|00:00:10|",
        "SLURMDONE|42429||classifier_batch|FAILED|00:00:05|",
        "SINFO|work*|2|mixed|gpu:1",
    ])

    def setUp(self):
        self.st = roast._parse_status(self.OUT)

    def test_steps_are_kept_in_order(self):
        self.assertEqual([n for n, _ in self.st["steps"]], ["C", "D"])

    def test_step_name_is_a_column_not_part_of_the_text(self):
        # cmd_status prints the name itself, so leaving it in the text would double it.
        for name, text in self.st["steps"]:
            self.assertFalse(text.startswith(name), f"{name!r} repeated in its own line text")

    def test_jobs_land_under_their_own_step(self):
        self.assertEqual([j["jid"] for j in self.st["done"]["C"]], ["42442"])
        self.assertEqual([j["jid"] for j in self.st["live"]["D"]], ["42722"])

    def test_a_rule_shared_by_two_steps_is_not_merged(self):
        # C's finished train must survive D's running train of the same rule name.
        c = roast._slurm_rows(self.st["live"].get("C", []), self.st["done"].get("C", []))
        d = roast._slurm_rows(self.st["live"].get("D", []), self.st["done"].get("D", []))
        self.assertIn("42442", "\n".join(c))
        self.assertIn("42722", "\n".join(d))
        self.assertNotIn("42442", "\n".join(d))

    def test_unattributed_jobs_go_to_the_empty_key(self):
        self.assertEqual([j["jid"] for j in self.st["done"][""]], ["42429"])

    def test_cluster_line_is_kept_aside(self):
        self.assertTrue(any("cluster" in e for e in self.st["extra"]))

    def test_missing_checkout_is_flagged(self):
        self.assertTrue(roast._parse_status("NOCHECKOUT")["nocheckout"])


class TestSlurmRowsSupersede(unittest.TestCase):
    """A retried rule must show the attempt that replaced the failure, not the failure.

    Snakemake resubmits failed rules and people retry jobs by hand, so the scheduler holds
    several job ids per rule; listing them all made a fixed rule look broken.
    """

    LIVE = ("jid", "rule", "state", "el", "lim", "node", "cpus", "mem", "tres", "tail")
    DONE = ("jid", "rule", "state", "el", "rss")

    def live(self, **kw):
        return dict({k: "" for k in self.LIVE}, **kw)

    def done(self, **kw):
        return dict({k: "" for k in self.DONE}, **kw)

    def render(self, live, done):
        return "\n".join(roast._slurm_rows(live, done))

    def test_latest_finished_attempt_wins(self):
        out = self.render([], [self.done(jid="41090", rule="rule_train", state="FAILED"),
                               self.done(jid="41095", rule="rule_train", state="COMPLETED")])
        self.assertIn("41095", out)
        self.assertNotIn("41090", out)
        self.assertIn("after 1 failed attempt", out)

    def test_a_live_attempt_supersedes_finished_ones(self):
        out = self.render([self.live(jid="41100", rule="rule_train", state="RUNNING")],
                          [self.done(jid="41090", rule="rule_train", state="FAILED")])
        self.assertIn("41100", out)
        self.assertNotIn("41090", out)

    def test_an_unretried_failure_is_still_reported(self):
        out = self.render([], [self.done(jid="41099", rule="rule_analyze", state="FAILED")])
        self.assertIn("41099", out)
        self.assertIn("FAILED", out)

    def test_other_rules_are_untouched(self):
        out = self.render([], [self.done(jid="41090", rule="rule_train", state="FAILED"),
                               self.done(jid="41095", rule="rule_train", state="COMPLETED"),
                               self.done(jid="41096", rule="rule_evaluate", state="COMPLETED")])
        self.assertIn("41096", out)
        self.assertIn("evaluate", out)

    def test_rule_prefix_is_stripped(self):
        out = self.render([], [self.done(jid="1", rule="rule_plot_weights", state="COMPLETED")])
        self.assertIn("plot_weights", out)
        self.assertNotIn("rule_plot_weights", out)


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
