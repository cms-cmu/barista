#!/usr/bin/env python3
"""Unified test runner for barista unit tests.

Executes all unit test suites with verbose per-test output (pytest -v / unittest verbosity=2)
and provides a clear pass/fail summary and exit code for CI pipelines.
"""
import os
import subprocess
import sys
import time


def run_command(title, cmd):
    """Run a test command with visual header and verbose output."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*80}\n", flush=True)

    start_time = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(cmd, env=env)
    elapsed = time.perf_counter() - start_time
    success = (result.returncode == 0)

    return success, elapsed


def main():
    total_start = time.perf_counter()
    all_results = []
    overall_success = True

    # 1. Core Framework & Runner Pytests
    runner_tests = [
        "src/tests/test_runner.py",
        "src/tests/test_dataset_resolution.py",
        "src/tests/test_cutflow_table.py",
        "src/tests/test_compare_coffea_hists.py",
        "src/tests/test_picoaod_splitting.py",
        "src/tests/test_validate_config.py",
    ]
    # Filter to existing files
    existing_runner_tests = [f for f in runner_tests if os.path.exists(f)]
    if existing_runner_tests:
        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"] + existing_runner_tests
        success, elapsed = run_command("1/5: Core Runner & Configuration Tests", cmd)
        all_results.append(("Core Runner Tests (pytest)", success, elapsed))
        if not success:
            overall_success = False

    # 2. Histogram Collection Module
    if os.path.exists("src/tests/hist_collection.py"):
        cmd = [sys.executable, "-m", "src.tests.hist_collection"]
        success, elapsed = run_command("2/5: Histogram Collection", cmd)
        all_results.append(("Histogram Collection", success, elapsed))
        if not success:
            overall_success = False

    # 3. Kappa Framework
    if os.path.exists("src/tests/kappa_framework.py"):
        cmd = [sys.executable, "-m", "src.tests.kappa_framework"]
        success, elapsed = run_command("3/5: Kappa Framework", cmd)
        all_results.append(("Kappa Framework", success, elapsed))
        if not success:
            overall_success = False

    # 4. Friend Tree Merge
    if os.path.exists("src/tests/friendtree_merge.py"):
        cmd = [sys.executable, "-m", "src.tests.friendtree_merge"]
        success, elapsed = run_command("4/5: Friend Tree Merge", cmd)
        all_results.append(("Friend Tree Merge", success, elapsed))
        if not success:
            overall_success = False

    # 5. Classifier Config Dependencies
    if os.path.exists("src/classifier/test/check_configs.py"):
        cmd = [sys.executable, "-m", "src.classifier.test.check_configs"]
        success, elapsed = run_command("5/5: Classifier Config Dependencies", cmd)
        all_results.append(("Classifier Configs", success, elapsed))
        if not success:
            overall_success = False

    total_elapsed = time.perf_counter() - total_start

    # Final Summary Table
    print(f"\n{'='*80}")
    print("  BARISTA UNIT TESTS SUMMARY")
    print(f"{'='*80}")

    for name, success, elapsed in all_results:
        status = "PASSED" if success else "FAILED"
        print(f"  {name:<35} | {status:<6} | {elapsed:>6.2f}s")

    print(f"{'-'*80}")
    status_str = "SUCCESS" if overall_success else "FAILURE"
    print(f"  OVERALL: {status_str} in {total_elapsed:.2f}s")
    print(f"{'='*80}\n", flush=True)

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
