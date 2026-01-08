#!/usr/bin/env python3
import math
import sys
import yaml

def close(a, b, rel=1e-9, abs_=1e-12):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)

def deep_compare(got, exp, path=""):
    # dict
    if isinstance(exp, dict):
        if not isinstance(got, dict):
            raise AssertionError(f"{path}: expected dict, got {type(got).__name__}")
        missing = set(exp) - set(got)
        extra = set(got) - set(exp)
        if missing:
            raise AssertionError(f"{path}: missing keys {sorted(missing)}")
        if extra:
            raise AssertionError(f"{path}: extra keys {sorted(extra)}")
        for k in exp:
            deep_compare(got[k], exp[k], path + f".{k}")
        return

    # list/tuple
    if isinstance(exp, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(got) != len(exp):
            raise AssertionError(f"{path}: expected seq len {len(exp)}, got {type(got).__name__} len {len(got) if hasattr(got,'__len__') else 'n/a'}")
        for i, (g, e) in enumerate(zip(got, exp)):
            deep_compare(g, e, path + f"[{i}]")
        return

    # numbers: compare with tolerance
    if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        if not close(got, exp, rel=1e-6, abs_=1e-9):  # adjust tolerances as needed
            raise AssertionError(f"{path}: {got} != {exp} (tolerance)")
        return

    # everything else: exact
    if got != exp:
        raise AssertionError(f"{path}: {got!r} != {exp!r}")

def main(got_path, exp_path):
    with open(got_path) as f:
        got = yaml.safe_load(f)
    with open(exp_path) as f:
        exp = yaml.safe_load(f)

    deep_compare(got, exp, path="$")
    print("YAML matches expectation ✅")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: check_yaml.py <generated.yaml> <expected.yaml>", file=sys.stderr)
        sys.exit(2)
    try:
        main(sys.argv[1], sys.argv[2])
    except AssertionError as e:
        print(f"YAML mismatch ❌: {e}", file=sys.stderr)
        sys.exit(1)
