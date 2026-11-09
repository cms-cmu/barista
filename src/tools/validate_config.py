#!/usr/bin/env python3
"""validate_config.py -- Workflow Configuration Validator for barista & coffea4bees.

Audits workflow YAML configs for:
  1. Base inheritance resolution and layering integrity.
  2. Cross-section duplicate parameter consistency:
     - Dataset synchronization (analysis signal vs classifier_inputs vs fvt_closure)
     - Year/era alignment (year_eras vs classifier_setting eras)
     - Producer-consumer path consistency (Phase C/D friend outputs vs Phase F friend inputs)
     - Flag conflicts (require_trigWeight, apply_trigWeight, blind vs combine_flags)
  3. Reference integrity (existence of datasets, processors, plot configs, weight files).
  4. Core schema and structural completeness.

Can be run standalone:
    python coffea4bees/workflows/scripts/validate_config.py coffea4bees/workflows/config/nominal_run3.yml
    python coffea4bees/workflows/scripts/validate_config.py --ci
    python coffea4bees/workflows/scripts/validate_config.py --all

Or via roast:
    roast validate coffea4bees/workflows/config/nominal_run3.yml
    roast validate --all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required to validate workflow configs. (pip install pyyaml)", file=sys.stderr)
    sys.exit(2)


# Default configs used in the CI pipeline
DEFAULT_CI_CONFIGS = [
    "coffea4bees/workflows/config/nominal_run3_ci.yml",
    "coffea4bees/workflows/config/nominal_run3.yml",
    "coffea4bees/workflows/config/lowpt_run2.yml",
    "coffea4bees/workflows/config/ttHbb.yml",
    "coffea4bees/workflows/config/nominal_run2.yml",
]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    severity: str  # "ERROR", "WARNING", "INFO"
    category: str  # "SYNTAX", "CONSISTENCY", "REFERENCE", "SCHEMA"
    key: str
    message: str
    suggestion: str = ""

    def __str__(self) -> str:
        s = f"[{self.severity}] [{self.category}] {self.key}: {self.message}"
        if self.suggestion:
            s += f" -> Suggestion: {self.suggestion}"
        return s


@dataclass
class ValidationReport:
    config_path: Path
    bases: list[Path] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "INFO"]

    def add_issue(self, issue: Issue) -> None:
        for existing in self.issues:
            if (existing.severity, existing.category, existing.key, existing.message) == (issue.severity, issue.category, issue.key, issue.message):
                return
        self.issues.append(issue)

    def is_valid(self, strict: bool = False) -> bool:
        if strict:
            return len(self.errors) == 0 and len(self.warnings) == 0
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Path & Repository Discovery
# ---------------------------------------------------------------------------

def find_repo_root(start: Path | None = None) -> Path:
    """Find barista repository root directory (or coffea4bees root)."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "coffea4bees").is_dir() and (cand / "run_container").exists():
            return cand
        if (cand / "metadata" / "datasets").is_dir() and (cand / "workflows").is_dir():
            return cand.parent if cand.name == "coffea4bees" else cand
    # Fallback to script location
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "coffea4bees").is_dir() and (cand / "run_container").exists():
            return cand
        if (cand / "metadata" / "datasets").is_dir() and (cand / "workflows").is_dir():
            return cand.parent if cand.name == "coffea4bees" else cand
    return p


# ---------------------------------------------------------------------------
# Config Layering & Base Chain Resolution
# ---------------------------------------------------------------------------

def _merge_dict(base: dict, over: dict) -> dict:
    """Recursive dict merge where `over` wins; lists and scalars are replaced."""
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config_chain(path: Path, repo_root: Path, seen: tuple[Path, ...] = ()) -> tuple[dict, list[Path], list[Issue]]:
    """Recursively resolve the base: inheritance chain, innermost first."""
    issues: list[Issue] = []
    path = path.resolve()

    if path in seen:
        cycle_str = " -> ".join(str(p.name) for p in (*seen, path))
        issues.append(Issue(
            severity="ERROR",
            category="SYNTAX",
            key="base",
            message=f"Cyclic config inheritance detected: {cycle_str}",
            suggestion="Remove circular base reference"
        ))
        return {}, [], issues

    if not path.is_file():
        issues.append(Issue(
            severity="ERROR",
            category="SYNTAX",
            key="config",
            message=f"Config file not found: {path}",
            suggestion="Verify file path"
        ))
        return {}, [], issues

    try:
        content = path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content) or {}
    except Exception as e:
        issues.append(Issue(
            severity="ERROR",
            category="SYNTAX",
            key="yaml",
            message=f"YAML parsing error in {path.name}: {e}",
            suggestion="Check YAML syntax, indentation, and quotes"
        ))
        return {}, [], issues

    if not isinstance(parsed, dict):
        issues.append(Issue(
            severity="ERROR",
            category="SYNTAX",
            key="root",
            message=f"Expected mapping/dictionary at root of {path.name}, got {type(parsed).__name__}",
        ))
        return {}, [], issues

    # Discover base file(s)
    base_val = parsed.get("base")
    bases_to_load: list[Path] = []
    if base_val:
        if isinstance(base_val, str):
            bases_to_load.append(Path(base_val))
        elif isinstance(base_val, list):
            bases_to_load.extend(Path(b) for b in base_val)

    merged_base: dict = {}
    resolved_bases: list[Path] = []

    for b in bases_to_load:
        chosen_base = None
        cand1 = path.parent / b
        cand2 = repo_root / b
        cand3 = repo_root / "coffea4bees/workflows/config" / b
        cand4 = repo_root / "workflows/config" / b

        for cand in [cand1, cand2, cand3, cand4]:
            if cand.is_file():
                chosen_base = cand.resolve()
                break

        if not chosen_base:
            issues.append(Issue(
                severity="ERROR",
                category="SYNTAX",
                key="base",
                message=f"Base config '{b}' referenced by {path.name} not found",
                suggestion=f"Checked: {cand1}, {cand2}, {cand3}"
            ))
            continue

        b_dict, b_chain, b_issues = resolve_config_chain(chosen_base, repo_root, (*seen, path))
        issues.extend(b_issues)
        resolved_bases.extend(b_chain)
        resolved_bases.append(chosen_base)
        merged_base = _merge_dict(merged_base, b_dict)

    final_merged = _merge_dict(merged_base, parsed)
    return final_merged, resolved_bases, issues


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

def audit_schema_and_typos(cfg: dict, issues: list[Issue]) -> None:
    """Audit top-level structure and check for common schema typos."""
    expected_top_keys = {
        "analysis", "year", "year_eras", "tag", "base", "dataset", "datasets",
        "analysis_config", "analysis_datasets", "classifier_setting",
        "classifier_inputs", "fvt_closure", "jcm", "jcm_datasets",
        "fvt", "svb", "friends", "combine_flags", "weights", "workflow_file",
        "output_path", "test", "plots", "cutflow", "inputs", "upstream_roasts",
        "run_period", "label", "tree", "sample_features", "training_options",
        # Workflow and stage top-level overrides:
        "plot_config", "known_counts", "known_counts_fatal", "known_counts_full",
        "jcm_known_counts_NoJCM", "jcm_known_counts_wJCM", "jcm_plot_config",
        "jcm_extra_arguments", "jcm_known_counts_fatal", "per_year_jcm",
        "make_combine_inputs", "known_counts_test", "test_chunksize",
        "trigger_weights", "svb_ma", "multijet", "subsamples", "fvt_closure_plots",
        "metadata_friends", "skim_registry", "output_file", "output_dir",
    }
    for k in cfg.keys():
        if k not in expected_top_keys:
            # Check for near matches (typos)
            for exp in expected_top_keys:
                if len(k) > 4 and (k in exp or exp in k) and k != exp:
                    issues.append(Issue(
                        severity="WARNING",
                        category="SCHEMA",
                        key=k,
                        message=f"Unrecognized top-level key '{k}'. Did you mean '{exp}'?",
                        suggestion=f"Verify spelling: '{exp}'"
                    ))
                    break

    # Analysis channel sanity
    analysis = cfg.get("analysis")
    if analysis and analysis not in ("HH4b", "ttHbb", "HH4b_lowpt", "HH4b_Run3", "mixeddata"):
        issues.append(Issue(
            severity="WARNING",
            category="SCHEMA",
            key="analysis",
            message=f"Non-standard analysis channel '{analysis}'. Expected HH4b, ttHbb, or HH4b_lowpt",
            suggestion="Ensure analysis matches one of the recognized pipelines"
        ))


def _extract_dataset_list(val: Any) -> list[str]:
    """Helper to extract flat string dataset list from string, list, or dict."""
    if isinstance(val, list):
        return [str(item) for item in val if isinstance(item, (str, int))]
    elif isinstance(val, dict):
        return list(str(k) for k in val.keys())
    elif isinstance(val, str):
        return [s.strip() for s in val.split(",") if s.strip()]
    return []


def is_signal_sample(name: str) -> bool:
    """Heuristic to identify signal datasets vs data or background."""
    lower = name.lower()
    if lower in ("data", "mixeddata"):
        return False
    if lower.startswith("tt") or "ttto" in lower or "ttbar" in lower or "qcd" in lower or "wjets" in lower or "zjets" in lower:
        return False
    if any(k in lower for k in ("glugluto", "vbf", "hh", "tth", "zh", "wh", "zz", "signal", "kl-", "kt-", "c2-", "xs_")):
        return True
    return True


def audit_datasets_consistency(cfg: dict, issues: list[Issue]) -> None:
    """Audit dataset coverage across analysis_config, classifier_inputs, and fvt_closure."""
    top_datasets = _extract_dataset_list(cfg.get("dataset") or cfg.get("datasets"))
    ac_datasets = _extract_dataset_list((cfg.get("analysis_config") or {}).get("datasets") or (cfg.get("analysis_config") or {}).get("dataset"))
    analysis_samples = set(top_datasets + ac_datasets)

    ci_datasets = set(_extract_dataset_list((cfg.get("classifier_inputs") or {}).get("datasets")))
    closure_datasets = set(_extract_dataset_list((cfg.get("fvt_closure") or {}).get("datasets")))

    # Signal sample coverage in Phase B.2 friend trees
    if ci_datasets:
        signals_in_analysis = {d for d in analysis_samples if is_signal_sample(d)}
        signals_in_ci = {d for d in ci_datasets if is_signal_sample(d)}

        missing_signals = signals_in_analysis - signals_in_ci
        if missing_signals:
            issues.append(Issue(
                severity="ERROR",
                category="CONSISTENCY",
                key="classifier_inputs.datasets",
                message=f"Signal dataset(s) {sorted(missing_signals)} present in analysis dataset list but MISSING from classifier_inputs.datasets",
                suggestion="Add missing signal datasets to classifier_inputs.datasets so Phase B.2 generates friend trees for Phase F"
            ))

    # Background coverage in closure vs classifier_inputs
    if closure_datasets and ci_datasets:
        ttbar_closure = {d for d in closure_datasets if "tt" in d.lower()}
        ttbar_ci = {d for d in ci_datasets if "tt" in d.lower()}
        missing_closure_ttbar = ttbar_closure - ttbar_ci
        if missing_closure_ttbar:
            issues.append(Issue(
                severity="WARNING",
                category="CONSISTENCY",
                key="fvt_closure.datasets",
                message=f"ttbar dataset(s) {sorted(missing_closure_ttbar)} are in fvt_closure but missing from classifier_inputs.datasets",
                suggestion="Align ttbar datasets between fvt_closure and classifier_inputs"
            ))

    # Data presence
    if analysis_samples and "data" not in [d.lower() for d in analysis_samples]:
        issues.append(Issue(
            severity="INFO",
            category="CONSISTENCY",
            key="dataset",
            message="No 'data' dataset listed in analysis dataset list (MC-only or test configuration)",
        ))


def _find_nested_eras(val: Any) -> list[dict[str, list[str]]]:
    """Recursively search for dictionaries mapping years/periods to era lists."""
    results: list[dict[str, list[str]]] = []
    if isinstance(val, dict):
        if "eras" in val and isinstance(val["eras"], dict):
            results.append(val["eras"])
        for k, v in val.items():
            if k == "eras" and isinstance(v, dict):
                results.append(v)
            else:
                results.extend(_find_nested_eras(v))
    elif isinstance(val, list):
        for item in val:
            results.extend(_find_nested_eras(item))
    return results


def audit_year_eras_consistency(cfg: dict, issues: list[Issue]) -> None:
    """Audit era consistency between top-level year_eras and classifier_setting."""
    top_eras = cfg.get("year_eras")
    if not isinstance(top_eras, dict):
        return

    for yr, eras in top_eras.items():
        if not isinstance(eras, list) or len(eras) == 0:
            issues.append(Issue(
                severity="ERROR",
                category="SCHEMA",
                key=f"year_eras.{yr}",
                message=f"Year/Period '{yr}' must map to a non-empty list of eras",
                suggestion=f"e.g. {yr}: ['C', 'D']"
            ))

    cls_settings = cfg.get("classifier_setting")
    if cls_settings:
        nested_eras_list = _find_nested_eras(cls_settings)
        for nested in nested_eras_list:
            top_keys = set(str(k) for k in top_eras.keys())
            nested_keys = set(str(k) for k in nested.keys())

            diff_top = top_keys - nested_keys
            if diff_top:
                issues.append(Issue(
                    severity="WARNING",
                    category="CONSISTENCY",
                    key="classifier_setting.eras",
                    message=f"Top-level year_eras has periods {sorted(diff_top)} which are missing from classifier_setting.eras",
                    suggestion="Add missing periods to classifier_setting.eras"
                ))

            for common_k in top_keys.intersection(nested_keys):
                top_v = set(str(e) for e in top_eras[common_k])
                nested_v = set(str(e) for e in nested[common_k])
                if top_v != nested_v:
                    issues.append(Issue(
                        severity="WARNING",
                        category="CONSISTENCY",
                        key=f"classifier_setting.eras.{common_k}",
                        message=f"Era mismatch for '{common_k}': year_eras has {sorted(top_v)} while classifier_setting has {sorted(nested_v)}",
                        suggestion="Align eras between top-level year_eras and classifier_setting"
                    ))


def audit_flags_and_duplicates(cfg: dict, issues: list[Issue]) -> None:
    """Audit duplicate flags across sections and check for conflicting values."""
    trig_flags: dict[str, bool] = {}

    def scan_trig(path_prefix: str, d: Any):
        if not isinstance(d, dict):
            return
        for k in ("require_trigWeight", "apply_trigWeight", "require_trigweight"):
            if k in d and isinstance(d[k], bool):
                trig_flags[f"{path_prefix}.{k}"] = d[k]
        for k, v in d.items():
            if isinstance(v, dict):
                scan_trig(f"{path_prefix}.{k}", v)

    scan_trig("root", cfg)
    if len(trig_flags) > 1:
        vals = set(trig_flags.values())
        if len(vals) > 1:
            locs = ", ".join(f"{k}={v}" for k, v in trig_flags.items())
            issues.append(Issue(
                severity="WARNING",
                category="CONSISTENCY",
                key="require_trigWeight",
                message=f"Conflicting trigger weight flags detected across sections: {locs}",
                suggestion="Unify trigger weight flags across jcm, fvt_closure, classifier_inputs, and analysis_config"
            ))
        else:
            locs = ", ".join(trig_flags.keys())
            issues.append(Issue(
                severity="INFO",
                category="CONSISTENCY",
                key="require_trigWeight",
                message=f"Parameter 'require_trigWeight' is redundantly defined in {len(trig_flags)} sections: {locs}",
                suggestion="Consider centralizing in a common definition or YAML anchor"
            ))

    ac_cfg = (cfg.get("analysis_config") or {}).get("config") or {}
    blind_flag = ac_cfg.get("blind")
    combine_flags = cfg.get("combine_flags", "")

    if blind_flag is True and combine_flags and "--blind" not in str(combine_flags):
        issues.append(Issue(
            severity="WARNING",
            category="CONSISTENCY",
            key="analysis_config.config.blind",
            message="analysis_config is set to blind: true, but combine_flags lacks '--blind'",
            suggestion="Add '--blind' to combine_flags to keep statistical inference consistent with processor blinding"
        ))


def audit_producer_consumer_paths(cfg: dict, issues: list[Issue]) -> None:
    """Audit that friend tree producers align with friend tree consumers."""
    fvt = cfg.get("fvt") or {}
    svb = cfg.get("svb") or {}
    ac_friends = ((cfg.get("analysis_config") or {}).get("config") or {}).get("friends") or {}

    if fvt and ac_friends:
        fvt_prod_label = fvt.get("label") or ""
        fvt_consumer_path = ac_friends.get("FvT") or ""
        if fvt_prod_label and fvt_consumer_path:
            if "fvt" in fvt_consumer_path.lower() and fvt_prod_label.lower() not in fvt_consumer_path.lower():
                issues.append(Issue(
                    severity="WARNING",
                    category="CONSISTENCY",
                    key="analysis_config.config.friends.FvT",
                    message=f"Producer Phase C label is '{fvt_prod_label}', but Phase F friend path is '{fvt_consumer_path}'",
                    suggestion=f"Verify that consumer friend path points to output of Phase C ({fvt_prod_label})"
                ))

    if svb and ac_friends:
        svb_prod_label = svb.get("label") or ""
        svb_consumer_path = ac_friends.get("SvB_MA") or ac_friends.get("SvB") or ""
        if svb_prod_label and svb_consumer_path:
            if "svb" in svb_consumer_path.lower() and svb_prod_label.lower() not in svb_consumer_path.lower():
                issues.append(Issue(
                    severity="WARNING",
                    category="CONSISTENCY",
                    key="analysis_config.config.friends.SvB",
                    message=f"Producer Phase D label is '{svb_prod_label}', but Phase F friend path is '{svb_consumer_path}'",
                    suggestion=f"Verify that consumer friend path points to output of Phase D ({svb_prod_label})"
                ))


_KNOWN_DATASETS_CACHE: dict[Path, set[str]] = {}


def get_known_datasets(datasets_dir: Path) -> set[str]:
    """Parse and return all known dataset names from metadata catalog YAML files with global caching."""
    if datasets_dir in _KNOWN_DATASETS_CACHE:
        return _KNOWN_DATASETS_CACHE[datasets_dir]

    names: set[str] = set()
    if not datasets_dir.is_dir():
        _KNOWN_DATASETS_CACHE[datasets_dir] = names
        return names

    for f in datasets_dir.glob("*.yml"):
        try:
            content = yaml.safe_load(f.read_text(encoding="utf-8"))
            if isinstance(content, dict):
                names.update(str(k) for k in content.keys())
        except Exception:
            pass
    for f in datasets_dir.glob("*.yaml"):
        try:
            content = yaml.safe_load(f.read_text(encoding="utf-8"))
            if isinstance(content, dict):
                names.update(str(k) for k in content.keys())
        except Exception:
            pass

    _KNOWN_DATASETS_CACHE[datasets_dir] = names
    return names


def audit_filesystem_references(cfg: dict, repo_root: Path, issues: list[Issue]) -> None:
    """Verify that file/dataset references in config exist in repository."""
    datasets_dir = repo_root / "coffea4bees/metadata/datasets"
    if not datasets_dir.is_dir():
        datasets_dir = repo_root / "metadata/datasets"

    known_datasets = get_known_datasets(datasets_dir)

    all_ds = set()
    for k in ("dataset", "datasets"):
        all_ds.update(_extract_dataset_list(cfg.get(k)))
    ci_ds = _extract_dataset_list((cfg.get("classifier_inputs") or {}).get("datasets"))
    all_ds.update(ci_ds)

    if known_datasets:
        for ds in all_ds:
            if ds.lower() in ("data", "mixeddata"):
                continue
            if ds not in known_datasets:
                matching = [k for k in known_datasets if ds.lower() in k.lower()]
                sugg = f"Did you mean '{matching[0]}'?" if matching else "Verify dataset name in metadata/datasets/"
                issues.append(Issue(
                    severity="WARNING",
                    category="REFERENCE",
                    key=f"dataset.{ds}",
                    message=f"Dataset '{ds}' not found in metadata/datasets/",
                    suggestion=sugg
                ))

    # Plot configs check
    plot_cfgs: list[str] = []
    fc = cfg.get("fvt_closure") or {}
    if fc.get("plot_config"):
        plot_cfgs.append(fc["plot_config"])
    ac = (cfg.get("analysis_config") or {}).get("config") or {}
    if ac.get("plot_config"):
        plot_cfgs.append(ac["plot_config"])

    for pc in plot_cfgs:
        p_path = Path(pc)
        if not p_path.is_absolute():
            cand1 = repo_root / pc
            cand2 = repo_root / "coffea4bees" / pc
            if not cand1.is_file() and not cand2.is_file():
                issues.append(Issue(
                    severity="WARNING",
                    category="REFERENCE",
                    key="plot_config",
                    message=f"Referenced plot config '{pc}' does not exist",
                    suggestion="Check path relative to repository root"
                ))

    # Weight files check
    weights = cfg.get("weights")
    if isinstance(weights, str):
        w_path = Path(weights)
        if not w_path.is_absolute():
            cand1 = repo_root / weights
            cand2 = repo_root / "coffea4bees" / weights
            if not cand1.is_file() and not cand2.is_file():
                issues.append(Issue(
                    severity="INFO",
                    category="REFERENCE",
                    key="weights",
                    message=f"Weight file '{weights}' does not exist locally (may be generated during workflow)",
                    suggestion="Verify if weight file is a static input or workflow product"
                ))


# ---------------------------------------------------------------------------
# Validation Orchestration
# ---------------------------------------------------------------------------

def validate_config(config_path: Path | str, repo_root: Path | None = None) -> ValidationReport:
    """Validate a workflow configuration file and return a structured report."""
    p = Path(str(config_path).strip()).expanduser().resolve()
    root = repo_root or find_repo_root(p)
    report = ValidationReport(config_path=p)

    if not p.is_file():
        report.add_issue(Issue(
            severity="ERROR",
            category="SYNTAX",
            key="config",
            message=f"Config file not found: {p}",
        ))
        return report

    merged_cfg, bases, chain_issues = resolve_config_chain(p, root)
    report.bases = bases
    for ci in chain_issues:
        report.add_issue(ci)

    if not merged_cfg:
        return report

    issues: list[Issue] = []
    audit_schema_and_typos(merged_cfg, issues)
    audit_datasets_consistency(merged_cfg, issues)
    audit_year_eras_consistency(merged_cfg, issues)
    audit_flags_and_duplicates(merged_cfg, issues)
    audit_producer_consumer_paths(merged_cfg, issues)
    audit_filesystem_references(merged_cfg, root, issues)

    for i in issues:
        report.add_issue(i)

    return report


# ---------------------------------------------------------------------------
# Reporting & Output Formatting
# ---------------------------------------------------------------------------

def format_report(report: ValidationReport, color: bool = True, verbose: bool = True) -> str:
    """Format a single ValidationReport for terminal display."""
    c_red = "\033[91m" if color else ""
    c_green = "\033[92m" if color else ""
    c_yellow = "\033[93m" if color else ""
    c_cyan = "\033[96m" if color else ""
    c_bold = "\033[1m" if color else ""
    c_reset = "\033[0m" if color else ""

    lines = []
    lines.append(f"{c_bold}Configuration Audit:{c_reset} {c_cyan}{report.config_path}{c_reset}")

    if report.bases:
        bases_str = " -> ".join(b.name for b in report.bases)
        lines.append(f"  {c_bold}Inheritance Base Chain:{c_reset} {bases_str}")

    errors = report.errors
    warnings = report.warnings
    infos = report.infos

    if not report.issues:
        lines.append(f"\n  {c_green}✔ No issues detected. Configuration is clean and consistent.{c_reset}")
    else:
        by_category: dict[str, list[Issue]] = {}
        for issue in report.issues:
            by_category.setdefault(issue.category, []).append(issue)

        cat_titles = {
            "SYNTAX": "Syntax & Inheritance Resolution",
            "SCHEMA": "Structure & Schema Checks",
            "CONSISTENCY": "Cross-Section Parameter Alignment",
            "REFERENCE": "Filesystem & Metadata References",
        }

        for cat, cat_issues in by_category.items():
            title = cat_titles.get(cat, cat)
            lines.append(f"\n  ● {c_bold}{title}:{c_reset}")
            for iss in cat_issues:
                if iss.severity == "ERROR":
                    prefix = f"{c_red}[ERROR]{c_reset}"
                elif iss.severity == "WARNING":
                    prefix = f"{c_yellow}[WARN]{c_reset}"
                else:
                    prefix = f"{c_cyan}[INFO]{c_reset}"

                lines.append(f"    {prefix} {c_bold}{iss.key}{c_reset}: {iss.message}")
                if iss.suggestion:
                    lines.append(f"           ↳ Suggestion: {iss.suggestion}")

    lines.append("\n" + "─" * 70)
    n_err, n_warn, n_info = len(errors), len(warnings), len(infos)
    if n_err > 0:
        verdict = f"{c_red}{c_bold}Status: FAILED{c_reset}  (Errors: {n_err}, Warnings: {n_warn}, Info: {n_info})"
    elif n_warn > 0:
        verdict = f"{c_yellow}{c_bold}Status: WARNINGS{c_reset}  (Errors: 0, Warnings: {n_warn}, Info: {n_info})"
    else:
        verdict = f"{c_green}{c_bold}Status: VALID{c_reset}  (Errors: 0, Warnings: 0, Info: {n_info})"
    lines.append(verdict)

    return "\n".join(lines)


def format_summary_matrix(reports: list[ValidationReport], color: bool = True) -> str:
    """Format multiple ValidationReports into a compact table."""
    use_color = color
    c_red = "\033[91m" if use_color else ""
    c_green = "\033[92m" if use_color else ""
    c_yellow = "\033[93m" if use_color else ""
    c_bold = "\033[1m" if use_color else ""
    c_reset = "\033[0m" if use_color else ""

    lines = []
    lines.append(f"{c_bold}{'Config File':<50} {'Status':<12} {'Errors':<8} {'Warnings':<10}{c_reset}")
    lines.append("─" * 82)

    total_err = 0
    total_warn = 0

    for rep in sorted(reports, key=lambda r: r.config_path.name):
        n_err = len(rep.errors)
        n_warn = len(rep.warnings)
        total_err += n_err
        total_warn += n_warn

        if n_err > 0:
            status = f"{c_red}FAIL{c_reset}"
        elif n_warn > 0:
            status = f"{c_yellow}WARN{c_reset}"
        else:
            status = f"{c_green}PASS{c_reset}"

        name = rep.config_path.name
        if len(name) > 48:
            name = name[:45] + "..."
        lines.append(f"{name:<50} {status:<21} {n_err:<8} {n_warn:<10}")

    lines.append("─" * 82)
    lines.append(f"Total Configs: {len(reports)} | Errors: {total_err} | Warnings: {total_warn}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Python API helpers
# ---------------------------------------------------------------------------

def validate_config_file(config_path: Path | str, repo_root: Path | None = None) -> ValidationReport:
    """Validate a single config file and return the ValidationReport."""
    p = Path(config_path).expanduser().resolve()
    return validate_config(p, repo_root=repo_root)


def validate_all_configs(
    target_dir: Path | str | None = None,
    repo_root: Path | None = None,
    strict: bool = False,
    color: bool = True,
    quiet: bool = False,
) -> int:
    """Validate all YAML configs under target_dir (or default workflows/config) and print report."""
    root = repo_root or find_repo_root()
    if target_dir:
        td = Path(target_dir).expanduser().resolve()
    else:
        td = root / "coffea4bees/workflows/config"

    targets: list[Path] = []
    if td.is_dir():
        targets.extend(sorted(td.glob("*.yml")))
        targets.extend(sorted(td.glob("*.yaml")))
    elif td.is_file():
        targets.append(td)

    if not targets:
        print(f"No configuration files found in {td}")
        return 1

    reports = [validate_config(t, repo_root=root) for t in targets]
    for rep in reports:
        if not rep.is_valid(strict=strict) or not quiet:
            print(format_report(rep, color=color))
            print()
    print(format_summary_matrix(reports, color=color))
    has_failures = any(not r.is_valid(strict=strict) for r in reports)
    return 1 if has_failures else 0


# ---------------------------------------------------------------------------
# CLI Command
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate barista & coffea4bees workflow configurations for cross-section consistency and integrity."
    )
    parser.add_argument("configs", nargs="*", help="Config YAML file(s) to validate")
    parser.add_argument("--ci", action="store_true", help="Validate standard CI workflow configs")
    parser.add_argument("--all", action="store_true", help="Validate all YAML configs under coffea4bees/workflows/config/")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors (fail if any warnings)")
    parser.add_argument("--quiet", action="store_true", help="Only show configs with errors")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    args = parser.parse_args()
    repo_root = find_repo_root()

    targets: list[Path] = []
    if args.all:
        config_dir = repo_root / "coffea4bees/workflows/config"
        if not config_dir.is_dir():
            config_dir = repo_root / "workflows/config"
        if config_dir.is_dir():
            targets.extend(sorted(config_dir.glob("*.yml")))
            targets.extend(sorted(config_dir.glob("*.yaml")))
    elif args.ci or (not args.configs and not args.all):
        # Default CI configs
        for c in DEFAULT_CI_CONFIGS:
            cand1 = repo_root / c
            cand2 = repo_root / c.replace("coffea4bees/", "")
            if cand1.is_file():
                targets.append(cand1)
            elif cand2.is_file():
                targets.append(cand2)
    elif args.configs:
        for c in args.configs:
            c = str(c).strip()
            if not c:
                continue
            p = Path(c).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            targets.append(p)
    else:
        parser.print_help()
        return 1

    if not targets:
        print("No configuration files found to validate.")
        return 1

    reports: list[ValidationReport] = []
    for t in targets:
        reports.append(validate_config(t, repo_root=repo_root))

    if args.json:
        data = []
        for r in reports:
            data.append({
                "config": str(r.config_path),
                "bases": [str(b) for b in r.bases],
                "valid": r.is_valid(strict=args.strict),
                "errors": [str(e) for e in r.errors],
                "warnings": [str(w) for w in r.warnings],
                "infos": [str(i) for i in r.infos],
            })
        print(json.dumps(data, indent=2))
    elif len(reports) == 1:
        rep = reports[0]
        if not args.quiet or not rep.is_valid(strict=args.strict):
            print(format_report(rep, color=not args.no_color))
    else:
        for rep in reports:
            if not rep.is_valid(strict=args.strict) or not args.quiet:
                print(format_report(rep, color=not args.no_color))
                print()
        print(format_summary_matrix(reports, color=not args.no_color))

    has_failures = any(not r.is_valid(strict=args.strict) for r in reports)
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
