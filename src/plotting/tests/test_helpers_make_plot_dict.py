"""Unit tests for plotting pipeline pure functions.

Covers:
  - _normalize_kwargs : kwarg alias folding (plots.py)
  - _normalize_year   : multi-year alias → sum sentinel
  - _build_hist_opts  : assembles the hist indexing dict
  - _entries_overlay  : shared loop core for all overlay builders
  - _squeeze_hist     : collapses extra leading dimensions
"""

import numpy as np
import pytest
import hist as Hist
from unittest.mock import MagicMock

from src.plotting.plots import _normalize_kwargs, _KWARG_ALIASES
from src.plotting.helpers_make_plot_dict import (
    _normalize_year,
    _build_hist_opts,
    _entries_overlay,
    _squeeze_hist,
)


# ---------------------------------------------------------------------------
# _normalize_kwargs
# ---------------------------------------------------------------------------

class TestNormalizeKwargs:
    def test_canonical_key_passes_through_unchanged(self):
        kwargs = {'doRatio': False, 'norm': True, 'yscale': 'log'}
        _normalize_kwargs(kwargs)
        assert kwargs == {'doRatio': False, 'norm': True, 'yscale': 'log'}

    def test_unrelated_key_passes_through_unchanged(self):
        kwargs = {'rebin': 2, 'year': 'UL18'}
        _normalize_kwargs(kwargs)
        assert kwargs == {'rebin': 2, 'year': 'UL18'}

    @pytest.mark.parametrize("alias", ['ratio', 'do_ratio', 'doratio', 'Ratio', 'do_Ratio'])
    def test_doRatio_aliases(self, alias):
        kwargs = {alias: False}
        _normalize_kwargs(kwargs)
        assert 'doRatio' in kwargs
        assert kwargs['doRatio'] is False
        assert alias not in kwargs

    @pytest.mark.parametrize("alias", ['normalize', 'normalise', 'normalized', 'normalised'])
    def test_norm_aliases(self, alias):
        kwargs = {alias: True}
        _normalize_kwargs(kwargs)
        assert kwargs.get('norm') is True
        assert alias not in kwargs

    @pytest.mark.parametrize("alias", ['addFlow', 'addflow', 'flow'])
    def test_add_flow_aliases(self, alias):
        kwargs = {alias: True}
        _normalize_kwargs(kwargs)
        assert kwargs.get('add_flow') is True
        assert alias not in kwargs

    @pytest.mark.parametrize("alias", ['uniformBins', 'uniformbins', 'uniform'])
    def test_uniform_bins_aliases(self, alias):
        kwargs = {alias: True}
        _normalize_kwargs(kwargs)
        assert kwargs.get('uniform_bins') is True
        assert alias not in kwargs

    @pytest.mark.parametrize("alias,canonical", [
        ('y_scale',       'yscale'),
        ('yScale',        'yscale'),
        ('x_scale',       'xscale'),
        ('xScale',        'xscale'),
        ('y_lim',         'ylim'),
        ('yLim',          'ylim'),
        ('x_lim',         'xlim'),
        ('xLim',          'xlim'),
        ('ratio_lim',     'rlim'),
        ('ratioLim',      'rlim'),
        ('ratio_label',   'rlabel'),
        ('ratioLabel',    'rlabel'),
        ('doLegend',      'legend'),
        ('do_legend',     'legend'),
        ('legendLoc',     'legend_loc'),
        ('legend_location','legend_loc'),
        ('doTitle',       'do_title'),
        ('cmsText',       'CMSText'),
        ('cms_text',      'CMSText'),
        ('output_folder', 'outputFolder'),
        ('outdir',        'outputFolder'),
        ('writeYaml',     'write_yaml'),
        ('saveYaml',      'write_yaml'),
        ('format',        'fmt'),
    ])
    def test_individual_aliases(self, alias, canonical):
        sentinel = object()
        kwargs = {alias: sentinel}
        _normalize_kwargs(kwargs)
        assert kwargs.get(canonical) is sentinel
        assert alias not in kwargs

    def test_alias_does_not_overwrite_existing_canonical(self):
        # If user passes both the alias and the canonical, canonical wins.
        kwargs = {'doRatio': True, 'ratio': False}
        _normalize_kwargs(kwargs)
        assert kwargs['doRatio'] is True  # setdefault preserves the original

    def test_all_aliases_map_to_known_canonical(self):
        # Every alias in _KWARG_ALIASES should be distinct from its canonical.
        for alias, canonical in _KWARG_ALIASES.items():
            assert alias != canonical, f"Alias '{alias}' is the same as its canonical"

    def test_returns_same_dict(self):
        kwargs = {'ratio': True}
        result = _normalize_kwargs(kwargs)
        assert result is kwargs


# ---------------------------------------------------------------------------
# _normalize_year
# ---------------------------------------------------------------------------

class TestNormalizeYear:
    @pytest.mark.parametrize("alias", ["RunII", "Run2", "Run3", "RunIII"])
    def test_aliases_become_sum(self, alias):
        assert _normalize_year(alias) is sum

    @pytest.mark.parametrize("yr", ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"])
    def test_specific_years_pass_through(self, yr):
        assert _normalize_year(yr) == yr


# ---------------------------------------------------------------------------
# _build_hist_opts
# ---------------------------------------------------------------------------

class TestBuildHistOpts:
    def _cfg(self, cut_list=None):
        cfg = MagicMock()
        cfg.cutList = cut_list or ["SR", "SB", "VR"]
        return cfg

    def test_process_and_year_always_present(self):
        opts, _ = _build_hist_opts("ttbar", "UL18", {"process": "ttbar"}, {}, None, self._cfg(), False)
        assert opts["process"] == "ttbar"
        assert opts["year"] == "UL18"

    def test_style_keys_excluded(self):
        style_keys = ["process", "scalefactor", "label", "fillcolor", "edgecolor",
                      "histtype", "alpha", "linewidth", "linestyle", "zorder", "year"]
        config = {k: "dummy" for k in style_keys}
        opts, _ = _build_hist_opts("sig", "UL18", config, {}, None, self._cfg(), False)
        for k in ["scalefactor", "label", "fillcolor", "edgecolor",
                  "histtype", "alpha", "linewidth", "linestyle", "zorder"]:
            assert k not in opts, f"Style key '{k}' should not appear in hist_opts"

    def test_non_style_config_key_included(self):
        config = {"process": "data", "tag": "3b"}
        opts, _ = _build_hist_opts("data", "UL18", config, {}, None, self._cfg(), False)
        assert opts["tag"] == "3b"

    def test_axis_opts_merged(self):
        config = {"process": "ttbar"}
        opts, _ = _build_hist_opts("ttbar", "UL18", config, {"region": "SR"}, None, self._cfg(), False)
        assert opts["region"] == "SR"

    def test_axis_opts_override_config(self):
        config = {"process": "data", "region": "SR"}
        opts, _ = _build_hist_opts("data", "UL18", config, {"region": "SB"}, None, self._cfg(), False)
        assert opts["region"] == "SB"

    def test_no_cut_returns_empty_cut_dict(self):
        _, cut_dict = _build_hist_opts("ttbar", "UL18", {"process": "ttbar"}, {}, None, self._cfg(), False)
        assert cut_dict == {}

    def test_cut_added_to_opts_and_returned(self):
        opts, cut_dict = _build_hist_opts("ttbar", "UL18", {"process": "ttbar"}, {}, "SR", self._cfg(), False)
        assert "SR" in cut_dict
        assert cut_dict["SR"] is True
        # Cut dict keys should appear in opts
        for k in cut_dict:
            assert k in opts

    def test_inverted_cut(self):
        _, cut_dict = _build_hist_opts("ttbar", "UL18", {"process": "ttbar"}, {}, "~SR", self._cfg(), False)
        assert cut_dict.get("SR") is False


# ---------------------------------------------------------------------------
# _entries_overlay
# ---------------------------------------------------------------------------

class TestEntriesOverlay:
    def _config(self):
        return {"process": "ttbar", "label": "TTbar", "fillcolor": "blue",
                "edgecolor": None, "histtype": "fill"}

    # --- basic count and types ---

    def test_cut_list_produces_correct_count(self):
        entries = _entries_overlay(
            process_config=self._config(), items=["SR", "SB", "VR"],
            base_var="jet_pt", base_cut=None, base_year="UL18", base_axis_opts={},
            item_is_cut=True,
        )
        assert len(entries) == 3

    def test_returns_list_of_load_specs(self):
        from src.plotting.helpers_make_plot_dict import LoadSpec
        entries = _entries_overlay(
            process_config=self._config(), items=["SR"],
            base_var="jet_pt", base_cut=None, base_year="UL18", base_axis_opts={},
            item_is_cut=True,
        )
        assert isinstance(entries[0], LoadSpec)

    # --- item_is_cut ---

    def test_item_is_cut_sets_cut(self):
        entries = _entries_overlay(
            process_config=self._config(), items=["SR", "SB"],
            base_var="jet_pt", base_cut=None, base_year="UL18", base_axis_opts={},
            item_is_cut=True,
        )
        assert entries[0].cut == "SR"
        assert entries[1].cut == "SB"
        assert entries[0].var == "jet_pt"   # base_var unchanged

    # --- item_is_var ---

    def test_item_is_var_sets_var(self):
        entries = _entries_overlay(
            process_config=self._config(), items=["jet_pt", "jet_eta"],
            base_var=None, base_cut="SR", base_year="UL18", base_axis_opts={},
            item_is_var=True, set_edge_color=True,
        )
        assert entries[0].var == "jet_pt"
        assert entries[1].var == "jet_eta"
        assert entries[0].cut == "SR"       # base_cut unchanged

    # --- item_is_year ---

    def test_item_is_year_sets_year(self):
        entries = _entries_overlay(
            process_config=self._config(), items=["UL16_preVFP", "UL17", "UL18"],
            base_var="jet_pt", base_cut="SR", base_year=None, base_axis_opts={},
            item_is_year=True, set_edge_color=True,
        )
        assert entries[0].year == "UL16_preVFP"
        assert entries[2].year == "UL18"
        assert entries[0].var == "jet_pt"   # base_var unchanged

    # --- axis_opt_key ---

    def test_axis_opt_key_sets_per_item_value(self):
        base_opts = {"region": "SR"}
        entries = _entries_overlay(
            process_config=self._config(), items=["3b", "4b"],
            base_var="jet_pt", base_cut="SR", base_year="UL18", base_axis_opts=base_opts,
            axis_opt_key="tag",
        )
        assert entries[0].axis_opts["tag"] == "3b"
        assert entries[1].axis_opts["tag"] == "4b"

    def test_axis_opt_key_does_not_mutate_base(self):
        base_opts = {"region": "SR"}
        _entries_overlay(
            process_config=self._config(), items=["3b"],
            base_var="jet_pt", base_cut="SR", base_year="UL18", base_axis_opts=base_opts,
            axis_opt_key="tag",
        )
        assert "tag" not in base_opts

    # --- keys ---

    def test_keys_are_unique(self):
        entries = _entries_overlay(
            process_config=self._config(), items=["SR", "SB", "VR"],
            base_var="jet_pt", base_cut=None, base_year="UL18", base_axis_opts={},
            item_is_cut=True,
        )
        keys = [e.key for e in entries]
        assert len(keys) == len(set(keys))

    # --- hist_key_list ---

    def test_hist_key_list_assigned(self):
        entries = _entries_overlay(
            process_config=self._config(), items=["SR", "SB"],
            base_var="jet_pt", base_cut=None, base_year="UL18", base_axis_opts={},
            item_is_cut=True, hist_key_list=["hists_a", "hists_b"],
        )
        assert entries[0].hist_key_override == "hists_a"
        assert entries[1].hist_key_override == "hists_b"

    def test_no_hist_key_list_gives_none(self):
        entries = _entries_overlay(
            process_config=self._config(), items=["SR"],
            base_var="jet_pt", base_cut=None, base_year="UL18", base_axis_opts={},
            item_is_cut=True,
        )
        assert entries[0].hist_key_override is None

    # --- label_override ---

    def test_label_override_applied(self):
        entries = _entries_overlay(
            process_config=self._config(), items=["SR", "SB"],
            base_var="jet_pt", base_cut=None, base_year="UL18", base_axis_opts={},
            item_is_cut=True, label_override=["Signal Region", "Sideband"],
        )
        assert entries[0].config["label"] == "Signal Region"
        assert entries[1].config["label"] == "Sideband"

    # --- config isolation ---

    def test_configs_are_independent_copies(self):
        entries = _entries_overlay(
            process_config=self._config(), items=["SR", "SB"],
            base_var="jet_pt", base_cut=None, base_year="UL18", base_axis_opts={},
            item_is_cut=True,
        )
        entries[0].config["label"] = "MODIFIED"
        assert entries[1].config["label"] != "MODIFIED"

    def test_original_process_config_not_mutated(self):
        config = self._config()
        original_label = config["label"]
        _entries_overlay(
            process_config=config, items=["SR", "SB"],
            base_var="jet_pt", base_cut=None, base_year="UL18", base_axis_opts={},
            item_is_cut=True,
        )
        assert config["label"] == original_label

    # --- file_index always None ---

    def test_file_index_is_none(self):
        entries = _entries_overlay(
            process_config=self._config(), items=["SR"],
            base_var="jet_pt", base_cut=None, base_year="UL18", base_axis_opts={},
            item_is_cut=True,
        )
        assert entries[0].file_index is None


# ---------------------------------------------------------------------------
# _squeeze_hist
# ---------------------------------------------------------------------------

class TestSqueezeHist:
    def _hist_1d(self, n=10):
        h = Hist.Hist(Hist.axis.Regular(n, 0, 100, name="x"))
        h.fill(x=np.linspace(0, 99, 50))
        return h

    def _hist_2d(self):
        h = Hist.Hist(
            Hist.axis.Regular(3, 0, 3, name="extra"),
            Hist.axis.Regular(10, 0, 100, name="x"),
        )
        h.fill(extra=np.ones(50), x=np.linspace(0, 99, 50))
        return h

    def _hist_3d(self):
        h = Hist.Hist(
            Hist.axis.Regular(3, 0, 3, name="extra"),
            Hist.axis.Regular(5, 0, 5, name="x"),
            Hist.axis.Regular(5, 0, 5, name="y"),
        )
        h.fill(extra=np.ones(50), x=np.linspace(0, 4, 50), y=np.linspace(0, 4, 50))
        return h

    def test_1d_passthrough_when_do2d_false(self):
        h = self._hist_1d()
        result = _squeeze_hist(h, do2d=False)
        assert len(result.shape) == 1

    def test_2d_squeezed_to_1d_when_do2d_false(self):
        h = self._hist_2d()
        assert len(h.shape) == 2
        result = _squeeze_hist(h, do2d=False)
        assert len(result.shape) == 1

    def test_2d_passthrough_when_do2d_true(self):
        h = self._hist_2d()
        result = _squeeze_hist(h, do2d=True)
        assert len(result.shape) == 2

    def test_3d_squeezed_to_2d_when_do2d_true(self):
        h = self._hist_3d()
        assert len(h.shape) == 3
        result = _squeeze_hist(h, do2d=True)
        assert len(result.shape) == 2

    def test_squeeze_preserves_bin_count(self):
        h = self._hist_2d()          # shape (3, 10)
        result = _squeeze_hist(h, do2d=False)
        assert result.shape[0] == 10  # x axis preserved
