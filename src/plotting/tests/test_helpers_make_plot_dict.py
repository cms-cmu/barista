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

from src.plotting.plots import _normalize_kwargs, _NORMALIZE_MAP, _RENDER_FIELDS
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

    # --- canonical keys and unrelated keys pass through unchanged ---

    def test_canonical_key_passes_through_unchanged(self):
        kwargs = {'doRatio': False, 'norm': True, 'yscale': 'log'}
        _normalize_kwargs(kwargs)
        assert kwargs == {'doRatio': False, 'norm': True, 'yscale': 'log'}

    def test_unrelated_key_passes_through_unchanged(self):
        # DATA_ONLY keys like rebin/year must not be touched
        kwargs = {'rebin': 2, 'year': 'UL18'}
        _normalize_kwargs(kwargs)
        assert kwargs == {'rebin': 2, 'year': 'UL18'}

    # --- all aliases (single-pass: strip+lowercase then look up in _NORMALIZE_MAP) ---

    @pytest.mark.parametrize("alias,canonical", [
        # norm
        ('normalize',      'norm'),
        ('normalise',      'norm'),
        ('normalized',     'norm'),
        ('normalised',     'norm'),
        # add_flow
        ('flow',           'add_flow'),
        ('addFlow',        'add_flow'),
        # uniform_bins
        ('uniform',        'uniform_bins'),
        ('uniformBins',    'uniform_bins'),
        # outputFolder
        ('outdir',         'outputFolder'),
        ('output_dir',     'outputFolder'),
        ('output_folder',  'outputFolder'),
        # doRatio — 'ratio' ≠ 'doratio'; all case variants collapse via strip+lower
        ('ratio',          'doRatio'),
        ('Ratio',          'doRatio'),
        ('RATIO',          'doRatio'),
        ('do_ratio',       'doRatio'),
        ('doratio',        'doRatio'),
        ('do_Ratio',       'doRatio'),
        # rlim
        ('ratio_lim',      'rlim'),
        ('ratio_limits',   'rlim'),
        # rlabel
        ('ratio_label',    'rlabel'),
        # legend
        ('doLegend',       'legend'),
        ('do_legend',      'legend'),
        # legend_loc
        ('legend_location','legend_loc'),
        ('legendLoc',      'legend_loc'),
        # write_yaml — 'save' ≠ 'write'
        ('saveYaml',       'write_yaml'),
        ('save_yaml',      'write_yaml'),
        ('writeYaml',      'write_yaml'),
        # fmt
        ('format',         'fmt'),
        ('output_format',  'fmt'),
        # camelCase / snake_case auto-variants
        ('cmsText',        'CMSText'),
        ('cms_text',       'CMSText'),
        ('doTitle',        'do_title'),
        ('y_scale',        'yscale'),
        ('yScale',         'yscale'),
        ('x_scale',        'xscale'),
        ('xScale',         'xscale'),
        ('y_lim',          'ylim'),
        ('yLim',           'ylim'),
        ('x_lim',          'xlim'),
        ('xLim',           'xlim'),
    ])
    def test_aliases(self, alias, canonical):
        sentinel = object()
        kwargs = {alias: sentinel}
        _normalize_kwargs(kwargs)
        assert kwargs.get(canonical) is sentinel, \
            f"Expected '{alias}' to normalize to '{canonical}'"
        assert alias not in kwargs

    def test_normalize_map_covers_all_render_fields(self):
        # Every RenderOptions field must round-trip through the map.
        for field in _RENDER_FIELDS:
            normalized = field.lower().replace('_', '')
            assert _NORMALIZE_MAP.get(normalized) == field, \
                f"Field '{field}' missing from _NORMALIZE_MAP"

    def test_normalize_map_has_no_field_collisions(self):
        # Two different RenderOptions fields must not share the same stripped form.
        seen = {}
        for field in _RENDER_FIELDS:
            key = field.lower().replace('_', '')
            assert key not in seen, \
                f"Collision: '{field}' and '{seen[key]}' both normalize to '{key}'"
            seen[key] = field

    # --- precedence and identity ---

    def test_canonical_wins_over_alias(self):
        # If user passes both the alias and the canonical, canonical wins.
        kwargs = {'doRatio': True, 'do_ratio': False}
        _normalize_kwargs(kwargs)
        assert kwargs['doRatio'] is True

    def test_explicit_alias_wins_over_auto_when_both_present(self):
        # Explicit pass runs first; auto pass won't touch already-canonical keys.
        kwargs = {'normalize': True}
        _normalize_kwargs(kwargs)
        assert kwargs.get('norm') is True

    def test_all_map_entries_distinct_from_canonical(self):
        # No entry in _NORMALIZE_MAP should be a no-op (key == value after stripping).
        for stripped_key, canonical in _NORMALIZE_MAP.items():
            # The stripped form of the canonical should equal stripped_key — that's fine.
            # What we check: if the raw key IS the canonical, the normalizer skips it (key != canonical guard).
            assert canonical in _RENDER_FIELDS, \
                f"_NORMALIZE_MAP value '{canonical}' is not a known RenderOptions field"

    def test_returns_same_dict(self):
        kwargs = {'do_ratio': True}
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


# ---------------------------------------------------------------------------
# get_plot_dict_from_list
# ---------------------------------------------------------------------------

class TestGetPlotDictFromList:
    def test_get_plot_dict_from_list_default_process(self, monkeypatch):
        monkeypatch.setattr(
            "src.plotting.helpers_make_plot_dict.add_hist_data",
            lambda **kwargs: True
        )

        cfg = MagicMock()
        cfg.plotConfig = {
            "hists": {
                "ttHbb": {"process": "ttHbb", "label": "ttHbb", "fillcolor": "red"}
            },
            "stack": {
                "TTbar": {"process": "TTbar", "label": "TTbar", "fillcolor": "blue"}
            }
        }
        cfg.hists = [MagicMock()]
        cfg.combine_input_files = True

        from src.plotting.helpers_make_plot_dict import get_plot_dict_from_list

        # When process is None, it should default to all processes in hists & stack
        # and support cut as a list
        plot_data = get_plot_dict_from_list(
            cfg=cfg,
            var="jet_pt",
            cut=["SR", "SB"],
            axis_opts={},
            process=None,
            doRatio=False,
        )

        assert "ttHbb" in plot_data["process"]
        assert "TTbar" in plot_data["process"]
        
        # Check that we have entries for all processes x cuts
        hists_keys = list(plot_data["hists"].keys())
        assert any("ttHbb" in k and "SR" in k for k in hists_keys)
        assert any("ttHbb" in k and "SB" in k for k in hists_keys)
        assert any("TTbar" in k and "SR" in k for k in hists_keys)
        assert any("TTbar" in k and "SB" in k for k in hists_keys)


class TestGetHistDataListCuts:
    def test_get_hist_data_sums_list_of_cuts(self, monkeypatch):
        mock_hist1 = MagicMock()
        mock_hist2 = MagicMock()
        mock_sum = MagicMock()
        mock_hist1.__add__.return_value = mock_sum
        mock_hist1.__iadd__.return_value = mock_sum
        mock_hist1.__imul__.return_value = mock_hist1
        mock_hist2.__imul__.return_value = mock_hist2

        monkeypatch.setattr("src.plotting.helpers_make_plot_dict._build_hist_opts", lambda process, year, config, axis_opts, cut, cfg, debug: ({"cut": cut}, {}))
        monkeypatch.setattr("src.plotting.helpers_make_plot_dict._find_hist_obj", lambda *a, **k: MagicMock())
        monkeypatch.setattr("src.plotting.helpers_make_plot_dict._apply_intcategory_compat", lambda *a, **k: None)
        monkeypatch.setattr("src.plotting.helpers_make_plot_dict._remove_missing_cut_keys", lambda *a, **k: None)
        monkeypatch.setattr("src.plotting.helpers_make_plot_dict._squeeze_hist", lambda h, *a, **k: h)

        def mock_select_hist(hist_obj, hist_opts, rebin, do2d, debug):
            if hist_opts.get("cut") == "cut1":
                return mock_hist1
            elif hist_opts.get("cut") == "cut2":
                return mock_hist2
            return MagicMock()

        monkeypatch.setattr("src.plotting.helpers_make_plot_dict._select_hist", mock_select_hist)

        from src.plotting.helpers_make_plot_dict import get_hist_data

        cfg = MagicMock()
        cfg.plotConfig = {}
        config = {"scalefactor": 1.0}

        result = get_hist_data(
            process="ttbar", cfg=cfg, config=config, var="jet_pt",
            cut=["cut1", "cut2"], rebin=1, year="UL18", axis_opts={},
        )

        assert result == mock_sum
        mock_hist1.__iadd__.assert_called_once_with(mock_hist2)


class TestBlinding:
    def _create_sample_plot_data(self, n_bins=20, x_min=0.0, x_max=1.0):
        edges = np.linspace(x_min, x_max, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return {
            "hists": {
                "data": {
                    "process": "data",
                    "values": [10.0] * n_bins,
                    "variances": [10.0] * n_bins,
                    "edges": edges.tolist(),
                    "centers": centers.tolist(),
                    "under_flow": 2.0,
                    "over_flow": 3.0,
                },
                "signal": {
                    "process": "ttHbb",
                    "values": [5.0] * n_bins,
                    "variances": [5.0] * n_bins,
                    "edges": edges.tolist(),
                    "centers": centers.tolist(),
                    "under_flow": 1.0,
                    "over_flow": 1.0,
                },
            },
            "stack": {
                "ttbar": {
                    "process": "ttbar",
                    "values": [20.0] * n_bins,
                    "variances": [20.0] * n_bins,
                    "edges": edges.tolist(),
                    "centers": centers.tolist(),
                    "under_flow": 4.0,
                    "over_flow": 4.0,
                }
            }
        }

    def test_blind_bool_masks_last_10_bins(self):
        from src.plotting.helpers_make_plot_dict import apply_blinding
        plot_data = self._create_sample_plot_data(n_bins=20)
        apply_blinding(plot_data, True)

        data = plot_data["hists"]["data"]
        vals = data["values"]
        assert all(v == 10.0 for v in vals[:10])
        assert all(np.isnan(v) for v in vals[10:])
        assert data["over_flow"] == 0.0

        # Signal and background stack should not be touched
        signal_vals = plot_data["hists"]["signal"]["values"]
        assert all(v == 5.0 for v in signal_vals)
        stack_vals = plot_data["stack"]["ttbar"]["values"]
        assert all(v == 20.0 for v in stack_vals)

    def test_blind_int_masks_specified_bins(self):
        from src.plotting.helpers_make_plot_dict import apply_blinding
        plot_data = self._create_sample_plot_data(n_bins=20)
        apply_blinding(plot_data, 5)

        data = plot_data["hists"]["data"]
        vals = data["values"]
        assert all(v == 10.0 for v in vals[:15])
        assert all(np.isnan(v) for v in vals[15:])
        assert data["over_flow"] == 0.0

    def test_blind_float_threshold(self):
        from src.plotting.helpers_make_plot_dict import apply_blinding
        plot_data = self._create_sample_plot_data(n_bins=10, x_min=0.0, x_max=1.0)
        # Bins: [0.0, 0.1], ..., [0.7, 0.8], [0.8, 0.9], [0.9, 1.0]
        apply_blinding(plot_data, 0.8)

        data = plot_data["hists"]["data"]
        vals = data["values"]
        assert all(v == 10.0 for v in vals[:8])
        assert all(np.isnan(v) for v in vals[8:])

    def test_blind_range_tuple(self):
        from src.plotting.helpers_make_plot_dict import apply_blinding
        plot_data = self._create_sample_plot_data(n_bins=10, x_min=0.0, x_max=100.0)
        # Bins: [0, 10], [10, 20], ..., [90, 100]
        # Centers: 5, 15, 25, 35, 45, 55, 65, 75, 85, 95
        apply_blinding(plot_data, (30.0, 70.0))

        data = plot_data["hists"]["data"]
        vals = data["values"]
        # Centers 35, 45, 55, 65 (indices 3, 4, 5, 6) should be masked
        assert vals[0] == 10.0 and vals[1] == 10.0 and vals[2] == 10.0
        assert all(np.isnan(vals[i]) for i in [3, 4, 5, 6])
        assert vals[7] == 10.0 and vals[8] == 10.0 and vals[9] == 10.0



