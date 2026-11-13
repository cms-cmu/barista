import os
import glob


def make_poi_maps(signals, poi_ranges="1,-10,10"):
    """
    Generate the --PO map strings for all signals.
    - signals: can be:
        1. A dictionary: {signal_name: range_string, ...} (for custom per-signal ranges)
        2. A list: [signal_name, ...] (uses `poi_ranges` for all)
        3. A string: space-separated list of signals (uses `poi_ranges` for all)
    - poi_ranges: default range string to use for signals without explicit ranges.
                  Can also be a list matching the signal list.
    """
    if isinstance(signals, dict):
        return " ".join([f"--PO 'map=.*/{sig}:r{sig}[{r}]'" for sig, r in signals.items()])
        
    if isinstance(signals, str):
        signals = signals.split()
        
    if isinstance(poi_ranges, str) and poi_ranges.count(",") == 2:
        ranges_list = [poi_ranges] * len(signals)
    elif isinstance(poi_ranges, list):
        ranges_list = poi_ranges + [poi_ranges[-1]] * (len(signals) - len(poi_ranges))
    else:
        ranges_list = [str(poi_ranges)] * len(signals)
        
    return " ".join([f"--PO 'map=.*/{sig}:r{sig}[{r}]'" for sig, r in zip(signals, ranges_list)])


def get_default_othersignals(wildcards, config):
    """
    Get the default othersignal list from the config channels based on wildcards.
    """
    if not config or "channels" not in config:
        return []
    
    # 1. Try to find the channel name by searching segments of wildcards.path
    if hasattr(wildcards, "path"):
        parts = wildcards.path.split("/")
        for channel, chan_cfg in config["channels"].items():
            if channel in parts:
                others = chan_cfg.get("othersignal", "")
                if isinstance(others, list):
                    return others
                return others.split()
                
    # 2. Fallback: find any channel in config that has this signallabel
    if hasattr(wildcards, "signallabel"):
        for channel, chan_cfg in config["channels"].items():
            if chan_cfg.get("signallabel") == wildcards.signallabel:
                others = chan_cfg.get("othersignal", "")
                if isinstance(others, list):
                    return others
                return others.split()
                
    return []


def is_channel_blinded(wildcards, config):
    """
    Check if the channel or workflow is configured to run blinded.
    """
    if not config:
        return False
    if hasattr(wildcards, "path"):
        if "unblinded" in wildcards.path or "mixeddata" in wildcards.path:
            return False
        channel = wildcards.path.rstrip("/").split("/")[-1]
        ch_cfg = config.get("channels", {}).get(channel, {})
        if ch_cfg.get("blind") is False:
            return False
        flags = ch_cfg.get("combine_flags", config.get("combine_flags", ""))
    else:
        flags = config.get("combine_flags", "")
    if "--blind" in str(flags):
        return True
    if config.get("blind") is False:
        return False
    if config.get("blind", False) or config.get("analysis_config", {}).get("config", {}).get("blind", False):
        return True
    return False


def get_grid_split_points(wildcards, config):
    """
    Get the (firstPoint, lastPoint) tuple for a given split chunk.
    """
    points = int(config.get("likelihood_scan_points", 50))
    split_size = int(config.get("likelihood_scan_split_size", 10))
    split_idx = int(wildcards.split_index)
    first = split_idx * split_size
    last = min(first + split_size - 1, points - 1)
    return first, last


def get_channel_groups(wildcards, config=None):
    """
    Find available nuisance groups for this channel/datacard by parsing the datacard text file on disk,
    or looking up config.
    Returns a dict mapping group_name -> list of nuisance parameter names.
    """
    groups = {}
    path = getattr(wildcards, "path", None)
    signallabel = getattr(wildcards, "signallabel", "")
    channel = path.rstrip("/").split("/")[-1] if path else ""

    # Potential datacard locations
    dc_candidates = []
    if path:
        dc_candidates.extend([
            f"{path}/workspace/datacard__{signallabel}.txt",
            f"{path}/datacards/datacard__{signallabel}.txt",
            f"{path}/workspace/datacard__{channel}.txt",
            f"{path}/datacards/datacard__{channel}.txt",
        ])
        dc_candidates.extend(glob.glob(f"{path}/datacards/datacard_*.txt"))
        dc_candidates.extend(glob.glob(f"{path}/workspace/datacard_*.txt"))

    for dc in dc_candidates:
        if os.path.exists(dc) and os.path.isfile(dc):
            try:
                with open(dc) as f:
                    for line in f:
                        line = line.strip()
                        if " group =" in line or " group=" in line:
                            parts = line.split("group", 1)
                            grp_name = parts[0].strip()
                            systs = parts[1].replace("=", "").strip().split()
                            if grp_name and grp_name not in groups:
                                groups[grp_name] = systs
                if groups:
                    break
            except Exception:
                pass

    return groups


def has_channel_signal_systematics(wildcards, config=None):
    """
    Check whether signal systematics are enabled and present for this channel.
    """
    if config:
        # Check global stat_only mode
        val = config.get("stat_only", None)
        if val is True or str(val).lower() in ["true", "1", "--stat_only"]:
            return False
        make_combine_stat = config.get("make_combine_inputs", {}).get("stat_only", "")
        if make_combine_stat is True or str(make_combine_stat).lower() in ["true", "1", "--stat_only"]:
            return False

        # Channel-specific stat_only
        path = getattr(wildcards, "path", None)
        channel = path.rstrip("/").split("/")[-1] if path else ""
        ch_cfg = config.get("channels", {}).get(channel, {})
        if isinstance(ch_cfg, dict) and ch_cfg.get("stat_only") is True:
            return False

        # If syst_groups or mc_systematics is configured in config, signal systematics are present
        if config.get("syst_groups") or config.get("mc_systematics") or ch_cfg.get("syst_groups"):
            return True

    # Check parsed groups from datacard
    groups = get_channel_groups(wildcards, config)
    sig_group_names = {"signal", "btag", "others", "ps_fsr", "mtop", "signal_norm_xsbr", "signal_norm_xs"}
    if any(g in groups for g in sig_group_names):
        return True

    return False


def get_likelihood_scan_fit_types(wildcards, config=None):
    """
    Determine the fit types for the likelihood scan.
    If full systematics are present (both background and signal systematics):
      returns ['exp', 'exp_bkg', 'exp_sig', 'exp_stat'] (blinded)
    If only background systematics are present:
      returns ['exp', 'exp_stat'] (blinded)
    If stat only:
      returns ['exp'] (blinded)
    """
    blinded = is_channel_blinded(wildcards, config)
    
    path = getattr(wildcards, "path", None)
    channel = path.rstrip("/").split("/")[-1] if path else ""
    ch_cfg = config.get("channels", {}).get(channel, {}) if config else {}
    is_stat_only = False
    if config:
        val = config.get("stat_only", None)
        if val is True or str(val).lower() in ["true", "1", "--stat_only"]:
            is_stat_only = True
        make_combine_stat = config.get("make_combine_inputs", {}).get("stat_only", "")
        if make_combine_stat is True or str(make_combine_stat).lower() in ["true", "1", "--stat_only"]:
            is_stat_only = True
    if isinstance(ch_cfg, dict) and ch_cfg.get("stat_only") is True:
        is_stat_only = True

    if is_stat_only:
        return ["exp"] if blinded else ["obs", "exp"]

    has_sig_systs = has_channel_signal_systematics(wildcards, config)
    groups = get_channel_groups(wildcards, config)
    has_bkg_systs = ("background" in groups) or ("multijet" in groups) or (not is_stat_only)

    if has_sig_systs and has_bkg_systs:
        exp_types = ["exp", "exp_bkg", "exp_sig", "exp_stat"]
        obs_types = ["obs", "obs_bkg", "obs_sig", "obs_stat"]
    elif has_bkg_systs:
        exp_types = ["exp", "exp_stat"]
        obs_types = ["obs", "obs_stat"]
    else:
        exp_types = ["exp"]
        obs_types = ["obs"]

    return exp_types if blinded else (obs_types + exp_types)


def get_likelihood_scan_freeze_groups(wildcards, config=None):
    """
    Return the nuisance group(s) to freeze for a given scan fit_type.
    - exp_bkg / obs_bkg: freeze signal systematics -> returns 'signal' (or existing signal groups)
    - exp_sig / obs_sig: freeze background systematics -> returns 'background' or 'multijet'
    - other fit_types: returns empty string
    """
    ft = getattr(wildcards, "fit_type", "")
    groups = get_channel_groups(wildcards, config)

    if ft in ["exp_bkg", "obs_bkg"]:
        # Freeze signal systematics to leave only background systematics (+ stat)
        if "signal" in groups:
            return "signal"
        # Fall back to listing existing individual signal groups
        sig_candidates = ["btag", "others", "ps_fsr", "mtop", "signal_norm_xsbr", "signal_norm_xs"]
        matched = [g for g in sig_candidates if g in groups]
        if matched:
            return ",".join(matched)
        return "signal"

    if ft in ["exp_sig", "obs_sig"]:
        # Freeze background systematics to leave only signal systematics (+ stat)
        if "background" in groups:
            return "background"
        if "multijet" in groups:
            return "multijet"
        return "background"

    return ""


def get_likelihood_scan_freeze_params(wildcards, config=None):
    """
    Return the list of parameters to freeze for a given scan fit_type.
    - exp_stat / obs_stat: freezes allConstrainedNuisances in addition to default othersignals
    """
    others = get_default_othersignals(wildcards, config) if config else []
    r_others = [f"r{p}" if not p.startswith("r") else p for p in others]
    ft = getattr(wildcards, "fit_type", "")
    if ft in ["exp_stat", "obs_stat"]:
        return r_others + ["allConstrainedNuisances"]
    return r_others


def get_likelihood_scan_chunks(wildcards, config):
    """
    Generate all chunk filenames for a given path and signallabel.
    In blinded mode, returns only 'exp' chunks; in unblinded mode, returns 'obs' and 'exp' chunks.
    When full systematics are present, returns chunks for all decomposition components.
    """
    points = int(config.get("likelihood_scan_points", 50))
    split_size = int(config.get("likelihood_scan_split_size", 10))
    num_splits = (points + split_size - 1) // split_size
    fit_types = get_likelihood_scan_fit_types(wildcards, config)
    
    chunks = []
    for ft in fit_types:
        chunks.append(
            f"{wildcards.path}/likelihood_scan/datacard_likelihood_scan_snapshot_{ft}__{wildcards.signallabel}.root"
        )
        for i in range(num_splits):
            chunks.append(
                f"{wildcards.path}/likelihood_scan/datacard_likelihood_scan_chunk_{ft}_{i}__{wildcards.signallabel}.root"
            )
    return chunks

