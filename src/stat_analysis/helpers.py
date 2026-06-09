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


def get_likelihood_scan_chunks(wildcards, config):
    """
    Generate all chunk filenames for a given path and signallabel.
    """
    points = int(config.get("likelihood_scan_points", 50))
    split_size = int(config.get("likelihood_scan_split_size", 10))
    num_splits = (points + split_size - 1) // split_size
    return [
        f"{wildcards.path}/likelihood_scan/datacard_likelihood_scan_chunk_{i}__{wildcards.signallabel}.root"
        for i in range(num_splits)
    ]

