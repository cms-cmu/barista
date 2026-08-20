import numpy as np
import awkward as ak
from coffea.lumi_tools import LumiMask
from src.physics.common import mask_event_decision

_lumimask_cache: dict[str, LumiMask] = {}

def _get_lumimask(json_path: str) -> LumiMask:
    if json_path not in _lumimask_cache:
        _lumimask_cache[json_path] = LumiMask(json_path)
    return _lumimask_cache[json_path]

def apply_event_selection(
    event: ak.Array, 
    corrections_metadata: dict, 
    cut_on_lumimask: bool = True
) -> ak.Array:
    """
    Applies basic event selection criteria for a 4b analysis.

    Parameters:
    -----------
    event : awkward.Array
        The event data containing fields such as `run`, `luminosityBlock`, `HLT`, and `Flag`.
    corrections_metadata : dict
        Metadata containing corrections and configuration information, such as:
        - `goldenJSON`: Path to the golden JSON file for luminosity masking.
        - `NoiseFilter`: List of noise filters to apply.
    cut_on_lumimask : bool, optional
        Whether to apply the luminosity mask cut. Defaults to True.

    Returns:
    --------
    event : awkward.Array
        The input event data with additional fields:
        - `lumimask`: Boolean mask indicating events passing the luminosity mask.
        - `passHLT`: Boolean mask indicating events passing the HLT trigger.
        - `passNoiseFilter`: Boolean mask indicating events passing noise filters.
    """
    # Apply luminosity mask
    if cut_on_lumimask:
        if 'goldenJSON' not in corrections_metadata:
            raise KeyError("Missing 'goldenJSON' in corrections_metadata.")
        lumimask = _get_lumimask(corrections_metadata['goldenJSON'])
        event['lumimask'] = np.array(lumimask(event.run, event.luminosityBlock))
    else:
        event['lumimask'] = np.full(len(event), True)

    # Apply HLT trigger mask
    event['passHLT'] = (
        np.full(len(event), True)
        if 'HLT' not in event.fields else mask_event_decision(
            event, decision="OR", branch="HLT", list_to_mask=event.metadata.get('trigger', [])
        )
    )

    # Apply noise filter mask
    noise_filters = corrections_metadata.get('NoiseFilter', [])
    event['passNoiseFilter'] = (
        np.full(len(event), True)
        if 'Flag' not in event.fields else mask_event_decision(
            event, decision="AND", branch="Flag", list_to_mask=noise_filters,
            list_to_skip=['BadPFMuonDzFilter', 'hfNoisyHitsFilter']
        )
    )

    return event