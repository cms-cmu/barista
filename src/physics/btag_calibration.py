"""b-tagging SF calibration.

The b-tag shape scale factors are meant to correct the shape of the b-tag discriminant, but they
also change the yield of each MC sample. The calibration is a per-sample 2D map in (Njet, HT),

    calib = N(without b-tag SF) / N(with b-tag SF),

measured before any b-tag requirement. Multiplying the b-tag SF by it restores the normalization of
the sample. Maps are stored as correctionlib JSON files, written by
``src/tools/compute_btagSF_calibration.py``.
"""

import logging

import awkward as ak
import correctionlib
import fsspec
import numpy as np

CORRECTION_NAME = "btagSF_norm_correction"

# Names of the inputs of the correction, in evaluation order
INPUT_SAMPLE = "Sample"
INPUT_VARIATION = "variation"
INPUT_NJET = "nJet"
INPUT_HT = "HT"

CENTRAL = "central"


def load_btag_sf_calibration(path: str) -> str:
    """Read a calibration file (local path or ``root://`` URL) and return its JSON text.

    The text, rather than a ``CorrectionSet``, is returned so that it can be stored on a processor
    and shipped to the workers; build the evaluator there with :func:`build_btag_sf_calibration`.
    """
    compression = "gzip" if str(path).endswith(".gz") else None
    with fsspec.open(path, "rt", compression=compression) as f:
        text = f.read()
    logging.info(f"Loaded b-tag SF calibration from {path}")
    return text


def build_btag_sf_calibration(text: str, name: str = CORRECTION_NAME):
    """Build the correctionlib evaluator from the JSON text of a calibration file."""
    return correctionlib.CorrectionSet.from_string(text)[name]


def btag_sf_calibration_samples(text: str, name: str = CORRECTION_NAME) -> list[str]:
    """Samples available in a calibration file."""
    import json

    for corr in json.loads(text)["corrections"]:
        if corr["name"] == name:
            return [str(c["key"]) for c in corr["data"]["content"]]
    raise KeyError(f"Correction '{name}' not found in the b-tag SF calibration file")


def evaluate_btag_sf_calibration(correction, sample, variation: str, njet, ht) -> np.ndarray:
    """Evaluate the calibration for one sample and one b-tag SF variation.

    Parameters
    ----------
    correction :
        Evaluator from :func:`build_btag_sf_calibration`.
    sample : str | dict
        Sample key of the map, or ``{sample key: boolean event mask}`` when different events of
        the sample use different maps (e.g. the components of a stitched sample). The masks
        must be disjoint and cover all the events.
    variation : str
        b-tag SF variation: ``central``, ``up_<unc>`` or ``down_<unc>``, as in ``apply_btag_sf``.
    njet, ht :
        Per-event jet multiplicity and HT, computed from the jets entering the b-tag SF.
    """
    njet = np.asarray(ak.to_numpy(njet), dtype=float)
    ht = np.asarray(ak.to_numpy(ht), dtype=float)
    if isinstance(sample, dict):
        masks = {key: np.asarray(ak.to_numpy(mask), dtype=bool) for key, mask in sample.items()}
        if not np.all(sum(m.astype(int) for m in masks.values()) == 1):
            raise ValueError(f"The event masks of the b-tag SF calibration samples {list(masks)} are not a partition of the events")
        out = np.ones(len(njet))
        for key, mask in masks.items():
            out[mask] = evaluate_btag_sf_calibration(correction, key, variation, njet[mask], ht[mask])
        return out
    try:
        return np.asarray(correction.evaluate(sample, variation, njet, ht))
    except IndexError as e:
        # correctionlib raises IndexError for a missing category key. Do not fall back to 1:
        # a silently uncalibrated sample is hard to spot downstream.
        raise KeyError(
            f"No b-tag SF calibration for sample '{sample}', variation '{variation}': {e}"
        ) from e


class BtagSFCalibration:
    """Per-year access to the calibration files, to be stored on a processor.

    Parameters
    ----------
    files : str | dict
        Either one path, which may contain a ``{year}`` placeholder, or a dict ``{year: path}``.
    sample_map : dict, optional
        ``{sample: key in the map}``, for samples that use the map of another sample. The value
        can also be a dict ``{event category: key in the map}`` for samples whose events use
        different maps (e.g. a stitched sample); the caller defines the categories and turns
        them into event masks for :func:`evaluate_btag_sf_calibration`. Samples not listed use
        their own name.

    Files are read on first use (on the worker) and the evaluators are not pickled.
    """

    def __init__(self, files, sample_map: dict | None = None):
        if not files:
            raise ValueError("b-tag SF calibration requested but no calibration file was given")
        self.files = dict(files) if isinstance(files, dict) else str(files)
        self.sample_map = dict(sample_map) if sample_map else {}
        self._cache = {}

    def __getstate__(self):
        return self.__dict__ | {"_cache": {}}

    def path(self, year: str) -> str:
        if isinstance(self.files, dict):
            if year not in self.files:
                raise KeyError(f"No b-tag SF calibration file for year '{year}' (available: {list(self.files)})")
            return self.files[year]
        return self.files.format(year=year)

    def get(self, year: str, sample: str):
        """Evaluator and map key (or ``{event category: map key}``) to use for ``sample`` in ``year``."""
        if year not in self._cache:
            text = load_btag_sf_calibration(self.path(year))
            self._cache[year] = (build_btag_sf_calibration(text), btag_sf_calibration_samples(text))
        correction, samples = self._cache[year]
        key = self.sample_map.get(sample, sample)
        missing = [k for k in (key.values() if isinstance(key, dict) else [key]) if k not in samples]
        if missing:
            raise KeyError(
                f"No b-tag SF calibration for sample '{sample}' (map key(s) {missing}) in {self.path(year)}. "
                f"Available: {samples}. Use btagSF_calibration_sample_map to point it to another sample."
            )
        return correction, (dict(key) if isinstance(key, dict) else key)
