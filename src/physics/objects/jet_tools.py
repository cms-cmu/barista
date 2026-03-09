import logging
import numpy as np
import awkward as ak
import correctionlib

logger = logging.getLogger(__name__)

# Cache parsed CorrectionSets so the JSON file is loaded once per path
# rather than once per chunk.
_cset_cache: dict = {}

def _get_correction_set(path: str):
    """Return a cached ``correctionlib.CorrectionSet`` for *path*."""
    if path not in _cset_cache:
        logger.debug("CorrectionSet cache miss — loading %s", path)
        _cset_cache[path] = correctionlib.CorrectionSet.from_file(path)
    return _cset_cache[path]


def compute_jet_id(jets, jet_id_file, jet_id_tag):
    """
    Evaluate a single jet ID working point from a correctionlib JSON-POG file.

    Parameters
    ----------
    jets : ak.Array
        Jet collection with fields: ``eta``, ``chHEF``, ``neHEF``,
        ``chEmEF``, ``neEmEF``, ``muEF``, ``chMultiplicity``,
        ``neMultiplicity``.
    jet_id_file : str
        Path to a correctionlib ``jetid.json.gz`` file.
    jet_id_tag : str
        Correction name inside the file, e.g. ``'AK4PUPPI_TightLeptonVeto'``.

    Returns
    -------
    ak.Array
        Boolean array (same jagged structure as *jets*).
        ``True`` where the jet passes the requested working point.
    """
    cset = _get_correction_set(jet_id_file)
    corr = cset[jet_id_tag]

    # Flatten jagged arrays for correctionlib evaluation
    counts = ak.num(jets)
    flat   = ak.flatten(jets)

    n_jets = len(flat)
    logger.debug("compute_jet_id: tag=%s, n_jets=%d, file=%s", jet_id_tag, n_jets, jet_id_file)

    # --- energy fractions (real) ---
    eta    = np.asarray(flat.eta,    dtype=np.float64)
    chHEF  = np.asarray(flat.chHEF,  dtype=np.float64)
    neHEF  = np.asarray(flat.neHEF,  dtype=np.float64)
    chEmEF = np.asarray(flat.chEmEF, dtype=np.float64)
    neEmEF = np.asarray(flat.neEmEF, dtype=np.float64)
    muEF   = np.asarray(flat.muEF,   dtype=np.float64)

    # --- multiplicities (int) ---
    chMult = np.asarray(flat.chMultiplicity, dtype=np.int32)
    neMult = np.asarray(flat.neMultiplicity, dtype=np.int32)
    mult   = np.asarray(chMult + neMult,     dtype=np.int32)

    result = corr.evaluate(eta, chHEF, neHEF, chEmEF, neEmEF, muEF,
                           chMult, neMult, mult)

    pass_mask = result.astype(bool)
    n_pass = np.sum(pass_mask)
    logger.debug("compute_jet_id: %d / %d jets pass %s (%.1f%%)",
                 n_pass, n_jets, jet_id_tag,
                 100.0 * n_pass / n_jets if n_jets > 0 else 0.0)

    return ak.unflatten(pass_mask, counts)
