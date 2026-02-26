import os, sys
import tarfile
import logging

import numpy as np
import awkward as ak
import correctionlib


def extract_jetmet_tar_files(tar_file_name: str=None,
                            jet_type: str='AK4PFchs'
                            ):
    """Extracts a tar.gz file to a specified path and returns a list of extracted files with their locations.

    Args:
        tar_file_name: The name of the tar.gz file.
        jet_type: The type of jet to apply correction

    Returns:
        A list of tuples, where each tuple contains the file name and its full path.
    """

    extracted_files = []
    # Prefer Condor scratch, then TMPDIR, then fallback to /tmp
    extract_path = (
        os.getenv("_CONDOR_SCRATCH_DIR")
        or os.getenv("TMPDIR")
        or f"/tmp/{os.getenv('USER') or os.getuid()}/"
    )
    os.makedirs(extract_path, exist_ok=True)

    with tarfile.open(tar_file_name, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                # Extract the file to the specified path
                member.name = os.path.basename(member.name)  # Remove any directory structure from the archive

                new_file_name = member.name
                if ('Puppi' in jet_type) and (jet_type in member.name):
                    if ('Summer23' not in member.name) and ('Summer24' not in member.name):  # only apply to old JECs where puppi and CHS files have different name formats
                        new_file_name = member.name.replace('_', '', 1) #22 and 23 corrections have different name formats
                        member.name = new_file_name

                new_file_path = os.path.join(extract_path, new_file_name)

                if not os.path.isfile(new_file_path):
                    tar.extract(member, path=extract_path)
                    old_file_path = os.path.join(extract_path, member.name)
                    # Rename the file if the name was changed
                    if old_file_path != new_file_path:
                        os.rename(old_file_path, new_file_path)

                # Get the full path of the extracted file
                file_path = os.path.join(extract_path, member.name)
                if jet_type in member.name:
                    extracted_files.append(f"* * {file_path}")

    logging.debug(f"Extracted files for jet type {jet_type}: {extracted_files}")

    return extracted_files

# following example here: https://github.com/CoffeaTeam/coffea/blob/master/tests/test_jetmet_tools.py#L529
def apply_jerc_corrections( event,
                    corrections_metadata: dict = {},
                    run_systematics: bool = False,
                    isMC: bool = False,
                    dataset: str = None,
                    jet_type: str = 'AK4PFchs',
                    jec_levels: list = ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
                    jer_levels: list = ["PtResolution", "SF"],
                    jet_corr_factor: float = 1.0
                    ):

    logging.info(f"Applying JEC/JER corrections for {dataset}")

    if isMC:
        jec_file = corrections_metadata['JEC_MC']
    else:
        jec_data = corrections_metadata['JEC_DATA']
        jec_file = None
        for key in sorted(jec_data, key=len, reverse=True):
            if dataset.endswith(key):
                jec_file = jec_data[key]
                break
        if jec_file is None:
            raise KeyError(f"No JEC_DATA key matched dataset {dataset!r}; available keys: {list(jec_data.keys())}")
    extracted_files = extract_jetmet_tar_files(jec_file, jet_type=jet_type)
    if run_systematics: jec_levels.append("RegroupedV2")
    weight_sets = list(set([file for level in jec_levels for file in extracted_files if level in file]))  ## list(set()) to remove duplicates

    if isMC and ('202' not in dataset):
        jer_file = corrections_metadata["JER_MC"]
        extracted_files = extract_jetmet_tar_files(jer_file, jet_type=jet_type)
        weight_sets += [file for level in jer_levels for file in extracted_files if level in file]

    logging.debug(f"For {dataset}, applying these corrections: {weight_sets}")

    event['Jet', 'pt_raw']    = (1 - event.Jet.rawFactor) * event.Jet.pt * jet_corr_factor
    event['Jet', 'mass_raw']  = (1 - event.Jet.rawFactor) * event.Jet.mass * jet_corr_factor
    nominal_jet = event.Jet
    if isMC: nominal_jet['pt_gen'] = ak.values_astype(ak.fill_none(nominal_jet.matched_gen.pt, 0), np.float32)

    nominal_jet['rho'] = ak.broadcast_arrays((event.Rho.fixedGridRhoFastjetAll if 'Rho' in event.fields else event.fixedGridRhoFastjetAll), nominal_jet.pt)[0]

    from coffea.lookup_tools import extractor
    extract = extractor()
    extract.add_weight_sets(weight_sets)
    extract.finalize()
    evaluator = extract.make_evaluator()

    from src.physics.objects.jetmet_tools import CorrectedJetsFactory
    from coffea.jetmet_tools import JECStack
    jec_stack_names = []
    for key in evaluator.keys():
        jec_stack_names.append(key)
        if 'UncertaintySources' in key:
            jec_stack_names.append(key)

    logging.debug(jec_stack_names)
    jec_inputs = {name: evaluator[name] for name in jec_stack_names}
    logging.debug('jec_inputs:')
    logging.debug(jec_inputs)
    jec_stack = JECStack(jec_inputs)
    logging.debug('jec_stack')
    logging.debug(jec_stack.__dict__)

    name_map = jec_stack.blank_name_map
    name_map["JetPt"]    = "pt"
    name_map["JetPhi"]    = "phi"
    name_map["JetMass"]  = "mass"
    name_map["JetEta"]   = "eta"
    name_map["JetA"]     = "area"
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw']    = 'pt_raw'
    name_map['massRaw']  = 'mass_raw'
    name_map['Rho']      = 'rho'
    logging.debug(name_map)

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    uncertainties = jet_factory.uncertainties()
    if uncertainties:
        for unc in uncertainties:
            logging.debug(unc)
    else:
        logging.warning('WARNING: No uncertainties were loaded in the jet factory')

    jet_variations = jet_factory.build(nominal_jet, event.event)

    return jet_variations

# ── JSON-POG (correctionlib) based JERC ───────────────────────────────────────

# Correction levels that appear in the JSON-POG key names but are NOT
# JES uncertainty sources (used to filter them out during auto-detection).
_JEC_LEVELS = frozenset({"L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual", "L1RC"})

def _detect_junc_sources(cset, prefix: str, suffix: str) -> list:
    """Return all JES uncertainty source names present in *cset*.

    Iterates over ``cset.keys()`` and strips *prefix* / *suffix* from each
    matching key, then filters out known JEC correction levels so only genuine
    uncertainty sources (``Total``, ``Regrouped_*``, individual sources, …)
    are returned.
    """
    return [
        key[len(prefix):-len(suffix)]
        for key in cset.keys()
        if key.startswith(prefix) and key.endswith(suffix)
        and key[len(prefix):-len(suffix)] not in _JEC_LEVELS
    ]


def apply_jerc_corrections_jsonpog(
    event,
    corrections_metadata: dict = {},
    isMC: bool = True,
    dataset: str = None,
    run_systematics: bool = False,
    jet_type: str = "AK4PFchs",
    seeds=("JER",),
    jet_corr_factor: float = 1.0,
):
    """Apply JEC/JER corrections from a CMS JSON-POG file via correctionlib.

    Drop-in replacement for :func:`apply_jerc_corrections` that reads directly
    from the JSON-POG files shipped in
    ``/cvmfs/.../jsonpog-integration/POG/JME/…/jet_jerc.json.gz``
    instead of extracting txt files from tar archives.

    All JEC/JER campaign names, versions, run tags, and uncertainty source
    lists are read from ``corrections_metadata``, mirroring the interface
    of :func:`apply_jerc_corrections`.

    The ``corrections_metadata`` dict must contain a ``jec`` sub-dict with::

        jec:
          file:          <path to jet_jerc.json.gz>
          jec_campaign:  Summer19UL18
          jec_version:   V5
          jer_campaign:  Summer19UL18      # optional; JER skipped if absent
          jer_version:   JRV2              # optional
          run_tags:                        # DATA only: era letter → run tag
            A: RunA
            ...

    And optionally ``jes_unc`` (list of JES uncertainty source names).

    Args:
        event:               NanoAOD event array.
        corrections_metadata: Year-level corrections dict (e.g.
                             ``corrections_metadata['UL18']``).
        isMC:                ``True`` for simulation, ``False`` for data.
        dataset:             Dataset name; last character is used to look up
                             the run tag for DATA.
        run_systematics:     Compute JER up/down and JES up/down variations.
        jet_type:            Jet algorithm label, e.g. ``"AK4PFchs"``.
        seeds:               Seed sequence for the deterministic JER RNG.
        jet_corr_factor:     Multiplicative factor on raw pt/mass (default 1).

    Returns:
        Awkward array with the same structure as :func:`apply_jerc_corrections`.
    """
    logging.info(f"Applying JSON-POG JEC/JER corrections for {dataset}")

    from src.physics.objects.jetmet_tools.CorrectedJetsFactory import (
        _JsonPogJEC, _JsonPogJER, _JsonPogJERSF, _JsonPogJUNC, _JsonPogJECStack,
    )

    # ── extract parameters from corrections_metadata ──────────────────────────
    jec_meta     = corrections_metadata["jec"]
    jerc_file    = jec_meta["file"]
    jec_campaign = jec_meta["jec_campaign"]
    jec_version  = jec_meta["jec_version"]
    jer_campaign = jec_meta.get("jer_campaign")
    jer_version  = jec_meta.get("jer_version")
    junc_sources = corrections_metadata.get("jes_unc") if run_systematics else None

    # Resolve run_tag for DATA: match the dataset suffix against run_tags keys
    # (longest match first to handle multi-char eras like C01, D1, etc.)
    run_tag = None
    if not isMC and dataset:
        run_tags = jec_meta.get("run_tags", {})
        for key in sorted(run_tags, key=len, reverse=True):
            if dataset.endswith(key):
                run_tag = run_tags[key]
                break
        if run_tag is None and run_tags:
            logging.warning(f"No run_tag matched for dataset {dataset!r} in {list(run_tags.keys())}")

    cset       = correctionlib.CorrectionSet.from_file(jerc_file)
    era        = "MC" if isMC else "DATA"
    jec_tag    = f"{run_tag}_{jec_version}_{era}" if (not isMC and run_tag) else f"{jec_version}_{era}"
    key_suffix = f"_{jet_type}"

    # ── prepare raw quantities (identical to apply_jerc_corrections) ──────────
    event["Jet", "pt_raw"]   = (1 - event.Jet.rawFactor) * event.Jet.pt   * jet_corr_factor
    event["Jet", "mass_raw"] = (1 - event.Jet.rawFactor) * event.Jet.mass * jet_corr_factor
    nominal_jet = event.Jet
    if isMC:
        nominal_jet["pt_gen"] = ak.values_astype(
            ak.fill_none(nominal_jet.matched_gen.pt, 0), np.float32
        )
    nominal_jet["rho"] = ak.broadcast_arrays(
        event.Rho.fixedGridRhoFastjetAll if "Rho" in event.fields else event.fixedGridRhoFastjetAll,
        nominal_jet.pt,
    )[0]

    # ── JEC adapter ───────────────────────────────────────────────────────────
    compound_key = f"{jec_campaign}_{jec_tag}_L1L2L3Res{key_suffix}"
    jec = _JsonPogJEC(cset.compound[compound_key])

    # ── JER adapters (MC only, when campaign/version are provided) ────────────
    jer   = None
    jersf = None
    if isMC and jer_campaign and jer_version:
        jer   = _JsonPogJER(cset[f"{jer_campaign}_{jer_version}_MC_PtResolution{key_suffix}"])
        jersf = _JsonPogJERSF(cset[f"{jer_campaign}_{jer_version}_MC_ScaleFactor{key_suffix}"])

    # ── JES uncertainty adapters ──────────────────────────────────────────────
    junc = None
    if run_systematics and junc_sources is not None:
        key_prefix   = f"{jec_campaign}_{jec_tag}_"
        sources      = junc_sources or _detect_junc_sources(cset, key_prefix, key_suffix)
        known_keys   = set(cset.keys())
        source_pairs = []
        for src in sources:
            key = f"{jec_campaign}_{jec_tag}_{src}{key_suffix}"
            if key in known_keys:
                source_pairs.append((src, cset[key]))
            else:
                logging.warning(f"JES source {key!r} not found in CorrectionSet, skipping")
        if source_pairs:
            junc = _JsonPogJUNC(source_pairs)

    # ── assemble mock JECStack and run FixedCorrectedJetsFactory ─────────────
    jec_stack = _JsonPogJECStack(jec=jec, jer=jer, jersf=jersf, junc=junc)

    name_map = {
        "JetPt":    "pt",
        "JetMass":  "mass",
        "JetEta":   "eta",
        "JetPhi":   "phi",
        "JetA":     "area",
        "ptGenJet": "pt_gen",
        "ptRaw":    "pt_raw",
        "massRaw":  "mass_raw",
        "Rho":      "rho",
    }

    from src.physics.objects.jetmet_tools import CorrectedJetsFactory
    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    return jet_factory.build(nominal_jet, event.event, seeds=seeds)


def apply_jet_veto_maps( corrections_metadata, jets, event_veto: bool = False ):
    '''
    taken from https://github.com/PocketCoffea/PocketCoffea/blob/main/pocket_coffea/lib/cut_functions.py#L65
    modified to veto jets not events
    '''

    mask_for_VetoMap = (
        ((jets.jetId & 2)==2) # Must fulfill tight jetId
        & (abs(jets.eta) < 5.19) # Must be within HCal acceptance
        & ((jets.neEmEF + jets.chEmEF) < 0.9) # Energy fraction not dominated by ECal
    )
    if 'muonSubtrFactor' in jets.fields:  ### AGE: this should be temporary for old picos. New skims should have this field
        mask_for_VetoMap = mask_for_VetoMap & (jets.muonSubtrFactor < 0.8) # May no be Muons misreconstructed as jets
    else: logging.warning("muonSubtrFactor NOT in jets fields. This is correct only for mixeddata and old picos.")

    corr = correctionlib.CorrectionSet.from_file(corrections_metadata['file'])[corrections_metadata['tag']]

    etaFlat, phiFlat, etaCounts = ak.flatten(jets.eta), ak.flatten(jets.phi), ak.num(jets.eta)
    phiFlat = np.clip(phiFlat, -3.14159, 3.14159) # Needed since no overflow included in phi binning
    weight = ak.unflatten(
        corr.evaluate("jetvetomap", etaFlat, phiFlat),
        counts=etaCounts,
    )
    jetMask = ak.where( weight == 0, True, False, axis=1 )  # if 0 is not vetoed, then True

    if event_veto == False:
        veto_mask = jetMask & mask_for_VetoMap
    else:
        veto_mask = jetMask | ~(mask_for_VetoMap)

    return veto_mask