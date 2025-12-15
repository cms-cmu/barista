import os
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
                    if ('Summer23' not in member.name):
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

    jec_file = corrections_metadata['JEC_MC'] if isMC else corrections_metadata['JEC_DATA'][dataset[-1]]
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

##### JER needs to be included
def jet_corrections( uncorr_jets,
                    fixedGridRhoFastjetAll,
                    isMC,
                    jercFile="/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/jet_jerc.json.gz",
                    data_campaign="Summer19UL18",
                    jec_campaign='V5_MC',
                    jer_campaign='JRV2_MC',
                    jec_type=["L1L2L3Res"],  ####, "PtResolution", 'ScaleFactor'
                    jettype='AK4PFchs',
                    variation='nom'
                    ):

    JECFile = correctionlib.CorrectionSet.from_file(jercFile)
    # preparing jets
    uncorr_jets['pt_raw'] = (1 - uncorr_jets['rawFactor']) * uncorr_jets['pt']
    uncorr_jets['mass_raw'] = (1 - uncorr_jets['rawFactor']) * uncorr_jets['mass']
    uncorr_jets['rho'] = ak.broadcast_arrays(fixedGridRhoFastjetAll, uncorr_jets.pt)[0]
    j, nj = ak.flatten(uncorr_jets), ak.num(uncorr_jets)

    jec_campaign = f'{jec_campaign}_{"MC" if isMC else "DATA"}'

    total_flat_jec = np.ones( len(j), dtype="float32" )
    for ijec in jec_type:

        if 'L1' in ijec:
            corr = JECFile.compound[f'{data_campaign}_{jec_campaign}_{ijec}_{jettype}'] if 'L1L2L3' in ijec else JECFile[f'{data_campaign}_{jec_campaign}_{ijec}_{jettype}']
            flat_jec = corr.evaluate( j['area'], j['eta'], j['pt_raw'], j['rho']  )
        else:
            corr = JECFile[f'{data_campaign}_{jec_campaign}_{ijec}_{jettype}']
            flat_jec = corr.evaluate( j['eta'], j['pt_raw']  )
        total_flat_jec *= flat_jec
    jec = ak.unflatten(total_flat_jec, nj)

    corr_jets = uncorr_jets
    corr_jets['jet_energy_correction'] = jec
    corr_jets['pt'] = corr_jets.pt_raw * jec
    corr_jets['mass'] = corr_jets.mass_raw * jec

    return corr_jets

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