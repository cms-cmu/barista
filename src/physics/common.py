import sys
import logging
import pickle
import warnings

import awkward as ak
import numpy as np
import correctionlib
from coffea.nanoevents import NanoAODSchema
from coffea.nanoevents.methods import vector

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")
ak.behavior.update(vector.behavior)


def mask_event_decision(event, decision='OR', branch='HLT', list_to_mask=[''], list_to_skip=['']):
    '''
    Takes event.branch and passes an boolean array mask with the decisions of all the list_to_mask
    '''

    tmp_list = []
    if branch in event.fields:
        for i in list_to_mask:
            if i in event[branch].fields:
                tmp_list.append( event[branch][i] )
            elif i in list_to_skip: continue
            else: logging.warning(f'\n{i} branch not in {branch} for event.')
    else: logging.warning(f'\n{branch} branch not in event.')
    if not tmp_list:
        tmp_list = [np.zeros(len(event), dtype=bool)]
        logging.warning(f'No {list_to_mask} branches found in event. Returning empty mask.')

    tmp_array = np.array( tmp_list )

    if decision.lower().startswith('or'): decision_array = np.any( tmp_array, axis=0 )
    else: decision_array = np.all( tmp_array, axis=0 )

    return decision_array

def apply_btag_sf( jets,
                    correction_file='src/data/JEC/BTagSF2016/btagging_legacy16_deepJet_itFit.json.gz',
                    correction_type="deepJet_shape",
                    sys_value = 'central',
                    btag_uncertainties = None,
                    dataset = '',
                    btagSF_norm_file='ZZ4b/nTupleAnalysis/weights/btagSF_norm.pkl',
                    ):
    '''
    Can be replace with coffea.btag_tools when btag_tools accept jsonpog files
    '''

    btagSF = correctionlib.CorrectionSet.from_file(correction_file)[correction_type]

    weights = {}
    nj = ak.num(jets)

    def get_sf(jets, sys):
        flat_jets = ak.flatten(jets)
        return np.prod(
            ak.unflatten(
                btagSF.evaluate(
                    sys,
                    ak.to_numpy(flat_jets.hadronFlavour),
                    ak.to_numpy(np.abs(flat_jets.eta)),
                    ak.to_numpy(flat_jets.pt),
                    ak.to_numpy(flat_jets.btagScore)
                ),
                ak.num(jets)
            ), axis=1
        )

    jets_bl = jets[jets.hadronFlavour != 4]
    SF_bl = get_sf(jets_bl, sys_value)

    jets_c = jets[jets.hadronFlavour == 4]
    SF_c = get_sf(jets_c, sys_value)

    ### btag norm
    try:
        with open(btagSF_norm_file, 'rb') as f:
            btagSF_norm = pickle.load(f)[dataset]
            logging.info(f'btagSF_norm {btagSF_norm}')
    except FileNotFoundError:
        btagSF_norm = 1.0


    btag_var = [sys_value]
    if btag_uncertainties:
        btag_var += [f'{updown}_{btagvar}' for updown in ['up', 'down'] for btagvar in btag_uncertainties]
    for sf in btag_var:
        if sf == 'central':
            SF = get_sf(jets, 'central')
        elif '_cf' in sf:
            SF = get_sf(jets_c, sf)
            SF = SF_bl * SF  # use central value for b,l jets
        elif any(x in sf for x in ['_hf', '_lf', '_jes']):
            SF = get_sf(jets_bl, sf)
            SF = SF_c * SF  # use central value for charm jets
        else:
            SF = get_sf(jets, sf)
        weights[f'btagSF_{sf}'] = SF * btagSF_norm



    logging.debug(weights)
    return weights


def drClean(coll1,coll2,cone=0.4):

    from coffea.nanoevents.methods import vector
    j_eta = coll1.eta
    j_phi = coll1.phi
    l_eta = coll2.eta
    l_phi = coll2.phi

    j_eta, l_eta = ak.unzip(ak.cartesian([j_eta, l_eta], nested=True))
    j_phi, l_phi = ak.unzip(ak.cartesian([j_phi, l_phi], nested=True))
    delta_eta = j_eta - l_eta
    delta_phi = vector._deltaphi_kernel(j_phi,l_phi)
    dr = np.hypot(delta_eta, delta_phi)
    nolepton_mask = ~ak.any(dr < cone, axis=2)
    jets_noleptons = coll1[nolepton_mask]
    return [jets_noleptons, nolepton_mask]

def update_events(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out

def compute_puid( jet, dataset ):
    """Compute the PUId for the given jet collection based on correctionlib. To be used in UL"""

    puid_WP_table = correctionlib.CorrectionSet.from_file('src/data/puId/puid_tightWP.json')['PUID']

    n, j = ak.num(jet), ak.flatten(jet)
    puid_WP = puid_WP_table.evaluate( j.pt, abs(j.eta), f"UL{dataset.split('UL')[1][:2]}" )

    logging.debug(f"puid_WP: {puid_WP[:10]}")
    logging.debug(f"puIdDisc: {j.puIdDisc[:10]}")
    logging.debug(f"eta: {j.eta[:10]}")
    logging.debug(f"pt: {j.pt[:10]}")
    logging.debug(f"puId: {j.puId[:10]}")
    j['is_pujet'] = ak.where( j.puIdDisc < puid_WP, True, False )
    logging.debug(f"is_pujet: {j['is_pujet'][:10]}\n\n")
    jet = ak.unflatten(j, n)

    return jet["is_pujet"]
