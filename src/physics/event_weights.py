import logging
from coffea.analysis_tools import Weights
import correctionlib
import awkward as ak
import numpy as np

def add_weights(event, do_MC_weights: bool = True,
                dataset: str = None,
                year_label: str = None,
                corrections_metadata: dict = None,
                apply_trigWeight: bool = True,
                friend_trigWeight: callable = None,
                isTTForMixed: bool = False,
                target: callable = None,
                run_systematics: bool = False
                ):
    """
    Add event weights for physics analysis, including generator weights, trigger weights,
    pileup reweighting, L1 prefiring corrections, and parton shower variations.

    This function constructs a Weights object containing all relevant scale factors and
    systematic uncertainties for Monte Carlo simulation events, or unity weights for data.

    Parameters
    ----------
    event : awkward.Array
        Event data containing fields like genWeight, trigWeight, Pileup, L1PreFiringWeight, etc.
    do_MC_weights : bool, default True
        Whether to apply Monte Carlo specific weights (genWeight, trigger, pileup, etc.).
        Set to False for data events.
    dataset : str, optional
        Name of the dataset being processed (currently unused but kept for future extensions).
    year_label : str, optional
        Year identifier for year-dependent corrections (e.g., "2018", "2022").
        Required when applying pileup or L1 prefiring weights.
    corrections_metadata : dict, optional
        Dictionary containing paths to correction files, particularly for pileup reweighting.
        Expected to have a "PU" key pointing to the pileup correction file.
    isTTForMixed : bool, default False
        Special flag for ttbar mixed samples to skip pileup reweighting.
    run_systematics : bool, default False
        Whether to include systematic uncertainty variations in addition to nominal weights.
        When True, adds up/down variations for trigger, pileup, prefiring, and PS weights.

    Returns
    -------
    weights : coffea.analysis_tools.Weights
        Weights object containing all computed scale factors and their systematic variations.
        Can be used to extract nominal weights with weights.weight() or partial weights.
    list_weight_names : list of str
        List of weight names that were added to the Weights object, useful for tracking
        which corrections were applied.

    Notes
    -----
    - Generator weights are normalized by cross-section, luminosity, k-factor, and sum of weights
    - Trigger weights use CMS naming convention "CMS_bbbb_resolved_ggf_triggerEffSF"
    - Pileup weights are applied using correctionlib with year-specific corrections
    - L1 prefiring weights are only available for certain years (2016-2017)
    - Parton shower weights (ISR/FSR) are only added when run_systematics=True
    - For data events (do_MC_weights=False), only unity weights are applied

    Examples
    --------
    >>> weights, weight_names = add_weights(
    ...     event, 
    ...     year_label="2018",
    ...     corrections_metadata={"PU": "/path/to/pileup.json"},
    ...     run_systematics=True
    ... )
    >>> nominal_weight = weights.weight()
    >>> pileup_up = weights.partial_weight(include=["CMS_pileup_2018"], modifier="up")
    """

    weights = Weights(len(event), storeIndividual=True)
    list_weight_names = []

    if do_MC_weights:
        # genWeight
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)
        weights.add( "genweight", event.genWeight * (lumi * xs * kFactor / event.metadata["genEventSumw"]) )
        list_weight_names.append('genweight')
        logging.debug( f"genweight {weights.partial_weight(include=['genweight'])[:10]}\n" )
        logging.debug( f" = {event.genWeight} * ({lumi} * {xs} * {kFactor} / {event.metadata['genEventSumw']})\n")

        # puWeight
        if not isTTForMixed:
            puWeight = list( correctionlib.CorrectionSet.from_file( corrections_metadata["PU"] ).values() )[0]
            if run_systematics:
                weights.add(
                    f"CMS_pileup_{year_label}",
                    puWeight.evaluate(event.Pileup.nTrueInt, "nominal"),
                    puWeight.evaluate(event.Pileup.nTrueInt, "up"),
                    puWeight.evaluate(event.Pileup.nTrueInt, "down"),
                )
            else:
                weights.add(
                    f"CMS_pileup_{year_label}",
                    puWeight.evaluate(event.Pileup.nTrueInt, "nominal")
                )
            list_weight_names.append(f"CMS_pileup_{year_label}")
            logging.debug( f"PU weight {weights.partial_weight(include=[f'CMS_pileup_{year_label}'])[:10]}\n" )

        # L1 prefiring weight
        if ( "L1PreFiringWeight" in event.fields ):
            if run_systematics:
                weights.add( 
                    f"CMS_prefire_{year_label}",
                    event.L1PreFiringWeight.Nom,
                    event.L1PreFiringWeight.Up,
                    event.L1PreFiringWeight.Dn,
                )
            else:
                weights.add(
                    f"CMS_prefire_{year_label}",
                    event.L1PreFiringWeight.Nom,
                )
            logging.debug( f"L1Prefire weight {weights.partial_weight(include=[f'CMS_prefire_{year_label}'])[:10]}\n" )
            list_weight_names.append(f"CMS_prefire_{year_label}")

        if run_systematics:
            nom      = np.ones(len(weights.weight()))
            up_isr   = np.ones(len(weights.weight()))
            down_isr = np.ones(len(weights.weight()))
            up_fsr   = np.ones(len(weights.weight()))
            down_fsr = np.ones(len(weights.weight()))

            if len(event.PSWeight[0]) == 4:
                up_isr   = event.PSWeight[:, 0]
                down_isr = event.PSWeight[:, 2]
                up_fsr   = event.PSWeight[:, 1]
                down_fsr = event.PSWeight[:, 3]

            else:
                logging.warning( f"PS weight vector has length {len(event.PSWeight[0])}" )

            weights.add("ps_isr", nom, up_isr, down_isr)
            weights.add("ps_fsr", nom, up_fsr, down_fsr)
            list_weight_names.append(f"ps_isr")
            list_weight_names.append(f"ps_fsr")

        # pdf_Higgs_ggHH, alpha_s, PDFaS weights are included in datacards through the inference tool. Kept this code for reference.
        # if "LHEPdfWeight" in event.fields:

        #     # https://github.com/nsmith-/boostedhiggs/blob/a33dca8464018936fbe27e86d52c700115343542/boostedhiggs/corrections.py#L53
        #     nom  = np.ones(len(weights.weight()))
        #     up   = np.ones(len(weights.weight()))
        #     down = np.ones(len(weights.weight()))

        #     # NNPDF31_nnlo_hessian_pdfas
        #     # https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_hessian_pdfas/NNPDF31_nnlo_hessian_pdfas.info
        #     if "306000 - 306102" in event.LHEPdfWeight.__doc__:
        #         # Hessian PDF weights
        #         # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf
        #         arg = event.LHEPdfWeight[:, 1:-2] - np.ones( (len(weights.weight()), 100) )

        #         summed = ak.sum(np.square(arg), axis=1)
        #         pdf_unc = np.sqrt((1.0 / 99.0) * summed)
        #         weights.add("pdf_Higgs_ggHH", nom, pdf_unc + nom)

        #         # alpha_S weights
        #         # Eq. 27 of same ref
        #         as_unc = 0.5 * ( event.LHEPdfWeight[:, 102] - event.LHEPdfWeight[:, 101] )

        #         weights.add("alpha_s", nom, as_unc + nom)

        #         # PDF + alpha_S weights
        #         # Eq. 28 of same ref
        #         pdfas_unc = np.sqrt(np.square(pdf_unc) + np.square(as_unc))
        #         weights.add("PDFaS", nom, pdfas_unc + nom)

        #     else:
        #         weights.add("alpha_s", nom, up, down)
        #         weights.add("pdf_Higgs_ggHH", nom, up, down)
        #         weights.add("PDFaS", nom, up, down)
        #     list_weight_names.append(f"alpha_s")
        #     list_weight_names.append(f"pdf_Higgs_ggHH")
        #     list_weight_names.append(f"PDFaS")
    else:
        weights.add("data", np.ones(len(event)))
        list_weight_names.append(f"data")

    logging.debug(f"weights event {weights.weight()[:10]}")
    logging.debug(f"Weight Statistics {weights.weightStatistics}")

    return weights, list_weight_names

def add_btagweights( event, weights,
                    list_weight_names: list = [],
                    shift_name: str = None,
                    run_systematics: bool = False,
                    use_prestored_btag_SF: bool = False,
                    corrections_metadata: dict = None,
                    jet_field: str = 'selJet_no_bRegCorr'
                    ):

    if use_prestored_btag_SF:
        weights.add( "CMS_btag", event.CMSbtag )
    else:

        sys_value = "central"
        if shift_name and ( 'CMS_scale_j_' in shift_name ):
            if 'Down' in shift_name:
                sys_value = f"down_jes{shift_name.replace('CMS_scale_j_', '').replace('Down', '')}"
            elif 'Up' in shift_name:
                sys_value = f"up_jes{shift_name.replace('CMS_scale_j_', '').replace('Up', '')}"
        logging.debug(f"shift_name: {shift_name}, sys_value: {sys_value}\n\n")

        btag_SF_weights = apply_btag_sf(
            event[jet_field],
            sys_value="central",
            correction_file=corrections_metadata["btagSF"],
            btag_uncertainties=corrections_metadata["btag_uncertainties"] if (not shift_name) & run_systematics else None
        )

        if (not shift_name) & run_systematics:
            weights.add_multivariation( f"CMS_btag", btag_SF_weights["btagSF_central"],
                                        corrections_metadata["btag_uncertainties"],
                                        [ var.to_numpy() for name, var in btag_SF_weights.items() if "_up" in name ],
                                        [ var.to_numpy() for name, var in btag_SF_weights.items() if "_down" in name ], )
        else:
            weights.add( "CMS_btag", btag_SF_weights["btagSF_central"] )
    list_weight_names.append(f"CMS_btag")

    return weights, list_weight_names
