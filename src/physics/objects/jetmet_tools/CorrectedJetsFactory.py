import math
from typing import Iterable

from cachetools import Cache
from coffea.jetmet_tools.CorrectedJetsFactory import *
from coffea.jetmet_tools.CorrectedJetsFactory import _JERSF_FORM
import numpy as np
from src.math_tools.random import Squares


# ── correctionlib → FixedCorrectedJetsFactory adapters ───────────────────────
#
# These thin wrappers expose the same interface that FixedCorrectedJetsFactory
# expects from a JECStack ( .signature + .getCorrection / .getResolution /
# .getScaleFactor / .getUncertainty ), but pull data from a correctionlib
# CorrectionSet (JSON-POG) instead of txt files.
#
# Inputs arrive as flat awkward arrays from out_dict; numpy.asarray() is used
# to materialise virtual arrays on the fly before calling correctionlib.


class _JsonPogJEC:
    """Compound JEC correction (L1L2L3Res) from a JSON-POG file."""

    def __init__(self, corr):
        self._corr = corr
        # Derive signature from the compound correction's input names,
        # mapping JSON-POG names to the name_map keys used by the factory.
        self._jsonpog_names = [inp.name for inp in corr.inputs]
        _pogname_to_namemap = {
            "JetA":   "JetA",
            "JetEta": "JetEta",
            "JetPt":  "JetPt",
            "JetPhi": "JetPhi",
            "Rho":    "Rho",
            "run":    "run",
        }
        self.signature = tuple(
            _pogname_to_namemap[n] for n in self._jsonpog_names
            if n in _pogname_to_namemap
        )

    def getCorrection(self, form, lazy_cache, **kwargs):
        arrays = [
            numpy.asarray(kwargs[k], dtype=numpy.float32)
            for k in self.signature
        ]
        vals = self._corr.evaluate(*arrays).astype(numpy.float32)
        return awkward.Array(vals)


class _JsonPogJER:
    """JER pt-resolution correction from a JSON-POG file."""

    signature = ("JetEta", "JetPt", "Rho")

    def __init__(self, corr):
        self._corr = corr

    def getResolution(self, JetEta, JetPt, Rho, form, lazy_cache):
        vals = self._corr.evaluate(
            numpy.asarray(JetEta, dtype=numpy.float32),
            numpy.asarray(JetPt,  dtype=numpy.float32),
            numpy.asarray(Rho,    dtype=numpy.float32),
        ).astype(numpy.float32)
        return awkward.Array(vals)


class _JsonPogJERSF:
    """JER scale-factor correction from a JSON-POG file.

    Returns a ``(n_jets, 3)`` array [nom, up, down] so that
    ``jer_smear`` can index it as ``jersf[:, variation]``.
    """

    signature = ("JetEta",)

    def __init__(self, corr):
        self._corr = corr

    def getScaleFactor(self, JetEta, form, lazy_cache):
        eta = numpy.asarray(JetEta, dtype=numpy.float32)
        stacked = numpy.stack(
            [
                self._corr.evaluate(eta, "nom").astype(numpy.float32),
                self._corr.evaluate(eta, "up").astype(numpy.float32),
                self._corr.evaluate(eta, "down").astype(numpy.float32),
            ],
            axis=1,
        )
        return awkward.Array(stacked)


class _JsonPogJUNC:
    """JES uncertainty sources from a JSON-POG file.

    ``getUncertainty`` returns ``[(name, (n_jets, 2) array)]`` where column
    0 is the up factor ``1+δ`` and column 1 is the down factor ``1-δ``,
    matching the convention expected by ``CorrectedJetsFactory.build()``.
    """

    signature = ("JetEta", "JetPt")

    def __init__(self, source_pairs):
        # source_pairs: list of (str, correctionlib.Correction) tuples
        self._sources = source_pairs

    def getUncertainty(self, JetEta, JetPt):
        eta = numpy.asarray(JetEta, dtype=numpy.float32)
        pt  = numpy.asarray(JetPt,  dtype=numpy.float32)
        result = []
        for name, corr in self._sources:
            delta    = corr.evaluate(eta, pt).astype(numpy.float32)
            unc_vals = awkward.Array(
                numpy.stack([1.0 + delta, 1.0 - delta], axis=1)
            )
            result.append((name, unc_vals))
        return result


class _JsonPogJECStack:
    """Minimal JECStack replacement that holds JSON-POG backed evaluators.

    Only the four attributes accessed by ``FixedCorrectedJetsFactory.build()``
    are needed: ``jec``, ``jer``, ``jersf``, ``junc``.
    """

    def __init__(self, jec=None, jer=None, jersf=None, junc=None):
        self.jec   = jec
        self.jer   = jer
        self.jersf = jersf
        self.junc  = junc


def _rand_gauss(event, count, phi, eta, rng: Squares):
    # construct bit-128 counter from [event[64], phi[32], eta[32]]
    counter = numpy.empty((len(phi), 2), dtype=numpy.uint64)
    counter[:, 0] = numpy.repeat(event, count)
    counter[:, 1] = np.round(numpy.asarray(phi), 3).view(numpy.uint32)
    counter[:, 1] <<= 32
    counter[:, 1] |= np.round(numpy.asarray(eta), 3).view(numpy.uint32)
    rand = rng.normal(counter).astype(numpy.float32)

    def getfunction(layout, depth):
        if isinstance(layout, awkward.layout.NumpyArray) or not isinstance(
            layout, (awkward.layout.Content, awkward.partition.PartitionedArray)
        ):
            return lambda: awkward.layout.NumpyArray(rand)
        return None

    out = awkward._util.recursively_apply(
        awkward.operations.convert.to_layout(phi), getfunction
    )
    assert out is not None
    return awkward._util.wrap(out, awkward._util.behaviorof(phi))


class FixedCorrectedJetsFactory(CorrectedJetsFactory):
    def build(
        self,
        jets,
        event,
        seeds: Iterable[str | int] = ("JER",),
        lazy_cache=None,
    ):
        ########## Patch start ##########
        if lazy_cache is None:
            lazy_cache = Cache(math.inf)
        ##########  Patch end  ##########
        lazy_cache = awkward._util.MappingProxy.maybe_wrap(lazy_cache)
        if not isinstance(jets, awkward.highlevel.Array):
            raise Exception("'jets' must be an awkward > 1.0.0 array of some kind!")
        fields = awkward.fields(jets)
        if len(fields) == 0:
            raise Exception(
                "Empty record, please pass a jet object with at least {self.real_sig} defined!"
            )
        out = awkward.flatten(jets)
        wrap = partial(awkward_rewrap, like_what=jets, gfunc=rewrap_recordarray)
        scalar_form = awkward.without_parameters(
            out[self.name_map["ptRaw"]]
        ).layout.form

        in_dict = {field: out[field] for field in fields}
        out_dict = dict(in_dict)

        # take care of nominal JEC (no JER if available)
        out_dict[self.name_map["JetPt"] + "_orig"] = out_dict[self.name_map["JetPt"]]
        out_dict[self.name_map["JetMass"] + "_orig"] = out_dict[
            self.name_map["JetMass"]
        ]
        if self.treat_pt_as_raw:
            out_dict[self.name_map["ptRaw"]] = out_dict[self.name_map["JetPt"]]
            out_dict[self.name_map["massRaw"]] = out_dict[self.name_map["JetMass"]]

        jec_name_map = dict(self.name_map)
        jec_name_map["JetPt"] = jec_name_map["ptRaw"]
        jec_name_map["JetMass"] = jec_name_map["massRaw"]
        if self.jec_stack.jec is not None:
            jec_args = {
                k: out_dict[jec_name_map[k]] for k in self.jec_stack.jec.signature
            }
            out_dict["jet_energy_correction"] = self.jec_stack.jec.getCorrection(
                **jec_args, form=scalar_form, lazy_cache=lazy_cache
            )
        else:
            out_dict["jet_energy_correction"] = awkward.without_parameters(
                awkward.ones_like(out_dict[self.name_map["JetPt"]])
            )

        # finally the lazy binding to the JEC
        init_pt = partial(
            awkward.virtual,
            operator.mul,
            args=(out_dict["jet_energy_correction"], out_dict[self.name_map["ptRaw"]]),
            cache=lazy_cache,
        )
        init_mass = partial(
            awkward.virtual,
            operator.mul,
            args=(
                out_dict["jet_energy_correction"],
                out_dict[self.name_map["massRaw"]],
            ),
            cache=lazy_cache,
        )

        out_dict[self.name_map["JetPt"]] = init_pt(length=len(out), form=scalar_form)
        out_dict[self.name_map["JetMass"]] = init_mass(
            length=len(out), form=scalar_form
        )

        out_dict[self.name_map["JetPt"] + "_jec"] = out_dict[self.name_map["JetPt"]]
        out_dict[self.name_map["JetMass"] + "_jec"] = out_dict[self.name_map["JetMass"]]

        # in jer we need to have a stash for the intermediate JEC products
        has_jer = False
        if self.jec_stack.jer is not None and self.jec_stack.jersf is not None:
            has_jer = True
            jer_name_map = dict(self.name_map)
            jer_name_map["JetPt"] = jer_name_map["JetPt"] + "_jec"
            jer_name_map["JetMass"] = jer_name_map["JetMass"] + "_jec"

            jerargs = {
                k: out_dict[jer_name_map[k]] for k in self.jec_stack.jer.signature
            }
            out_dict["jet_energy_resolution"] = self.jec_stack.jer.getResolution(
                **jerargs, form=scalar_form, lazy_cache=lazy_cache
            )

            jersfargs = {
                k: out_dict[jer_name_map[k]] for k in self.jec_stack.jersf.signature
            }
            out_dict["jet_energy_resolution_scale_factor"] = (
                self.jec_stack.jersf.getScaleFactor(
                    **jersfargs, form=_JERSF_FORM, lazy_cache=lazy_cache
                )
            )
            ########## Patch start ##########
            out_dict["jet_resolution_rand_gauss"] = awkward.virtual(
                _rand_gauss,
                args=(
                    event,
                    awkward.num(jets, axis=1),
                    out_dict["phi"],
                    out_dict[self.name_map["JetEta"]],
                    Squares(seeds),
                ),
                cache=lazy_cache,
                length=len(out),
                form=scalar_form,
            )
            ##########  Patch end  ##########
            init_jerc = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    0,
                    self.forceStochastic,
                    out_dict[jer_name_map["ptGenJet"]],
                    out_dict[jer_name_map["JetPt"]],
                    out_dict[jer_name_map["JetEta"]],
                    out_dict["jet_energy_resolution"],
                    out_dict["jet_resolution_rand_gauss"],
                    out_dict["jet_energy_resolution_scale_factor"],
                ),
                cache=lazy_cache,
            )
            out_dict["jet_energy_resolution_correction"] = init_jerc(
                length=len(out), form=scalar_form
            )

            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    out_dict["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetPt"]],
                ),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    out_dict["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetMass"]],
                ),
                cache=lazy_cache,
            )
            out_dict[self.name_map["JetPt"]] = init_pt_jer(
                length=len(out), form=scalar_form
            )
            out_dict[self.name_map["JetMass"]] = init_mass_jer(
                length=len(out), form=scalar_form
            )

            out_dict[self.name_map["JetPt"] + "_jer"] = out_dict[self.name_map["JetPt"]]
            out_dict[self.name_map["JetMass"] + "_jer"] = out_dict[
                self.name_map["JetMass"]
            ]

            # JER systematics
            jerc_up = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    1,
                    self.forceStochastic,
                    out_dict[jer_name_map["ptGenJet"]],
                    out_dict[jer_name_map["JetPt"]],
                    out_dict[jer_name_map["JetEta"]],
                    out_dict["jet_energy_resolution"],
                    out_dict["jet_resolution_rand_gauss"],
                    out_dict["jet_energy_resolution_scale_factor"],
                ),
                cache=lazy_cache,
            )
            up = awkward.flatten(jets)
            up["jet_energy_resolution_correction"] = jerc_up(
                length=len(out), form=scalar_form
            )
            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    up["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetPt"]],
                ),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    up["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetMass"]],
                ),
                cache=lazy_cache,
            )
            up[self.name_map["JetPt"]] = init_pt_jer(length=len(out), form=scalar_form)
            up[self.name_map["JetMass"]] = init_mass_jer(
                length=len(out), form=scalar_form
            )

            jerc_down = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    2,
                    self.forceStochastic,
                    out_dict[jer_name_map["ptGenJet"]],
                    out_dict[jer_name_map["JetPt"]],
                    out_dict[jer_name_map["JetEta"]],
                    out_dict["jet_energy_resolution"],
                    out_dict["jet_resolution_rand_gauss"],
                    out_dict["jet_energy_resolution_scale_factor"],
                ),
                cache=lazy_cache,
            )
            down = awkward.flatten(jets)
            down["jet_energy_resolution_correction"] = jerc_down(
                length=len(out), form=scalar_form
            )
            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    down["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetPt"]],
                ),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    down["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetMass"]],
                ),
                cache=lazy_cache,
            )
            down[self.name_map["JetPt"]] = init_pt_jer(
                length=len(out), form=scalar_form
            )
            down[self.name_map["JetMass"]] = init_mass_jer(
                length=len(out), form=scalar_form
            )
            out_dict["JER"] = awkward.zip(
                {"up": up, "down": down}, depth_limit=1, with_name="JetSystematic"
            )

        if self.jec_stack.junc is not None:
            juncnames = {}
            juncnames.update(self.name_map)
            if has_jer:
                juncnames["JetPt"] = juncnames["JetPt"] + "_jer"
                juncnames["JetMass"] = juncnames["JetMass"] + "_jer"
            else:
                juncnames["JetPt"] = juncnames["JetPt"] + "_jec"
                juncnames["JetMass"] = juncnames["JetMass"] + "_jec"
            juncargs = {
                k: out_dict[juncnames[k]] for k in self.jec_stack.junc.signature
            }
            juncs = self.jec_stack.junc.getUncertainty(**juncargs)

            def junc_smeared_val(uncvals, up_down, variable):
                return awkward.materialized(uncvals[:, up_down] * variable)

            def build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, updown):
                var_dict = dict(in_dict)
                var_dict[jetpt] = awkward.virtual(
                    junc_smeared_val,
                    args=(
                        unc,
                        updown,
                        jetpt_orig,
                    ),
                    length=len(out),
                    form=scalar_form,
                    cache=lazy_cache,
                )
                var_dict[jetmass] = awkward.virtual(
                    junc_smeared_val,
                    args=(
                        unc,
                        updown,
                        jetmass_orig,
                    ),
                    length=len(out),
                    form=scalar_form,
                    cache=lazy_cache,
                )
                return awkward.zip(
                    var_dict,
                    depth_limit=1,
                    parameters=out.layout.parameters,
                    behavior=out.behavior,
                )

            def build_variant(unc, jetpt, jetpt_orig, jetmass, jetmass_orig):
                up = build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, 0)
                down = build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, 1)
                return awkward.zip(
                    {"up": up, "down": down}, depth_limit=1, with_name="JetSystematic"
                )

            for name, func in juncs:
                out_dict[f"jet_energy_uncertainty_{name}"] = func
                out_dict[f"JES_{name}"] = build_variant(
                    func,
                    self.name_map["JetPt"],
                    out_dict[juncnames["JetPt"]],
                    self.name_map["JetMass"],
                    out_dict[juncnames["JetMass"]],
                )

        out_parms = out.layout.parameters
        out_parms["corrected"] = True
        out = awkward.zip(
            out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior
        )

        return wrap(out)
