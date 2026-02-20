import openpnm.models as mods
import models.numpy as models
import properties as prpts

D = prpts.properties["D"]
rho_sep = prpts.properties["rho_sep"]
K = prpts.properties["K"]
mu = prpts.properties["mu"]
separator = {
    "pore.effective_diffusivity": {
        "model": models.misc.effective_diffusivity,
        "D": D,
        "epsilon": rho_sep,
        "tau": 1/rho_sep**(0.5),
    },
    "throat.effective_diffusivity": {
        "model": models.misc.effective_diffusivity,
        "D": D,
        "epsilon": rho_sep,
        "tau": 1/rho_sep**(0.5),
    },
    "pore.diffusivity": {
        "model": mods.misc.scaled,
        "prop": "pore.effective_diffusivity",
        "factor": rho_sep,
    },
    "throat.diffusivity": {
        "model": mods.misc.scaled,
        "prop": "throat.effective_diffusivity",
        "factor": rho_sep,
    },
    "pore.permeability": {
        "model": mods.misc.constant,
        "value": K,
    },
    "throat.permeability": {
        "model": mods.misc.constant,
        "value": K,
    },
    "pore.viscosity_water": {
        "model": mods.misc.constant,
        "value": mu,
    },
    "throat.viscosity_water": {
        "model": mods.misc.constant,
        "value": mu,
    },
    "pore.viscosity": {
        "model": mods.misc.fraction,
        "numerator": "pore.viscosity_water",
        "denominator": "pore.permeability",
    },
    "throat.viscosity": {
        "model": mods.misc.fraction,
        "numerator": "throat.viscosity_water",
        "denominator": "throat.permeability",
    },
}

