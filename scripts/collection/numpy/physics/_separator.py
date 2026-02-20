import openpnm.models as mods
import models.numpy as models

separator = {
    "throat.diffusive_conductance": {
        "model": mods.physics.diffusive_conductance.generic_diffusive,
        "pore_diffusivity": "pore.diffusivity",
        "throat_diffusivity": "throat.diffusivity",
        "size_factors": "throat.diffusive_size_factors",
    },
    "throat.hydraulic_conductance": {
        "model": mods.physics.hydraulic_conductance.generic_hydraulic,
        "pore_viscosity": "pore.viscosity",
        "throat_viscosity": "throat.viscosity",
        "size_factors": "throat.hydraulic_size_factors",
    },
    "throat.mass_conductance": {
        "model": mods.physics.ad_dif_conductance.ad_dif,
        "pore_pressure": "pore.pressure",
        "throat_hydraulic_conductance": "throat.hydraulic_conductance",
        "throat_diffusive_conductance": "throat.diffusive_conductance",
        "s_scheme": "powerlaw",
    },
    "pore.conductivity": {
        "model": models.misc.conductivity,
        "concentration": "pore.concentration_old",
        "diffusivity": "pore.diffusivity",
        "temperature": "pore.temperature",
    },
    "throat.conductivity": {
        "model": models.misc.conductivity,
        "concentration": "throat.concentration_old",
        "diffusivity": "throat.diffusivity",
        "temperature": "throat.temperature",
    },
    "throat.ionic_conductance": {
        "model": mods.physics.diffusive_conductance.generic_diffusive,
        "pore_diffusivity": "pore.conductivity",
        "throat_diffusivity": "throat.conductivity",
        "size_factors": "throat.diffusive_size_factors",
    },
}
