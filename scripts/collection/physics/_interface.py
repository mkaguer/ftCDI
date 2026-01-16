import openpnm.models as mods
import models

interface = {
    "throat.mass_conductance": {
        "model": models.misc.generic_diffusive,
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
