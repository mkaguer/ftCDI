import models.jax as mods

sources = {
    "pore.mass_source": {
        "model": mods.misc.mass_source,
        "pore_micro_concentration": "pore.micro_concentration",
        "pore_micro_concentration_old": "pore.micro_concentration_old",
        "pore_micro_volume": "pore.micro_volume",
        "time_step": "time_step",
    },
    "pore.charge_source": {
        "model": mods.misc.charge_source,
        "pore_donnan_potential": "pore.donnan_potential",
        "pore_donnan_potential_old": "pore.donnan_potential_old",
        "pore_volume": "pore.effective_volume",
        "pore_capacitance": "pore.capacitance",
        "pore_surface_area": "pore.surface_area",
        "time_step": "time_step",
    },
    "pore.mass_source_effective": {
        "model": mods.misc.mass_source_effective,
        "pore_mass_source": "pore.mass_source",
        "pore_concentration": "pore.concentration_old",
        "pore_volume": "pore.effective_volume",
        "time_step": "time_step"
    },
}
