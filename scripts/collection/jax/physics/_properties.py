import models.jax as mods

properties = {
    "pore.donnan_potential": {
        "model": mods.misc.donnan_potential,
        "pore_potential": "pore.potential",
        "pore_electrode_potential": "pore.electrode_potential",
        "pore_attraction_term": "pore.attraction_term",
        "pore_temperature": "pore.temperature",
        "pore_capacitance": "pore.capacitance",
        "pore_surface_area": "pore.surface_area",
        "pore_concentration": "pore.concentration_old",
    },
    "pore.micro_concentration": {
        "model": mods.misc.micropore_concentration,
        "pore_concentration": "pore.concentration_old",
        "pore_donnan_potential": "pore.donnan_potential",
        "pore_temperature": "pore.temperature",
        "pore_attraction_term": "pore.attraction_term",
    },
}
