import openpnm.models as mods
import properties as prpts

D = prpts.properties["D"]
mu = prpts.properties["mu"]
macropore = {
    "pore.diffusivity": {
        "model": mods.misc.constant,
        "value": D,
    },
    "throat.diffusivity": {
        "model": mods.misc.constant,
        "value": D,
    },
    "pore.viscosity": {
        "model": mods.misc.constant,
        "value": mu,
    },
    "throat.viscosity": {
        "model": mods.misc.constant,
        "value": mu,
    },   
}

