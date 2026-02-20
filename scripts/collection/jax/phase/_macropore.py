import pnmlib.models as mods
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
    "throat.concentration": {
        "model": mods.misc.from_neighbor_pores,
        "prop": "pore.concentration",
        "mode": "mean",
    },
    "throat.temperature": {
        "model": mods.misc.from_neighbor_pores,
        "prop": "pore.temperature",
        "mode": "mean",
    },
}
