import pnmlib.models as mods
import properties as props

Km = props.properties["Km"]
rho_mi = 1 # props.properties["rho_mi"]
micropore = {
    "pore.diffusivity": {
        "model": mods.misc.constant,
        "value": 1e32,
    },
    "throat.diffusivity": {
        "model": mods.misc.constant,
        "value": Km*rho_mi,
    },
    "pore.viscosity": {
        "model": mods.misc.constant,
        "value": 1e32,
    },
    "throat.viscosity": {
        "model": mods.misc.constant,
        "value": 1e32,
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
