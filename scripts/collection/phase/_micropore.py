import openpnm.models as mods
import properties as props

Km = props.properties["Km"]
rho_mi = props.properties["rho_mi"]
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
}
