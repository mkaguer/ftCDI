import jax.numpy as jnp

__all__ = ["generic_hydraulic"]

def generic_hydraulic(
    network,
    pore_viscosity='pore.viscosity',
    throat_viscosity='throat.viscosity',
    size_factors='throat.hydraulic_size_factors'
):
    r"""
    Calculates the hydraulic conductance of conduits in network.

    Parameters
    ----------
    %(phase)s
    pore_viscosity : str
        %(dict_blurb)s pore viscosity
    throat_viscosity : str
        %(dict_blurb)s throat viscosity
    size_factors : str
        %(dict_blurb)s conduit hydraulic size factors

    Returns
    -------
    %(return_arr)s hydraulic conductance

    """
    conns = network['throat.conns']
    mu1, mu2 = network[pore_viscosity][conns].T
    mut = network[throat_viscosity]

    SF = network[size_factors]
    if isinstance(SF, dict):  # Legacy approach
        F1, Ft, F2 = SF.values()
    elif SF.ndim > 1:  # Nt-by-3 array
        F1, Ft, F2 = SF.T
    else:  # Nt array, like from network extraction predictions
        F1, Ft, F2 = jnp.inf, SF, jnp.inf

    g1 = F1 / mu1
    gt = Ft / mut
    g2 = F2 / mu2
    
    G = 1 / (1/g1 + 1/gt + 1/g2)
    
    return G