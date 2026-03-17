from pnmlib.models.misc._misc import _poisson_conductance
import jax.numpy as jnp

__all__ = ["generic_diffusive"]

def generic_diffusive(network,
                      pore_diffusivity="pore.diffusivity",
                      throat_diffusivity="throat.diffusivity",
                      size_factors="throat.diffusive_size_factors"):
    r"""
    Calculates the diffusive conductance of conduits in network.

    Parameters
    ----------
    %(phase)s
    pore_diffusivity : str
        %(dict_blurb)s pore diffusivity
    throat_diffusivity : str
        %(dict_blurb)s throat diffusivity
    size_factors : str
        %(dict_blurb)s conduit diffusive size factors

    Returns
    -------
    %(return_arr)s diffusive conductance

    """
    G = _poisson_conductance(network=network,
                                pore_conductivity=pore_diffusivity,
                                throat_conductivity=throat_diffusivity,
                                size_factors=size_factors)
    return jnp.vstack((G, G)).T 