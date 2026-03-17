import jax.numpy as jnp
from pnmlib.models.misc._misc import _get_conduit_data
from pnmlib.models.misc._misc import _get_L_ctc

__all__ = ["spheres_and_cylinders",
           "cubes_and_cuboids"]

def spheres_and_cylinders(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
):
    r"""
    Calculates conduit lengths in the network assuming pores are spheres
    and throats are cylinders. No overlapping is assumed!

    A conduit is defined as ( 1/2 pore - full throat - 1/2 pore ).

    Parameters
    ----------
    %(network)s
    %(Dp)s
    %(Dt)s

    Returns
    -------
    lengths : ndarray
        Array (Nt by 3) containing conduit values for each element
        of the pore-throat-pore conduits. The array is formatted as
        ``[pore1, throat, pore2]``.

    """
    L_ctc = _get_L_ctc(network)
    D1, Dt, D2 = _get_conduit_data(network, pore_diameter.split(".", 1)[-1]).T
    
    # If spheres do not overlap:
    # considers case where D1 <= Dt
    L1 = jnp.where(D1 > Dt, jnp.sqrt(D1**2 - Dt**2) / 2, 1e-32)
    L2 = jnp.where(D2 > Dt, jnp.sqrt(D1**2 - Dt**2) / 2, 1e-32)
    Lt = L_ctc - (L1 + L2)

    return jnp.vstack((L1, Lt, L2)).T


def cubes_and_cuboids(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
):
    r"""
    Calculates conduit lengths in the network assuming pores are spheres
    and throats are cylinders. No overlapping is assumed!

    A conduit is defined as ( 1/2 pore - full throat - 1/2 pore ).

    Parameters
    ----------
    %(network)s
    %(Dp)s
    %(Dt)s

    Returns
    -------
    lengths : ndarray
        Array (Nt by 3) containing conduit values for each element
        of the pore-throat-pore conduits. The array is formatted as
        ``[pore1, throat, pore2]``.

    """
    L_ctc = _get_L_ctc(network)
    D1, Dt, D2 = _get_conduit_data(network, pore_diameter.split(".", 1)[-1]).T

    # For cubes/cuboids:
    L1 = D1/2
    L2 = D2/2
    Lt = L_ctc - (L1 + L2)

    return jnp.vstack((L1, Lt, L2)).T
