import jax.numpy as jnp
from pnmlib.models.misc._misc import _get_conduit_data

__all__ = ["spheres_and_cylinders",
           "cubes_and_cuboids",
           "intersecting_cylinders"]

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
    L2 = jnp.where(D2 > Dt, jnp.sqrt(D2**2 - Dt**2) / 2, 1e-32)
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


def intersecting_trapezoidals(network):
    r"""
    Calculates conduit lengths in the network assuming intersecting trapezoids

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
    # get coords
    coords = network["pore.coords"]
    t_coords = network["throat.coords"]
    # get connecting pores 1 and 2
    conns = network["throat.conns"]
    P1 = conns[:, 0]
    # get ctc length
    L_ctc = _get_L_ctc(network)

    # For intersecting trapezoidals:
    L1 = jnp.linalg.norm(t_coords - coords[P1], axis=1)
    L2 = L_ctc - L1
    Lt = jnp.ones_like(L1) * 1e-32

    return jnp.vstack((L1, Lt, L2)).T


def intersecting_cylinders(network):
    r"""
    Calculates conduit lengths in the network assuming intersecting cylinders.
    Note this happens to be the same as intersecting_trapezoidals!

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
    # get coords
    coords = network["pore.coords"]
    t_coords = network["throat.coords"]
    # get connecting pores 1 and 2
    conns = network["throat.conns"]
    P1 = conns[:, 0]
    # get ctc length
    L_ctc = _get_L_ctc(network)

    # For intersecting cylinders:
    L1 = jnp.linalg.norm(t_coords - coords[P1], axis=1)
    L2 = L_ctc - L1
    Lt = jnp.ones_like(L1) * 1e-32

    return jnp.vstack((L1, Lt, L2)).T


def cylinders_and_trapezoidals(network,
                               pore_diameter="pore.diameter"):
    r"""
    Calculates conduit lengths in the network assuming intersecting cylinders.
    Note this happens to be the same as intersecting_trapezoidals!

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
    # get connecting pores 1 and 2
    conns = network["throat.conns"]
    P1 = conns[:, 0]
    # get diameter of pore 1
    D1 = network[pore_diameter][P1]
    # get ctc length
    L_ctc = _get_L_ctc(network)

    # For intersecting trapezoidals:
    L1 = D1/2
    L2 = L_ctc - L1
    Lt = jnp.ones_like(L1) * 1e-32

    return jnp.vstack((L1, Lt, L2)).T


def _get_L_ctc(network):
    """Returns throat spacing if it exists, otherwise calculates it."""
    try:
        L_ctc = network["throat.spacing"]
    except KeyError:
        P12 = network["throat.conns"]
        C1 = network["pore.coords"][P12[:, 0]]
        C2 = network["pore.coords"][P12[:, 1]]
        L_ctc = jnp.linalg.norm(C1 - C2, axis=1)
    return L_ctc