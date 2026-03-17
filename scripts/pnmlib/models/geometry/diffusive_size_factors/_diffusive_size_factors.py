from pnmlib.models.geometry.conduit_lengths import _conduit_lengths
from pnmlib.models.misc._misc import _get_conduit_data
import jax.numpy as jnp

__all__ = ["spheres_and_cylinders",
           "cubes_and_cuboids"]

def spheres_and_cylinders(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
):
    r"""
    Computes diffusive shape coefficient for conduits assuming pores are
    spheres and throats are cylinders.

    Parameters
    ----------
    %(network)s
    %(Dp)s
    %(Dt)s

    Returns
    -------
    size_factors : ndarray
        Array (Nt by 3) containing conduit values for each element
        of the pore-throat-pore conduits. The array is formatted as
        ``[pore1, throat, pore2]``.

    Notes
    -----
    The diffusive size factor is the geometrical part of the pre-factor in
    Fick's law:

    .. math::

        n_A = \frac{A}{L} \Delta C_A
            = S_{diffusive} D_{AB} \Delta C_A

    Thus :math:`S_{diffusive}` represents the combined effect of the area and
    length of the *conduit*, which consists of a throat and 1/2 of the pore
    on each end.

    """
    D1, Dt, D2 = _get_conduit_data(network, pore_diameter.split('.', 1)[1]).T
    L1, Lt, L2 = _conduit_lengths.spheres_and_cylinders(
        network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    ).T

    # Fi is the integral of (1/A) dx, x = [0, Li]
    F1 = 2 / (D1 * jnp.pi) * jnp.arctanh(2 * L1 / D1)
    F2 = 2 / (D2 * jnp.pi) * jnp.arctanh(2 * L2 / D2)
    Ft = Lt / (jnp.pi / 4 * Dt ** 2)

    vals = jnp.vstack([1/F1, 1/Ft, 1/F2]).T
    return vals


def cubes_and_cuboids(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
    pore_aspect=[1, 1, 1],
    throat_aspect=[1, 1, 1],
):
    r"""
    Computes diffusive shape coefficient for conduits assuming pores are
    cubes and throats are cuboids.

    Parameters
    ----------
    %(network)s
    %(Dp)s
    %(Dt)s
    pore_aspect : list
        Aspect ratio of the pores
    throat_aspect : list
        Aspect ratio of the throats

    Returns
    -------

    Notes
    -----
    This model should only be used for true 2D networks, i.e. with planar
    symmetry.

    """
    D1, Dt, D2 = _get_conduit_data(network, pore_diameter.split('.', 1)[-1]).T
    L1, Lt, L2 = _conduit_lengths.cubes_and_cuboids(
        network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    ).T

    # Fi is the integral of (1/A) dx, x = [0, Li]
    F1 = L1 / D1**2
    F2 = L2 / D2**2
    Ft = Lt / Dt**2

    vals = jnp.vstack([1/F1, 1/Ft, 1/F2]).T
    return vals