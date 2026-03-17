from pnmlib.models.geometry.conduit_lengths import _conduit_lengths
from pnmlib.models.misc._misc import _get_conduit_data
import jax.numpy as jnp

__all__ = ["spheres_and_cylinders",
           "cubes_and_cuboids",
           "no_flow"]

def spheres_and_cylinders(network,
                          pore_diameter="pore.diameter",
                          throat_diameter="throat.diameter"):
    r"""
    Computes hydraulic size factors for conduits assuming pores are
    spheres and throats are cylinders.

    Parameters
    ----------
    %(networkwork)s
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
    The hydraulic size factor is the geometrical part of the pre-factor in
    Stoke's flow:

    .. math::

        Q = \frac{A^2}{8 \pi \mu L} \Delta P
          = \frac{S_{hydraulic}}{\mu} \Delta P

    Thus :math:`S_{hydraulic}` represents the combined effect of the area
    and length of the *conduit*, which consists of a throat and 1/2 of the
    pores on each end.
    """
    D1, Dt, D2 = _get_conduit_data(network, pore_diameter.split('.', 1)[-1]).T
    L1, Lt, L2 = _conduit_lengths.spheres_and_cylinders(
        network=network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    ).T
    
    # Fi is the integral of (1/A^2) dx, x = [0, Li]
    a = 4 / (D1**3 * jnp.pi**2)
    b = 2 * D1 * L1 / (D1**2 - 4 * L1**2) + jnp.arctanh(2 * L1 / D1)
    F1 = a * b
    a = 4 / (D2**3 * jnp.pi**2)
    b = 2 * D2 * L2 / (D2**2 - 4 * L2**2) + jnp.arctanh(2 * L2 / D2)
    F2 = a * b
    Ft = Lt / (jnp.pi / 4 * Dt**2)**2

    # I is the integral of (y^2 + z^2) dA, divided by A^2
    I1 = I2 = It = 1 / (2 * jnp.pi)

    # S is 1 / (16 * pi^2 * I * F)
    S1 = 1 / (16 * jnp.pi**2 * I1 * F1)
    St = 1 / (16 * jnp.pi**2 * It * Ft)
    S2 = 1 / (16 * jnp.pi**2 * I2 * F2)
    
    return jnp.vstack([S1, St, S2]).T


def cubes_and_cuboids(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
    pore_aspect=[1, 1, 1],
    throat_aspect=[1, 1, 1],
):
    r"""
    Computes hydraulic size factors for conduits assuming pores are cubes
    and throats are cuboids.

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

    """
    D1, Dt, D2 = _get_conduit_data(network, pore_diameter.split('.', 1)[-1]).T
    L1, Lt, L2 = _conduit_lengths.cubes_and_cuboids(
        network=network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    ).T

    # Fi is the integral of (1/A^2) dx, x = [0, Li]
    F1 = L1 / D1**4
    F2 = L2 / D2**4
    Ft = Lt / Dt**4

    # I is the integral of (y^2 + z^2) dA, divided by A^2
    I1 = I2 = It = 1 / 6

    # S is 1 / (16 * pi^2 * I * F)
    S1 = 1 / (16 * jnp.pi**2 * I1 * F1)
    St = 1 / (16 * jnp.pi**2 * It * Ft)
    S2 = 1 / (16 * jnp.pi**2 * I2 * F2)

    return jnp.vstack([S1, St, S2]).T


def no_flow(network):
    r"""
    Computes hydraulic size factors assuming no flow can enter conduit. Hence,
    these sie factors should be very very small!

    Parameters
    ----------
    %(network)s

    Returns
    -------
    size_factors : ndarray
        Array (Nt by 3) containing conduit values for each element
        of the pore-throat-pore conduits. The array is formatted as
        ``[pore1, throat, pore2]``.
    """
    # Fi is the integral of (1/A^2) dx, x = [0, Li]
    F1 = 1e32
    F2 = 1e32
    Ft = 1e32

    # S is 1 / (16 * pi^2 * I * F)
    S1 = 1 / F1
    St = 1 / Ft
    S2 = 1 / F2

    return jnp.vstack([S1, St, S2]).T
