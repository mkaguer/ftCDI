from models.numpy.conduit_lengths._conduit_lengths import _get_L_ctc
import models.numpy.conduit_lengths as _conduit_lengths
import numpy as np

__all__ = ["spheres_and_cylinders",
           "cubes_and_cuboids",
           "continuum",
           "no_flow",
           "intersecting_cylinders"]


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
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split('.', 1)[-1]).T
    L1, Lt, L2 = _conduit_lengths.spheres_and_cylinders(
        network=network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    ).T

    # Fi is the integral of (1/A^2) dx, x = [0, Li]
    a = 4 / (D1**3 * np.pi**2)
    b = 2 * D1 * L1 / (D1**2 - 4 * L1**2) + np.arctanh(2 * L1 / D1)
    F1 = a * b
    a = 4 / (D2**3 * np.pi**2)
    b = 2 * D2 * L2 / (D2**2 - 4 * L2**2) + np.arctanh(2 * L2 / D2)
    F2 = a * b
    Ft = Lt / (np.pi / 4 * Dt**2)**2

    # I is the integral of (y^2 + z^2) dA, divided by A^2
    I1 = I2 = It = 1 / (2 * np.pi)

    # S is 1 / (16 * pi^2 * I * F)
    S1 = 1 / (16 * np.pi**2 * I1 * F1)
    St = 1 / (16 * np.pi**2 * It * Ft)
    S2 = 1 / (16 * np.pi**2 * I2 * F2)

    return np.vstack([S1, St, S2]).T


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
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split('.', 1)[-1]).T
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
    S1 = 1 / (16 * np.pi**2 * I1 * F1)
    St = 1 / (16 * np.pi**2 * It * Ft)
    S2 = 1 / (16 * np.pi**2 * I2 * F2)

    return np.vstack([S1, St, S2]).T


def continuum(
    network,
    throat_diameter="throat.diameter"
):
    r"""
    This model calculates the size factor for a conduit assuming one throat
    with square cross-section that is constant across it's length. The pores
    in the conduit are assumed to have negligeable resistance. We can use this
    size factor for both hydraulic and diffusive size factors in a continuum.
    It is simply A/L!
    """
    # get Nt
    Nt = len(network['throat.conns'])

    # get throat diameter and length
    D = network[throat_diameter]
    L = _get_L_ctc(network)

    # calculate area
    A = D ** 2

    # calculate size factors
    vals = np.vstack([np.ones(Nt)*1e16, A/L, np.ones(Nt)*1e16]).T

    return vals


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
    F1 = np.ones(network.Nt) * 1e32
    F2 = np.ones(network.Nt) * 1e32
    Ft = np.ones(network.Nt) * 1e32

    # S is 1 / (16 * pi^2 * I * F)
    S1 = 1 / F1
    St = 1 / Ft
    S2 = 1 / F2

    return np.vstack([S1, St, S2]).T


def intersecting_cylinders(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
):
    r"""
    Computes hydraulic size factors assuming intersecting cylinders

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
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split('.', 1)[-1]).T
    L1, Lt, L2 = _conduit_lengths.intersecting_cylinders(network).T

    # hageon-pousielle conductance
    S1 = np.pi * (D1/2) ** 4 / 8 / L1
    St = np.ones_like(S1) * 1e32
    S2 = np.pi * (D2/2) ** 4 / 8 / L2

    return np.vstack([S1, St, S2]).T
