import numpy as np
import models.conduit_lengths as _conduit_lengths
from models.conduit_lengths._conduit_lengths import _get_L_ctc

__all__ = ["spheres_and_cylinders",
           "cubes_and_cuboids",
           "continuum",
           "intersecting_trapezoidals",
           "cylinders_and_trapezoidals",
           "intersecting_cylinders",
           "custom",
           "custom2"]


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
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split('.', 1)[1]).T
    L1, Lt, L2 = _conduit_lengths.spheres_and_cylinders(
        network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    ).T

    # Fi is the integral of (1/A) dx, x = [0, Li]
    F1 = 2 / (D1 * np.pi) * np.arctanh(2 * L1 / D1)
    F2 = 2 / (D2 * np.pi) * np.arctanh(2 * L2 / D2)
    Ft = Lt / (np.pi / 4 * Dt ** 2)

    vals = np.vstack([1/F1, 1/Ft, 1/F2]).T
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
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split('.', 1)[-1]).T
    L1, Lt, L2 = _conduit_lengths.cubes_and_cuboids(
        network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    ).T

    # Fi is the integral of (1/A) dx, x = [0, Li]
    F1 = L1 / D1**2
    F2 = L2 / D2**2
    Ft = Lt / Dt**2

    vals = np.vstack([1/F1, 1/Ft, 1/F2]).T
    return vals


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


def intersecting_trapezoidals(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
    throat_thickness="throat.thickness"
):
    r"""
    Computes diffusive shape coefficient for intersecting trapezoidals. This
    size factor was written for custom use by Mike McKague

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
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split('.', 1)[-1]).T
    L1, Lt, L2 = _conduit_lengths.intersecting_trapezoidals(network).T
    # get thickness
    w = network["throat.thickness"]
    # Fi is the integral of (1/A) dx, x = [0, Li]
    F1 = L1 / (Dt - D1) / w * np.log(Dt/D1)
    F2 = L2 / (Dt - D2) / w * np.log(Dt/D2)
    Ft = np.ones_like(F1) * 1e-32

    vals = np.vstack([1/F1, 1/Ft, 1/F2]).T
    return vals


def cylinders_and_trapezoidals(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
    throat_thickness="throat.thickness"
):
    r"""
    Computes diffusive shape coefficient for trapezoidal intersecting with a 
    cylinder. This size factor was written for custom use by Mike McKague.

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
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split('.', 1)[-1]).T
    L1, Lt, L2 = _conduit_lengths.cylinders_and_trapezoidals(network,
                                                             pore_diameter).T

    # get thickness
    w = network["throat.thickness"]

    # Fi is the integral of (1/A) dx, x = [0, Li]
    F1 = 1/2/w*np.arcsin(2*L1/D1)
    F2 = L2 / (Dt - D2) / w * np.log(Dt/D2)  # TODO: assumes that trap is P2!
    Ft = np.ones_like(F1) * 1e-32

    vals = np.vstack([1/F1, 1/Ft, 1/F2]).T
    return vals


def intersecting_cylinders(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
):
    r"""
    Computes diffusive shape coefficient for intersecting cylinder elements.
    This is a custom model written by Mike McKague.

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
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split('.', 1)[-1]).T
    L1, Lt, L2 = _conduit_lengths.intersecting_cylinders(network).T

    # Fi is the integral of (1/A) dx, x = [0, Li]
    F1 = 4 * L1 / np.pi / D1**2
    F2 = 4 * L2 / np.pi / D2**2
    Ft = np.ones_like(F1) * 1e-32

    vals = np.vstack([1/F1, 1/Ft, 1/F2]).T
    return vals


def custom(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
    throat_zeta="throat.zeta",
):
    r"""
    Computes diffusive shape coefficient for conduits assuming pores are
    spheres and throats are cylinders. It is custom because it uses a 
    characteristic length of reaction/diffusion problems.

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
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split('.', 1)[1]).T
    L1, Lt, L2 = _conduit_lengths.spheres_and_cylinders(
        network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    ).T
    
    # adjust for characteristic length
    zeta = network[throat_zeta]
    Lt = zeta * Lt

    # Fi is the integral of (1/A) dx, x = [0, Li]
    F1 = 2 / (D1 * np.pi) * np.arctanh(2 * L1 / D1)
    F2 = 2 / (D2 * np.pi) * np.arctanh(2 * L2 / D2)
    Ft = Lt / (np.pi / 4 * Dt ** 2)

    vals = np.vstack([1/F1, 1/Ft, 1/F2]).T
    return vals


def custom2(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter",
    throat_zeta="throat.zeta",
):
    r"""
    Computes diffusive shape coefficient for conduits assuming pores are
    spheres and throats are cylinders. It is custom because it uses a 
    characteristic length of reaction/diffusion problems.

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
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split('.', 1)[1]).T
    L1, Lt, L2 = _conduit_lengths.spheres_and_cylinders(
        network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    ).T
    
    # adjust for characteristic length
    zeta = network[throat_zeta]
    Lt = zeta * Lt

    # Fi is the integral of (1/A) dx, x = [0, Li]
    F1 = 2 / (D1 * np.pi) * np.arctanh(2 * L1 / D1)
    F2 = 2 / (D2 * np.pi) * np.arctanh(2 * L2 / D2)
    Ft = Lt / (np.pi / 4 * Dt ** 2)

    vals = np.vstack([1/F1, 1/Ft, np.ones(network.Nt)*1e32]).T
    return vals