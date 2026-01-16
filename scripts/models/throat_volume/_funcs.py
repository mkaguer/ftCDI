import numpy as np

__all__ = ["continuum",
           "lens",
           "zero_volume"]

def continuum(network,
              throat_diameter="throat.diameter",
              throat_length="throat.length"):
    
    D = network[throat_diameter]
    L = network[throat_length]
    vol = D ** 2 * L
    
    return vol


def lens(
    network,
    throat_diameter='throat.diameter',
    pore_diameter='pore.diameter'):
    r"""
    NOTE: I wrote this custom for micropores where Rt > Rp
    
    Calculates the volume residing the hemispherical caps formed by the
    intersection between cylindrical throats and spherical pores.

    This volume should be subtracted from throat volumes if the throat lengths
    were found using throat end points.

    Parameters
    ----------
    %(network)s
    %(Dt)s
    %(Dp)s

    Returns
    -------

    Notes
    -----
    This model does not consider the possibility that multiple throats might
    overlap in the same location which could happen if throats are large and
    connectivity is random.

    See Also
    --------
    pendular_ring
    """
    conns = network['throat.conns']
    Rp = network[pore_diameter]/2
    Rt = network[throat_diameter]/2
    a = np.atleast_2d(Rt).T
    # ensure that Rt/Rp does not exceed 1!
    ratio = a/Rp[conns]  # Nt by 2
    mask = ratio > 1
    ratio[mask] = 1
    q = np.arcsin(ratio)
    b = Rp[conns]*np.cos(q)
    h = Rp[conns] - b
    V = 1/6*np.pi*h*(3*a**2 + h**2)
    return np.sum(V, axis=1)


def zero_volume(network):
    r"""
    Assigns effectively zero volume to throat

    Parameters
    ----------
    %(network)s

    Returns
    -------
    throat volume
    """
    Nt = len(network["throat.conns"])
    value = np.ones(Nt) * 1e-32
    return value
