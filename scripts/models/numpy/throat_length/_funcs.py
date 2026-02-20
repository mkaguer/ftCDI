
def spheres_and_cylinders(
    network,
    pore_diameter='pore.diameter',
    throat_diameter='throat.diameter'
):
    r"""
    Finds throat length assuming pores are spheres and throats are
    cylinders.

    Parameters
    ----------
    %(network)s
    %(Dp)s
    %(Dt)s

    Returns
    -------
    lengths : ndarray
        A numpy ndarray containing throat length values

    """
    from models.numpy import conduit_lengths
    out = conduit_lengths.spheres_and_cylinders(
        network=network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    )
    return out[:, 1]


def continuum(
    network,
    pore_diameter='pore.diameter',
    throat_diameter='throat.diameter'
):
    r"""
    Finds throat length assuming pores are spheres and throats are
    cylinders.

    Parameters
    ----------
    %(network)s
    %(Dp)s
    %(Dt)s

    Returns
    -------
    lengths : ndarray
        A numpy ndarray containing throat length values

    """
    from models.numpy import conduit_lengths
    out = conduit_lengths.cubes_and_cuboids(
        network=network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    )
    return out.sum(axis=1)


def intersecting_trapezoidals(network,
                              pore_diameter="pore.diameter",
                              throat_diameter="throat.diameter"):
    r"""
    Calculates throat length assuming pores are intersecting
    trapezoidals

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
    from models.numpy import conduit_lengths
    L = conduit_lengths.intersecting_trapezoidals(network=network)
    return L[:, 1]
