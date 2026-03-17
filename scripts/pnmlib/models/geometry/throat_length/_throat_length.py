from pnmlib.models.geometry.conduit_lengths import _conduit_lengths

__all__ = ["spheres_and_cylinders",
           "cubes_and_cuboids"]

def spheres_and_cylinders(network,
                          pore_diameter='pore.diameter',
                          throat_diameter='throat.diameter'):
    r"""
    Finds throat length assuming pores are spheres and throats are
    cylinders.

    Parameters
    ----------
    network : dict
        The network dictionary
    pore_diameter : str
        The dictionary key used to fetch pore diameter
    throat_diameter : str
        The dictionary key used to fetch throat diameter

    Returns
    -------
    lengths : ndarray
        A numpy ndarray containing throat length values

    """
    L = _conduit_lengths.spheres_and_cylinders(network=network,
                                               pore_diameter=pore_diameter,
                                               throat_diameter=throat_diameter)
    return L[:, 1]


def cubes_and_cuboids(network,
                      pore_diameter="pore.diameter",
                      throat_diameter="throat.diameter"):
    r"""
    Calculates throat length assuming pores are spheres
    and throats are cylinders.

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
    L = _conduit_lengths.cubes_and_cuboids(network=network,
                                           pore_diameter=pore_diameter,
                                           throat_diameter=throat_diameter)
    return L[:, 1]