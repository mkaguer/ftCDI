import numpy as np

__all__ = ["intersecting_trapezoidals"]

def intersecting_trapezoidals(network, pore_diameter='pore.diameter'):
    r"""
    Finds throat diameter of intersecting trapezoidals

    Parameters
    ----------
    network : dict
        The network dictionary
    pore_diameter : str
        The dictionary key used to fetch pore diameter

    Returns
    -------
    Dt : ndarray
        A numpy ndarray containing throat diameter values

    """
    # get coords
    coords = network["pore.coords"]
    t_coords = network["throat.coords"]
    # get connecting pores 1 and 2
    conns = network["throat.conns"]
    P1 = conns[:, 0]
    P2 = conns[:, 1]
    # get diameter of connecting pores
    D1 = network[pore_diameter][P1]
    D2 = network[pore_diameter][P2]
    # get distance between pores
    L = np.linalg.norm(coords[P2] - coords[P1], axis=1)
    # calculate Dt
    x = np.linalg.norm(t_coords - coords[P1], axis=1)
    Dt = (D2 - D1)/L * x + D1
    
    return Dt