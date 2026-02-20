from jax.numpy import pi as _pi

__all__ = ["sphere",
           "cube"]


def sphere(
    network,
    pore_diameter='pore.diameter'
):
    r"""
    Calculate pore volume from diameter assuming a spherical pore body

    Parameters
    ----------
    %(network)s
    %(Dp)s

    Returns
    -------
    volumes : ndarray
        Numpy ndarray containing pore volume values

    """
    return 4/3*_pi*(network[pore_diameter]/2)**3


def cube(
    network,
    pore_diameter='pore.diameter'
):
    r"""
    Calculate pore volume from diameter assuming a cubic pore body

    Parameters
    ----------
    %(network)s
    %(Dp)s

    Returns
    -------

    """
    return network[pore_diameter]**3
