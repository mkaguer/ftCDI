import jax.numpy as jnp

__all__ = ["cylinder",
           "intersecting_trapezoidals",
           "effective"]


def cylinder(
    network,
    pore_diameter='pore.diameter',
    pore_thickness='pore.thickness',
):
    r"""
    Calculate pore volume from diameter assuming a cylindrical body

    Parameters
    ----------
    %(network)s
    %(Dp)s

    Returns
    -------

    """
    Dp = network[pore_diameter]
    w = network[pore_thickness]
    return jnp.pi * (Dp/2) ** 2 * w


def intersecting_trapezoidals(
    network,
    pore_diameter='pore.diameter',
    throat_diameter='throat.diameter',
    throat_thickness='throat.thickness'
):
    r"""
    Calculate pore volume assuming intersecting trapezoidals

    Parameters
    ----------
    %(network)s
    %(Dp)s

    Returns
    -------

    """
    Np = len(network["pore.coords"])
    # get connecting pores 1 and 2
    conns = network["throat.conns"]
    P1 = conns[:, 0]
    P2 = conns[:, 1]
    # get diameters
    Dt = network["throat.diameter"]
    Dp = network["pore.diameter"][conns]
    # get thickness
    w = network["throat.thickness"]
    # get lengths
    coords = network["pore.coords"]
    t_coords = network["throat.coords"]
    h1 = jnp.linalg.norm(t_coords - coords[P1], axis=1)
    h2 = jnp.linalg.norm(t_coords - coords[P2], axis=1)
    # get volume
    x = jnp.zeros(Np)
    jnp.add.at(x, P1, (Dp[:, 0] + Dt)/2*h1*w)
    jnp.add.at(x, P2, (Dp[:, 1] + Dt)/2*h2*w)
    
    return x


def effective(
    network,
    pore_volume='pore.volume',
    throat_volume='throat.volume',
    factor=1/2,
):
    r"""
    Calculate the effective pore volume for optional use in transient
    simulations. The effective pore volume is calculated by adding half
    the volume of all neighbouring throats to the pore volume.

    Parameters
    ----------
    %(network)s
    %(Dp)s
    %(Dt)s

    Returns
    -------

    """
    cn = network['throat.conns']
    P1 = cn[:, 0]
    P2 = cn[:, 1]
    eff_vol = jnp.copy(network[pore_volume])
    jnp.add.at(eff_vol, P1, factor*network[throat_volume])
    jnp.add.at(eff_vol, P2, factor*network[throat_volume])
    return eff_vol
