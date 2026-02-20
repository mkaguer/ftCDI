import jax.numpy as jnp


__all__ = ["cylinder",
           "cuboid",
           "lens",
           "zero_volume"]


def cylinder(
    network,
    throat_diameter='throat.diameter',
    throat_length='throat.length',
):
    r"""
    Calculate throat volume assuing a cylindrical shape

    Parameters
    ----------
    %(network)s
    %(Dt)s
    %(Lt)s

    Returns
    -------
    volumes : ndarray
        A numpy ndarray containing throat volume values

    Notes
    -----
    This models does not account for the volume reprsented by the
    intersection of the throat with a spherical pore body.  Use the ``lens``
    or ``pendular_ring`` models in addition to this one to account for this
    volume.

    """
    leng = network[throat_length]
    diam = network[throat_diameter]
    value = jnp.pi/4*leng*diam**2
    return value


def cuboid(
    network,
    throat_diameter='throat.diameter',
    throat_length='throat.length',
):
    r"""
    Calculate throat volume assuing a square cross-section

    Parameters
    ----------
    %(network)s
    %(Dt)s
    %(Lt)s

    Returns
    -------

    Notes
    -----
    At present this models does NOT account for the volume reprsented by the
    intersection of the throat with a spherical pore body.

    """
    leng = network[throat_length]
    diam = network[throat_diameter]
    value = leng*diam**2
    return value


def lens(network,
         throat_diameter='throat.diameter',
         pore_diameter='pore.diameter'):
    r"""
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
    a = jnp.atleast_2d(Rt).T
    r = jnp.where(a/Rp[conns] > 1, 1, a/Rp[conns])  # case where r > 1
    q = jnp.arcsin(r)
    b = Rp[conns]*jnp.cos(q)
    h = Rp[conns] - b
    V = 1/6*jnp.pi*h*(3*a**2 + h**2)
    return jnp.sum(V, axis=1)


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
    value = jnp.ones(Nt) * 1e-32
    return value
