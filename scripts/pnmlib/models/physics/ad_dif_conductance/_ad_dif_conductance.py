import jax.numpy as jnp

__all__ = ["ad_dif"]

def ad_dif(
    network,
    pore_pressure='pore.pressure',
    throat_hydraulic_conductance='throat.hydraulic_conductance',
    throat_diffusive_conductance='throat.diffusive_conductance',
    s_scheme='powerlaw'
):
    r"""
    Calculates the advective-diffusive conductance of conduits in network.

    Parameters
    ----------
    %(phase)s
    pore_pressure : str
        %(dict_blurb)s pore pressure
    throat_hydraulic_conductance : str
        %(dict_blurb)s hydraulic conductance
    throat_diffusive_conductance : str
        %(dict_blurb)s throat diffusive conductance
    s_scheme : str
        Name of the space discretization scheme to use

    Returns
    -------
    %(return_arr)s advection-diffuvsion conductance values

    Notes
    -----
    This function calculates the specified property for the *entire*
    network then extracts the values for the appropriate throats at the
    end.

    This function assumes cylindrical throats with constant cross-section
    area. Corrections for different shapes and variable cross-section area
    can be imposed by passing the proper conduit_shape_factors argument
    when computing the diffusive and hydraulic conductances.

    shape_factor depends on the physics of the problem, i.e.
    diffusion-like processes and fluid flow need different shape factors.

    """
    cn = network['throat.conns']
    # get Nt
    Nt = len(network["throat.conns"])
    # Find g for half of pore 1, throat, and half of pore 2
    P = network[pore_pressure]
    gh = network[throat_hydraulic_conductance]
    gd = network[throat_diffusive_conductance]
    if gd.size == Nt:
        gd = jnp.tile(gd, 2)
    # Special treatment when gd is not Nt by 1 (ex. mass partitioning)
    elif gd.size == 2 * Nt:
        gd = gd.reshape(Nt * 2, order='F')
    else:
        raise Exception(f"Shape of {throat_diffusive_conductance} must either"
                        r" be (Nt,1) or (Nt,2)")

    Qij = -gh * jnp.diff(P[cn], axis=1).squeeze()
    Qij = jnp.append(Qij, -Qij)

    Peij = Qij / gd
    # FIXME: only necessary for exponential and maybe powerlaw
    # maybe switch model based on peclet number, but really I need to check Pe!
    # But could probably find way to comment out these lines
    # Peij = Peij.at[(Peij < 1e-10) & (Peij >= 0)].set(1e-10)
    # Peij = Peij.at[(Peij > -1e-10) & (Peij <= 0)].set(-1e-10)

    # Correct the flow rate
    Qij = Peij * gd

    if s_scheme == 'upwind':
        w = gd + jnp.maximum(0, -Qij)
    elif s_scheme == 'hybrid':
        w = jnp.maximum(0, jnp.maximum(-Qij, gd - Qij / 2))
    elif s_scheme == 'powerlaw':
        w = gd * jnp.maximum(0, (1 - 0.1 * jnp.absolute(Peij))**5) + \
            jnp.maximum(0, -Qij)
    elif s_scheme == 'exponential':
        w = -Qij / (1 - jnp.exp(Peij))
    else:
        raise Exception('Unrecognized discretization scheme: ' + s_scheme)
    w = w.reshape(Nt, 2, order='F')
    return w
