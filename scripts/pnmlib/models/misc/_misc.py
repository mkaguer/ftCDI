import jax.numpy as jnp

__all__ = ["difference",
           "constant",
           "scaled",
           "fraction",
           "from_neighbor_pores",
           "conductivity",
           "find_neighbor_throats"]


def difference(network, props):
    r"""
    Subtracts elements 1:N in `props` from element 0

    Parameters
    ----------
    network : dict
        The network dictionary
    props : list
        A list of dict keys containing the values to operate on.  If the first
        element is A, and the next are B and C, then the results is A - B - C.
    """
    A = network[props[0]]
    for B in props[1:]:
        A = A - network[B]
    return A


def constant(network, value):
    r"""
    Places the given constant value into the target object

    Parameters
    ----------
    target : Base
        The object which this model is associated with. This controls the
        length of the calculated array, and also provides access to other
        necessary properties.
    value : scalar
        The numerical value to apply

    Returns
    -------
    value : ndarray
        Array containing constant values equal to ``value``.

    Notes
    -----
    This model is mostly useless and for testing purposes, but might be used
    to 'reset' an array back to a default value.

    """
    return value


def scaled(network, prop, factor):
    r"""
    Scales an existing value by a factor.

    Useful for constricting some throat property.

    Parameters
    ----------
    target : Base
        The object which this model is associated with. This controls the
        length of the calculated array, and also provides access to other
        necessary properties.
    prop : str
        The dictionary key of the array containing the values to be scaled.
    factor : str
        The factor by which the values should be scaled.

    Returns
    -------
    value : ndarray
        Array containing ``target[prop]`` values scaled by ``factor``.

    """
    value = network[prop]*factor
    return value


def fraction(target, numerator, denominator):
    r"""
    Calculates the ratio between two values

    Parameters
    ----------
    target : Base
        The object which this model is associated with. This controls the
        length of the calculated array, and also provides access to other
        necessary properties.
    numerator : str
        Dictionary key pointing the numerator values
    denominator : str
        Dictionary key pointing the denominator values

    """
    x = target[numerator]
    y = target[denominator]
    return x/y


def from_neighbor_pores(network, prop, mode='min'):
    r"""
    Adopt a value based on the values in neighboring pores

    Parameters
    ----------
    target : Base
        The object which this model is associated with. This controls the
        length of the calculated array, and also provides access to other
        necessary properties.
    prop : str
        The dictionary key to the array containing the pore property to be
        used in the calculation.
    mode : str
        Controls how the pore property is calculated. The default value is
        'min'. Options are:

            ===========  =====================================================
            mode         meaning
            ===========  =====================================================
            'min'        Returns the value of the minimum property of the
                         neighboring pores
            'max'        Returns the value of the maximum property of the
                         neighboring pores
            'mean'       Returns the value of the mean property of the
                         neighboring pores
            'sum'        Returns the sum of the property of the neighrboring
                         pores
            ===========  =====================================================

    ignore_nans : bool (default is ``True``)
        If ``True`` the result will ignore ``nans`` in the neighbors

    Returns
    -------
    value : ndarray
        Array containing customized values based on those of adjacent pores.

    """
    P12 = network["throat.conns"]
    pvalues = network[prop][P12]
    if mode == 'min':
        value = jnp.amin(pvalues, axis=1)
    if mode == 'max':
        value = jnp.amax(pvalues, axis=1)
    if mode == 'mean':
        value = jnp.mean(pvalues, axis=1)
    if mode == 'sum':
        value = jnp.sum(pvalues, axis=1)
    return jnp.array(value)


def conductivity(
    phase,
    concentration="pore.concentration",
    diffusivity="pore.diffusivity",
    temperature="pore.temperature"):
    
    R = 8.31446262  # J/mol/K
    F = 96485.3399  # C/mol
    c = phase[concentration]
    T = phase[temperature]
    D = phase[diffusivity]
    
    K = 2*F**2*c*D/R/T
    
    return K


def _get_conduit_data(network, propname):
    r"""
    Fetches an Nt-by-3 array of the requested property

    Parameters
    ----------
    propname : str
        The dictionary key of the property to fetch.

    Returns
    -------
    data : ndarray
        An Nt-by-3 array with each column containing the requrested data
        for pore1, throat, and pore2 respectively.

    """
    poreprop = 'pore.' + propname.split('.', 1)[-1]
    throatprop = 'throat.' + propname.split('.', 1)[-1]
    conns = network['throat.conns']
    T = network[throatprop]
    P1, P2 = network[poreprop][conns.T]

    vals = jnp.vstack((P1, T, P2)).T

    return vals


def find_neighbor_throats(network, pores):
    """
    pores must be an array of indices
    """
    conns = network['throat.conns']
    throats = jnp.any(jnp.isin(conns, pores), axis=1)

    return throats
