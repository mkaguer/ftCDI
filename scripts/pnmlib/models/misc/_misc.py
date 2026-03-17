import jax.numpy as jnp
import jax.experimental.sparse as sprs

__all__ = ["difference",
           "constant",
           "scaled",
           "fraction",
           "from_neighbor_pores",
           "conductivity",
           "find_neighbor_throats",
           "from_neighbor_throats",
           "_get_conduit_data",
           "_get_L_ctc",
           "_poisson_conductance"]


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


def from_neighbor_throats(network, prop, mode='min', ignore_nans=True):
    r"""
    Adopt a value from the values found in neighboring throats

    Parameters
    ----------
    target : Base
        The object which this model is associated with. This controls the
        length of the calculated array, and also provides access to other
        necessary properties.
    prop : str
        The dictionary key of the array containing the throat property to be
        used in the calculation.
    mode : str
        Controls how the pore property is calculated. The default value is
        'min'. Options are:

            ===========  =====================================================
            mode         meaning
            ===========  =====================================================
            'min'        Returns the value of the minimum property of the
                         neighboring throats
            'max'        Returns the value of the maximum property of the
                         neighboring throats
            'mean'       Returns the value of the mean property of the
                         neighboring throats
            'sum'        Returns the sum of the property of the neighboring
                         throats
            ===========  =====================================================

    Returns
    -------
    value : ndarray
        Array containing customized values based on those of adjacent throats.

    """
    Np = len(network["pore.coords"])
    Nt = len(network["throat.conns"])
    data = network[prop]
    nans = jnp.isnan(data)
    im = _create_incidence_matrix(network)
    if mode == 'min':
        if ignore_nans:
            data[nans] = jnp.inf
        values = jnp.ones((Np, ))*jnp.inf
        jnp.minimum.at(values, im.row, data[im.col])
    if mode == 'max':
        if ignore_nans:
            data[nans] = -jnp.inf
        values = jnp.ones((network.Np, ))*-jnp.inf
        jnp.maximum.at(values, im.row, data[im.col])
    if mode == 'mean':
        if ignore_nans:
            data = data.at[nans].set(0)
        values = jnp.zeros((Np, ))
        values = jnp.add.at(values, im.row, data[im.col], inplace=False)
        counts = jnp.zeros((Np, ))
        counts = jnp.add.at(counts, im.row, jnp.ones((Nt, ))[im.col], inplace=False)
        if ignore_nans:
            counts = jnp.subtract.at(counts, im.row, nans[im.col], inplace=False)
        values = values/counts
    if mode == 'sum':
        if ignore_nans:
            data[nans] = 0
        values = jnp.zeros((network.Np, ))
        jnp.add.at(values, im.row, data[im.col])
    return values


def _create_incidence_matrix(network, weights=None, fmt='coo',
                            drop_zeros=False):
    r"""
    Creates a weighted incidence matrix in the desired sparse format

    Parameters
    ----------
    weights : array_like, optional
        An array containing the throat values to enter into the matrix
        (in graph theory these are known as the 'weights'). If
        omitted, ones are used to create a standard incidence matrix
        representing connectivity only.
    fmt : str, default is 'coo'
        The sparse storage format to return. Options are:
            **'coo'** : This is the native format of OpenPNM's data
            **'lil'** : Enables row-wise slice of the matrix
            **'csr'** : Favored by most linear algebra routines
            **'dok'** : Enables subscript access of locations
    drop_zeros : bool, default is ``False``
        If ``True``, applies the ``eliminate_zeros`` method of the sparse
        array to remove all zero locations.

    Returns
    -------
    sparse_array
        An incidence matrix in the specified sparse format

    Notes
    -----
    The incidence matrix is a cousin to the adjacency matrix, and used
    by OpenPNM for finding the throats connected to a give pore or set
    of pores. Specifically, an incidence matrix has Np rows and Nt
    columns, and each row represents a pore, containing non-zero
    values at the locations corresponding to the indices of the
    throats connected to that pore. The ``weights`` argument indicates
    what value to place at each location, with the default being 1's
    to simply indicate connections. Another useful option is throat
    indices, such that the data values on each row indicate which
    throats are connected to the pore, though this is redundant as it
    is identical to the locations of non-zeros.

    Examples
    --------
    >>> import openpnm as op
    >>> pn = op.network.Cubic(shape=[5, 5, 5])
    >>> weights = np.random.rand(pn.num_throats(), ) < 0.5
    >>> im = pn.create_incidence_matrix(weights=weights, fmt='csr')

    """
    Np = len(network["pore.coords"])
    Nt = len(network["throat.conns"])
    # Check if provided data is valid
    if weights is None:
        weights = jnp.ones((Nt,), dtype=int)
    if jnp.shape(weights)[0] == Nt:
        weights = jnp.append(weights, weights)
    elif jnp.shape(weights)[0] != 2*Nt:
        raise Exception('Received dataset of incorrect length')

    conn = network['throat.conns']
    row = conn[:, 1]
    row = jnp.append(row, conn[:, 0])
    col = jnp.arange(Nt)
    col = jnp.append(col, col)

    # temp = sprs.coo.coo_matrix((weights, (row, col)), (Np, Nt))
    # indices = jnp.stack([row, col], axis=1)
    temp = sprs.COO((weights, row, col), shape=(Np, Nt))

    if drop_zeros:
        temp.eliminate_zeros()

    # Convert to requested format
    if fmt == 'coo':
        pass  # temp is already in coo format
    elif fmt == 'csr':
        temp = temp.tocsr()
    elif fmt == 'lil':
        temp = temp.tolil()
    elif fmt == 'dok':
        temp = temp.todok()

    return temp


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


def _get_L_ctc(network):
    """Returns throat spacing if it exists, otherwise calculates it."""
    try:
        L_ctc = network["throat.spacing"]
    except KeyError:
        P12 = network["throat.conns"]
        C1 = network["pore.coords"][P12[:, 0]]
        C2 = network["pore.coords"][P12[:, 1]]
        L_ctc = jnp.linalg.norm(C1 - C2, axis=1)


def _poisson_conductance(network,
                         pore_conductivity=None,
                         throat_conductivity=None,
                         size_factors=None):
    r"""
    Calculates the conductance of the conduits in the network, where a
    conduit is (1/2 pore - full throat - 1/2 pore). See the notes section.

    Parameters
    ----------
    phase : OpenPNM Phase
        The object which this model is associated with. This controls the
        length of the calculated array, and also provides access to other
        necessary properties.
    pore_conductivity : str
        Dictionary key of the pore conductivity values
    throat_conductivity : str
        Dictionary key of the throat conductivity values
    size_factors: str
        Dictionary key of the conduit size factors' values.

    Returns
    -------
    g : ndarray
        Array containing conductance values for conduits in the
        geometry attached to the given physics object.

    Notes
    -----
    This function requires that all the necessary phase properties
    already be calculated.

    """
    cn = network["throat.conns"]
    Dt = network[throat_conductivity]
    D1, D2 = network[pore_conductivity][cn].T
    # If individual size factors for conduit constiuents are known
    SF = network[size_factors]
    if SF.ndim == 2:
        F1, Ft, F2 = SF.T
        g1 = D1 * F1
        gt = Dt * Ft
        g2 = D2 * F2
        return 1 / (1 / g1 + 1 / gt + 1 / g2)
    else:
        # Otherwise, i.e., the size factor for the entire conduit is only known
        F = network[size_factors]
        return Dt * F
