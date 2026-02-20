import jax.numpy as jnp
import jax.experimental.sparse as js
from mypnmlib.generators import cubic

__all__ = ["make_cubic_network",
           "create_adjacency_matrix",
           "graph_laplacian"]

def make_cubic_network(shape, spacing=1, connectivity=6):
    '''
    Create custom cubic network that returns dictionary of JAX arrays.
    '''
    network = cubic(shape=shape,
                    spacing=spacing,
                    connectivity=connectivity,
                    node_prefix='pore',
                    edge_prefix='throat')
    network = label_faces_cubic(network, rtol=0.0)
    return network


def label_faces_cubic(network, rtol=0.0):
    r"""
    Label the nodes sitting on the faces of the domain assuming the domain
    is cubic

    Parameters
    ----------
    network : dict
        The network dictionary contain 'node.coords'
    rtol : float
        Controls how closely a node must be to a face to be counted. It is
        computed relative to the fraction of domain size, as:
        ``hi_label = abs(1 - x[i]/x.max()) < rtol`` and
        ``lo_label = abs(x[i]/x.max()) < rtol``

    Returns
    -------
    network : dict
        The network dictionary with the face labels added

    """
    node_prefix = get_node_prefix(network)
    coords = network[node_prefix + '.coords']
    dims = dimensionality(network, cache=False)
    coords = jnp.around(coords, decimals=10)
    min_labels = ['left', 'front', 'bottom']
    max_labels = ['right', 'back', 'top']
    min_coords = jnp.amin(coords, axis=0)
    max_coords = jnp.amax(coords, axis=0)
    for ax in jnp.where(dims)[0]:
        network[node_prefix + '.' + min_labels[ax]] = \
            jnp.abs((coords[:, ax] - min_coords[ax]) / max_coords[ax]) <= rtol
        network[node_prefix + '.' + max_labels[ax]] = \
            jnp.abs(1 - coords[:, ax] / max_coords[ax]) <= rtol
    return network


def dimensionality(network, cache=True):
    r"""
    Checks the dimensionality of the network

    Parameters
    ----------
    network : dict
        The network dictionary
    cache : boolean, optional (default is True)
        If ``False`` then the dimensionality is recalculated even if it has
        already been calculated and stored in the graph dictionary.

    Returns
    -------
    dims : list
        A 3-by-1 array containing ``True`` for each axis that contains
        multiple values, indicating that the pores are spatially distributed
        in that dimension.

    """
    if cache:
        try:
            return network["params.dimensionality"]
        except (KeyError, AttributeError):
            pass
    n = get_node_prefix(network)
    coords = network[n + '.coords']
    eps = jnp.finfo(float).resolution
    dims_unique = [
        not jnp.allclose(xk, xk.mean(), atol=0, rtol=eps) for xk in coords.T
    ]
    if cache:
        network["params.dimensionality"] = jnp.array(dims_unique)
    return jnp.array(dims_unique)


def get_node_prefix(network):
    r"""
    Determines the prefix used for node arrays from ``<edge_prefix>.coords``

    Parameters
    ----------
    network : dict
        The network dictionary

    Returns
    -------
    node_prefix : str
        The value of ``<node_prefix>`` used in ``g``.  This is found by
        scanning ``g.keys()`` until an array ending in ``'.coords'`` is found,
        then returning the prefix.

    Notes
    -----
    This process is surprisingly fast, on the order of nanoseconds, so this
    overhead is worth it for the flexibility it provides in array naming.
    However, since all ``dict`` are now sorted in Python, it may be helpful
    to ensure the ``'conns'`` array is near the beginning of the list.
    """
    for item in network.keys():
        if item.endswith('.coords'):
            return item.split('.')[0]


def create_adjacency_matrix(net, weights=None, fmt='coo', triu=False, drop_zeros=False):
    """
    Generates a weighted adjacency matrix in the desired sparse format using JAX.

    Parameters
    ----------
    weights : array_like, optional
        An array containing the throat values to enter into the matrix
        (in graph theory these are known as the 'weights').
        - If `weights` is `None`, all connections will have weight 1.
        - If `weights` has length Nt, it is assumed the matrix is symmetric.
        - If `weights` has length 2*Nt, the first Nt values are for the upper
          triangle, and the second Nt for the lower triangle.
    fmt : str, optional
        The sparse storage format to return. Options are:
            - `'coo'`: Coordinate format (default).
            - `'csr'`: Compressed Sparse Row format.
    triu : bool, default is False
        If True, only the upper-triangular part is included (ignored if weights
        has length 2*Nt).
    drop_zeros : bool, default is False
        If True, entries with zero weights are removed.

    Returns
    -------
    A JAX sparse adjacency matrix in the specified format.
    """
    # get Nt and Np
    Nt = len(net['throat.conns'])
    Np = len(net['pore.coords'])

    allowed_weights = [(Nt,), (2 * Nt,), (Nt, 2)]

    # Handle weights
    if weights is None:
        weights = jnp.ones((Nt,), dtype=jnp.float32)
    elif weights.shape not in allowed_weights:
        raise ValueError("Received weights are of incorrect length")

    # Extract throat connections
    conn = net['throat.conns']
    row, col = conn[:, 0], conn[:, 1]

    # Modify rows and columns for symmetric or triangular matrices
    if weights.shape == (2 * Nt,):
        row = jnp.concatenate([row, conn[:, 1]])
        col = jnp.concatenate([col, conn[:, 0]])
    elif weights.shape == (Nt, 2):
        row = jnp.concatenate([row, conn[:, 1]])
        col = jnp.concatenate([col, conn[:, 0]])
        weights = weights.flatten(order='F')
    elif not triu:
        row = jnp.concatenate([row, conn[:, 1]])
        col = jnp.concatenate([col, conn[:, 0]])
        weights = jnp.concatenate([weights, weights])

    # Eliminate zeros if needed
    if drop_zeros:
        mask = weights != 0
        row, col, weights = row[mask], col[mask], weights[mask]

    # Shape of the adjacency matrix
    shape = (Np, Np)

    # Create the adjacency matrix in COO format
    coo_matrix = js.BCOO((weights, jnp.stack([row, col]).T), shape=shape)

    # Convert to desired format
    if fmt == 'coo':
        return coo_matrix
    elif fmt == 'csr':
        return js.BCSR.from_bcoo(coo_matrix)
    else:
        raise ValueError(f"Format {fmt} is not supported for JAX sparse matrices")


def graph_laplacian(adj_matrix, fmt='csr'):
    """
    Compute the graph Laplacian for a JAX sparse adjacency matrix.

    Parameters:
    adj_matrix (js.BCOO): JAX sparse adjacency matrix.

    fmt : str, optional
        The sparse storage format to return. Options are:
            - `'coo'`: Coordinate format (default).
            - `'csr'`: Compressed Sparse Row format.

    Returns:
    laplacian (js.BCOO or js.BCSR): The Laplacian matrix.
    """
    # Compute the degree vector by summing the adjacency matrix along columns
    diag = adj_matrix.sum(axis=0).todense()  # should by Np long!

    # Create the degree matrix in sparse form (diagonal matrix)
    rows = jnp.arange(len(diag))
    diag_matrix = js.BCOO((diag, jnp.stack([rows, rows]).T), shape=adj_matrix.shape)

    # Laplacian: D - A
    laplacian = diag_matrix - adj_matrix

    # Convert to desired format
    if fmt == 'coo':
        return laplacian
    elif fmt == 'csr':
        return js.BCSR.from_bcoo(laplacian)
    else:
        raise ValueError(f"Format {fmt} is not supported for JAX sparse matrices")