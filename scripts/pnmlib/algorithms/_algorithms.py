import jax
import jax.numpy as jnp
import jax.experimental.sparse as js
from pnmlib.network import create_adjacency_matrix, graph_laplacian
from pnmlib.models import misc

__all__ = ["build_A",
           "build_b",
           "set_BC",
           "apply_BC",
           "solve",
           "rate",
           "set_outflow_BC",
           "set_source",
           "apply_sources",
           "update_A_and_b"]

def build_A(proj,
            alg):
    
    # get algorithm
    algorithm = proj[alg]
    # get network
    network = proj["network"]
    # get conductance
    throat_conductance = algorithm["conductance"]
    # retrieve conductance
    g = network[throat_conductance]
    # create adjacency matrix
    am = create_adjacency_matrix(network, weights=g, fmt='coo')
    # make laplacian
    A = graph_laplacian(am, fmt='coo')

    return A


def build_b(proj,
            alg):
    
    # get network
    network = proj["network"]
    # get number of pores
    Np = len(network["pore.coords"])
    # create b as array of zeros Np long
    b = jnp.zeros(Np, dtype=float)

    return b


def set_BC(proj,
           alg,
           pores,
           bctype,
           bcvalues,
           mode="overwrite"):
    
    # get algorithm
    algorithm = proj[alg]
    # get network
    network = proj["network"]
    # get Np
    Np = len(network["pore.coords"])
    # set 'value' BCs
    if bctype == "value":
        if mode == "overwrite":
            algorithm["pore.bc.value"] = jnp.zeros(Np, dtype=float)
            algorithm["pore.bc.value_mask"] = jnp.zeros(Np, dtype=bool)
            algorithm["pore.bc.value"] = algorithm["pore.bc.value"].at[pores].set(bcvalues)
            algorithm["pore.bc.value_mask"] = algorithm["pore.bc.value_mask"].at[pores].set(True)
        elif mode == "add":
            algorithm["pore.bc.value"] = algorithm["pore.bc.value"].at[pores].set(bcvalues)
            algorithm["pore.bc.value_mask"] = algorithm["pore.bc.value_mask"].at[pores].set(True)
        else:
            raise ValueError(f"{mode} is not a supported mode")
    elif bctype == "outflow":
         if mode == "overwrite":
            algorithm["pore.bc.outflow"] = jnp.zeros(Np, dtype=float)
            algorithm["pore.bc.outflow_mask"] = jnp.zeros(Np, dtype=bool)
            algorithm["pore.bc.outflow"] = algorithm["pore.bc.outflow"].at[pores].set(bcvalues)
            algorithm["pore.bc.outflow_mask"] = algorithm["pore.bc.outflow_mask"].at[pores].set(True)
         elif mode == "add":
            algorithm["pore.bc.outflow"] = algorithm["pore.bc.outflow"].at[pores].set(bcvalues)
            algorithm["pore.bc.outflow_mask"] = algorithm["pore.bc.outflow_mask"].at[pores].set(True)
    else:
        raise ValueError(f"{bctype} is not a supported bctype")


def apply_BC(proj,
             alg):

    """Applies specified boundary conditions by modifying A and b."""
    # get algorithm
    algorithm = proj[alg]
    # get network
    network = proj["network"]
    # get pure A and pure b from networkwork
    A = algorithm["A"]
    b = algorithm["b"]
    # get Nt and Np
    Np = len(network["pore.coords"])
    Nt = len(network["throat.conns"])
    # apply rate BC to b
    if 'pore.bc.rate' in algorithm.keys():
        ind = jnp.isfinite(algorithm['pore.bc.rate'])
        b = b.at[ind].set(-algorithm['pore.bc.rate'][ind])  # negative for production
    # apply value BC to A and b
    if 'pore.bc.value' in algorithm.keys():
        # get average of diagonal, note this only works for 'coo' format
        diag = A.data[0:Np]  # this only works for 'coo' format
        f = diag.mean()
        # Update b (impose bc values)
        bc_values = algorithm['pore.bc.value']
        bc_mask = algorithm['pore.bc.value_mask']
        b = jnp.where(bc_mask, bc_values*f, b)
        # Update b (subtract quantities from b to keep A symmetric)
        x_BC = jnp.where(bc_mask, bc_values, 0.0)
        temp = b - A @ x_BC
        b = jnp.where(bc_mask, b, temp)
        # update A
        P_bc = jnp.where(algorithm["pore.bc.value_mask"], jnp.arange(Np), -1)
        mask = jnp.isin(A.indices[:, 0], P_bc) | jnp.isin(A.indices[:, 1], P_bc)
        A_data = jnp.where(mask, 0, A.data)
        # Add diagonal entries back into A
        isdiag = A.indices[:, 0] == A.indices[:, 1]
        mask = isdiag * jnp.isin(A.indices[:, 0], P_bc)
        A.data = jnp.where(mask, f, A_data)
        # Finally, update A in BCOO format
        # A = js.BCOO((A_data, A_indices), shape=(Np, Np))
    # apply outflow BC to A and b
    if "pore.bc.outflow" in algorithm.keys():
        # modify A: diag(A) += r
        r = algorithm["pore.bc.outflow"]
        r_mask = algorithm["pore.bc.outflow_mask"]
        mask = jnp.zeros(2*Nt + Np, dtype=bool)
        mask = mask.at[0:Np].set(r_mask)  # FIXME: assumes diags are first Np
        values = jnp.zeros(2*Nt + Np, dtype=float)
        values = values.at[0:Np].set(r)  # FIXME: assumes diags are first Np
        A.data += values * mask
        # Finally, update A in BCOO format
        # A = js.BCOO((A_data, A_indices), shape=(Np, Np))

    return A, b


def solve(proj, alg, x0=None, tol=1e-5, atol=0.0, maxiter=None):
    
    # get algorithm
    algorithm = proj[alg]
    # get network
    network = proj["network"]
    # get A and b
    A = algorithm["A"]
    b = algorithm["b"]
    # get initial guess
    Np = len(network["pore.coords"])
    if x0 is None:
        x0 = jnp.zeros(Np)
    # solve
    x, _ = jax.scipy.sparse.linalg.cg(A, b, x0=x0,
                                      tol=tol, atol=atol,
                                      maxiter=maxiter)

    return x


def rate(proj, alg, x, pores=[], throats=[], mode='group'):
    """
    Calculates the net rate of material moving into a given set of
    pores or throats

    Parameters
    ----------
    x : array_like
        The solved for quantity
    pores : array_like
        The pores for which the rate should be calculated
    throats : array_like
        The throats through which the rate should be calculated
    mode : str, optional
        Controls how to return the rate. The default value is 'group'.
        Options are:

        ===========  =====================================================
        mode         meaning
        ===========  =====================================================
        'group'      Returns the cumulative rate of material
        'single'     Calculates the rate for each pore individually
        ===========  =====================================================

    Returns
    -------
    If ``pores`` are specified, then the returned values indicate the
    net rate of material exiting the pore or pores.  Thus a positive
    rate indicates material is leaving the pores, and negative values
    mean material is entering.

    If ``throats`` are specified the rate is calculated in the
    direction of the gradient, thus is always positive.

    If ``mode`` is 'single' then the cumulative rate through the given
    pores (or throats) are returned as a vector, if ``mode`` is
    'group' then the individual rates are summed and returned as a
    scalar.

    """
    # get algorithm
    algorithm = proj[alg]
    # get network
    network = proj["network"]
    # get conductance
    throat_conductance = algorithm["conductance"]

    pores = jnp.array(pores)
    throats = jnp.array(throats)

    if throats.size > 0 and pores.size > 0:
        raise Exception('Must specify either pores or throats, not both')
    if (throats.size == 0) and (pores.size == 0):
        raise Exception('Must specify either pores or throats')

    # get Nt and Np
    Nt = len(network['throat.conns'])
    Np = len(network['pore.coords'])

    # get conductance
    g = network[throat_conductance]

    P12 = network['throat.conns']
    X12 = x[P12]
    if g.size == Nt:
        g = jnp.tile(g, (2, 1)).T    # Make conductance an Nt by 2 matrix
    # The next line is critical for rates to be correct
    # We could also do "g.T.flatten()" or "g.flatten('F')"
    g = jnp.flip(g, axis=1)
    Qt = jnp.diff(g*X12, axis=1).ravel()

    if throats.size:
        R = jnp.absolute(Qt[throats])
        if mode == 'group':
            R = jnp.sum(R)
    elif pores.size:
        Qp = jnp.zeros((Np, ))
        Qp = Qp.at[P12[:, 0]].add(-Qt)
        Qp = Qp.at[P12[:, 1]].add(Qt)
        R = jnp.where(pores, Qp, 0)  # FIXME: check that this works, R = Qp[pores]
        if mode == 'group':
            R = jnp.sum(R)

    return jnp.array(R, ndmin=1)


def set_outflow_BC(proj,
                   alg,
                   pores,
                   throats,
                   mode='add'):
        r"""
        Adds outflow boundary condition to the selected pores

        Parameters
        ----------
        pores : array_like
            The pore indices where the condition should be applied
        mode : str, optional
            Controls how the boundary conditions are applied. The default value
            is 'merge'. For definition of various modes, see the
            docstring for ``set_BC``.
        force : bool, optional
            If ``True`` then the ``'mode'`` is applied to all other bctypes as
            well. The default is ``False``.

        Notes
        -----
        Outflow condition means that the gradient of the solved quantity
        does not change, i.e. is 0.

        """
        # get algorithm
        algorithm = proj[alg]
        # get algorithm
        network = proj["network"]
        # Calculating A[i,i] values to ensure the outflow condition
        Np = len(network["pore.coords"])
        pores = jnp.arange(Np)[pores]
        # throats = misc.find_neighbor_throats(network, pores=pores)
        C12 = network["throat.conns"][throats]
        P12 = network[algorithm['pressure']][C12]
        gh = network[algorithm['hydraulic_conductance']][throats]
        Q12 = -gh * jnp.diff(P12, axis=1).squeeze()
        Qp = jnp.zeros(Np)
        Qp = jnp.add.at(Qp, C12[:, 0], -Q12, inplace=False)
        Qp = jnp.add.at(Qp, C12[:, 1], Q12, inplace=False)
        # set BC
        set_BC(proj, alg,
               pores=pores,
               bcvalues=Qp[pores],
               bctype='outflow',
               mode=mode)


def set_source(proj, alg, pores, propname):
        r"""
        Simple set source helper function
        """
        # get algorithm
        algorithm = proj[alg]
        # assign label where source terms are applied
        algorithm["pore.source." + propname] = pores


def apply_sources(proj, alg):
    """
    Updates ``A`` and ``b``, applying source terms to specified pores.

    """
    try:
        # get algorithm
        algorithm = proj[alg]
        # get network
        network = proj["network"]
        # get A and b
        A = algorithm["A"]
        b = algorithm["b"]
        # get Np
        Np = len(network["pore.coords"])
        Nt = len(network["throat.conns"])
        # apply sources
        for source in algorithm["sources"]:
            S1 = network["pore." + source]["S1"]
            S2 = network["pore." + source]["S2"]
            S_mask = algorithm["pore.source." + source]
            # modify A: diag(A) += -S1
            mask = jnp.zeros(2*Nt + Np, dtype=bool)
            mask = mask.at[0:Np].set(S_mask)  # FIXME: assumes diags are first Np
            values = jnp.zeros(2*Nt + Np, dtype=float)
            values = values.at[0:Np].set(S1)  # FIXME: assumes diags are first Np
            A.data -= values * mask
            # modify b: b += S2
            b += S2 * S_mask
    except KeyError:
        pass
    
    return A, b

    
def update_A_and_b(proj, alg):

    # get algorithm
    algorithm = proj[alg]
    # build A and b
    algorithm["A"] = build_A(proj, alg)
    algorithm["b"] = build_b(proj, alg)
    # apply BCs
    algorithm["A"], algorithm["b"] = apply_BC(proj, alg)
    # apply sources
    algorithm["A"], algorithm["b"] = apply_sources(proj, alg)
