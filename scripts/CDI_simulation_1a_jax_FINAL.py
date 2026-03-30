# import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as _np
import openpnm as op
import pnmlib as pnm
import jax
import jax.numpy as jnp
import collection.jax as co
from jax import config
import models.numpy.effective_volume as effective_volume
import properties as prpts
import matplotlib.pyplot as plt
import pnmlib.models as mods
import models.jax as models
from IPython.display import clear_output
import time

op.visualization.set_mpl_style()

config.update("jax_disable_jit", False)
config.update("jax_enable_x64", True)

# create blank project dict
proj = {}

# import network (full cell with labels)
data = _np.load('../networks/create_full_cell_1a.npz')
data = {key: _np.array(data[key]) for key in data.files}

# convert to jax
net = pnm.io.numpy_to_jax(data)

# add network to project (for organization)
proj["network"] = net

# get Np and Nt
Np = len(net["pore.coords"])
Nt = len(net["throat.conns"])

# assign throat outlet label
pores = jnp.arange(Np)[net["pore.outlet"]]
net["throat.outlet"] = mods.misc.find_neighbor_throats(net, pores=pores)

# calculate throats coords, for intersecting_cylinders!
conns = net["throat.conns"]
coords = net["pore.coords"]
t_coords = jnp.diff(coords[conns], axis=1).squeeze(1)/2 + coords[conns[:, 0]]
net["throat.coords"] = t_coords

# infer spacing
conns = net["throat.conns"]
coords = net["pore.coords"]
spacing = jnp.max(jnp.abs(coords[conns[0, 0]] - coords[conns[0, 1]]))
spacing = jnp.round(spacing, 10)

# set pore thickness for perforated pores (i.e. cylinders)
net["pore.thickness"] = spacing

# add zeta parameter
zeta = 0.1
mask = net["throat.micropore"]
net["throat.zeta"] = jnp.where(mask, zeta, 1.0)

# add geometry models, this is important for zeta that may change!
pnm.models.apply_models(net,
                        models=co.geometry.macropore.spheres_and_cylinders,
                        domain="macropore")
pnm.models.apply_models(net,
                        models=co.geometry.micropore.spheres_and_cylinders,
                        domain="micropore")
pnm.models.apply_models(net,
                        models=co.geometry.separator.continuum,
                        domain="separator")
pnm.models.apply_models(net,
                        models=co.geometry.macropore.intersecting_cylinders,
                        domain="perforated")

# retrieve properties
rho_sep = prpts.properties["rho_sep"]
rho_mi = prpts.properties["rho_mi"]
V_cell = prpts.properties["V_cell"]
Ca = prpts.properties["Ca"]
mu_att = 2.0  # prpts.properties["mu_att"]
P_in = 3.1  # prpts.properties["P_in"]
P_out = prpts.properties["P_out"]
cf = prpts.properties["cf"]
T = prpts.properties["T"]

# Quick fix: offload to numpy!
temp = pnm.io.jax_to_numpy(net)
temp = op.io.network_from_porespy(temp)

# calculate effective volume
start = time.time()
eff_vol = effective_volume.bcc_fast(network=temp,
                                    rho_sep=rho_sep,
                                    rho_mi=rho_mi)
# eff_vol = jnp.array(_np.load("V.npy"))
stop = time.time()
print(f"time to calc volumes: {stop - start}s")
net["pore.effective_volume"] = jnp.asarray(eff_vol)
del temp

# assign micropore volume
theta = 1.0
net["pore.micro_volume"] = theta * net["pore.effective_volume"]

# set concentration and pressure
net["pore.concentration"] = jnp.ones(Np) * cf
net["pore.pressure"] = jnp.ones(Nt)
net["pore.temperature"] = jnp.ones(Np) * T

# add phase models, I added interpolation to models
pnm.models.apply_models(net,
                        models=co.phase.macropore,
                        domain="macropore")
pnm.models.apply_models(net,
                        models=co.phase.micropore,
                        domain="micropore")
pnm.models.apply_models(net,
                        models=co.phase.separator,
                        domain="separator")
pnm.models.apply_models(net,
                        models=co.phase.macropore,
                        domain="perforated")

# set "old" concentrations
# FIXME: remove these but MUST change default model behaviour
net["pore.concentration_old"] = net["pore.concentration"].copy()
net["throat.concentration_old"] = net["throat.concentration"].copy()

# add physics models to phase
pnm.models.apply_models(net,
                        models=co.physics.perforated,
                        domain="perforated")
pnm.models.apply_models(net,
                        models=co.physics.macropore,
                        domain="macropore")
pnm.models.apply_models(net,
                        models=co.physics.micropore,
                        domain="micropore")
pnm.models.apply_models(net,
                        models=co.physics.separator,
                        domain="separator")

# add properties to phase, these get from imported properties dict!
# set potential
V_cell = jnp.asarray(V_cell, dtype=jnp.float64)  # ensure weak_type is False, avoid recompilation
phi = jnp.where(net["pore.anode"], V_cell/2, 0.0)
phi = jnp.where(net["pore.cathode"], -V_cell/2, phi)
net["pore.potential"] = phi
net["pore.electrode_potential"] = phi
net["pore.attraction_term"] = jnp.where(net["pore.micropore"], mu_att, 0.0)
net["pore.capacitance"] = jnp.where(net["pore.micropore"], Ca, 1.0)
net["pore.surface_area"] = jnp.ones(Np)*1

# select time step
dt = 0.5
net["time_step"] = dt

# set iniital guess for donnan potential
net["pore.donnan_potential"] = jnp.zeros(Np)

# add phi_d and c_mi properties
pnm.models.apply_models(net,
                        models=co.physics.properties,
                        domain="micropore")

# set old c_mi and phi_d
net["pore.micro_concentration_old"] = net["pore.micro_concentration"].copy()
net["pore.donnan_potential_old"] = net["pore.donnan_potential"].copy()

# add source term models
pnm.models.apply_models(net,
                        models=co.physics.sources,
                        domain="micropore")

#%% Stokes Flow

sf = {}
proj["alg1"] = sf

# write keys to sf
sf["quantity"] = "pore.pressure"
sf["conductance"] = "throat.hydraulic_conductance"

# set BCs
inlet = net["pore.inlet"]
outlet = net["pore.outlet"]
pnm.algorithms.set_BC(proj, "alg1", inlet, bctype='value',
                      bcvalues=P_in, mode="overwrite")
pnm.algorithms.set_BC(proj, "alg1", outlet, bctype='value',
                      bcvalues=P_out, mode="add")

# build A and b
sf["A"] = pnm.algorithms.build_A(proj, "alg1")
sf["b"] = pnm.algorithms.build_b(proj, "alg1")

# apply BCs
sf["A"], sf["b"] = pnm.algorithms.apply_BC(proj, alg="alg1")

# solve
x = pnm.algorithms.solve(proj, "alg1")
# x = jnp.ones(Np) * P_in

# calculate the rate
Q = pnm.algorithms.rate(proj, "alg1", x, pores=inlet)
print(f"PNM Flow Rate: {Q*1e6*60} mL/min")

# scale to cell flow rate
coords = net["pore.coords"]
Ax = (jnp.max(coords[:, 1]) + spacing/2) * (jnp.max(coords[:, 2]) + spacing/2)
Acell = 1.55/100 * 1.55/100  # m2
f = Acell/Ax
print(f"Cell Flow Rate: {f*Q*1e6*60} mL/min")

# regenerate conductance models!
net["pore.pressure"] = x
co.regenerate_models(proj, net)

# calculate outflow
Qp = models.misc.outflow(net, pores=net["pore.outlet"], throats=net["throat.outlet"])
net["pore.outflow"] = Qp


#%% Implicit Solvers


def charge_solve(phi0, phi, dt, args, tol=1e-6, atol=0.0, maxiter=50):

    A, b, V = args

    def matvec(x):
        return A @ x + V / dt * x

    # compute diagonal ON THE FLY (safe for JIT)
    Np = len(b)
    A_diag = A.data[:Np]  # A.diagonal()
    M_inv = 1.0 / (A_diag + V / dt + 1e-12)

    def precond(x):
        return M_inv * x

    b_t = b + V * phi0 / dt

    phi_new, info = jax.scipy.sparse.linalg.bicgstab(
        matvec,
        b_t,
        x0=phi,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=precond
    )

    return phi_new, info


def mass_solve(c0, c, dt, args, tol=1e-6, atol=0.0, maxiter=50):

    A, b, V = args

    def matvec(x):
        return A @ x / V + x / dt

    Np = len(b)
    A_diag = A.data[:Np]  # FIXME: works only for COO
    M_inv = 1.0 / (A_diag / V + 1.0 / dt + 1e-12)

    def precond(x):
        return M_inv * x

    b_t = b / V + c0 / dt

    c_new, info = jax.scipy.sparse.linalg.bicgstab(
        matvec,
        b_t,
        x0=c,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=precond
    )

    return c_new, info


charge_solve = jax.jit(charge_solve)
mass_solve = jax.jit(mass_solve)


#%% Build A and b

from pnmlib.network import create_adjacency_matrix, graph_laplacian
import jax.experimental.sparse as js


def build_A(network, g):
    
    # create adjacency matrix
    am = create_adjacency_matrix(network, weights=g, fmt='coo')
    # make laplacian
    A = graph_laplacian(am, fmt='coo')

    return A


def build_b(Np):
    
    # create b as array of zeros Np long
    b = jnp.zeros(Np, dtype=float)

    return b


def apply_value_BC(A, b, bc_values, bc_mask):

    Np = len(b)
    # get row and col
    row = A.indices[:, 0]
    col = A.indices[:, 1]
    # safer diagonal extraction
    isdiag = row == col
    diag_vals = A.data[:Np]  # FIXME: :Np avoids boolean index
    f = diag_vals.mean()
    # update b
    b = jnp.where(bc_mask, bc_values * f, b)
    x_BC = jnp.where(bc_mask, bc_values, 0.0)
    temp = b - A @ x_BC
    b = jnp.where(bc_mask, b, temp)
    # fast masking (NO isin)
    mask = bc_mask[row] | bc_mask[col]
    A_data = jnp.where(mask, 0.0, A.data)
    # restore diagonal
    diag_mask = isdiag & bc_mask[row]
    A_data = jnp.where(diag_mask, f, A_data)
    # rebuild BCOO
    A = js.BCOO((A_data, A.indices), shape=A.shape)

    return A, b


def apply_outflow_BC(A, b, bc_values, bc_mask):

    # get row and col
    row = A.indices[:, 0]
    col = A.indices[:, 1]
    # identify diagonal entries robustly
    isdiag = row == col
    # map node-based quantities onto sparse entries
    diag_nodes = row  # since row == col for diag entries
    # build contribution only where needed (no big masks)
    diag_update = jnp.where(
        isdiag,
        jnp.where(bc_mask[diag_nodes], bc_values[diag_nodes], 0.0),
        0.0
    )
    # update A (NO in-place mutation)
    A_data = A.data + diag_update
    # rebuild sparse matrix (same structure → no recompilation)
    A = js.BCOO((A_data, A.indices), shape=A.shape)

    return A, b


def apply_sources(A, b, S1, S2, S_mask):

    # get row and col
    row = A.indices[:, 0]
    col = A.indices[:, 1]
    # identify diagonal entries robustly
    isdiag = row == col
    # map node-based quantities onto sparse entries
    diag_nodes = row  # since row == col for diag entries
    # build contribution only where needed (no big masks)
    diag_update = jnp.where(
        isdiag,
        jnp.where(S_mask[diag_nodes], S1[diag_nodes], 0.0),
        0.0
    )
    # update A (NO in-place mutation)
    A_data = A.data - diag_update
    # rebuild sparse matrix (same structure → no recompilation)
    A = js.BCOO((A_data, A.indices), shape=A.shape)
    # update RHS (cheap, no issues)
    b = b + S2 * S_mask

    return A, b


def rate(network, g, x, pores=[], throats=[], mode='group'):
    """
    Calculates the net rate of material moving into a given set of
    pores or throats

    Parameters
    ----------
    g : array_like
        The conductance
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
    # get pores/throat as jax arrays
    pores = jnp.array(pores)
    throats = jnp.array(throats)

    if throats.size > 0 and pores.size > 0:
        raise Exception('Must specify either pores or throats, not both')
    if (throats.size == 0) and (pores.size == 0):
        raise Exception('Must specify either pores or throats')

    # get Nt and Np
    Nt = len(network['throat.conns'])
    Np = len(network['pore.coords'])

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


def _get_charge_A_and_b(network):

    # get Np
    Np = len(network["pore.coords"])
    # get conductance
    g = network["throat.ionic_conductance"]
    # build A
    A = build_A(network, g)
    # build b
    b = build_b(Np)
    # apply value BCs
    A, b = apply_value_BC(A, b,
                          bc_values=0.0,
                          bc_mask=network["pore.separator"])
    # apply sources
    A, b = apply_sources(A, b,
                         S1=network["pore.charge_source"]["S1"],
                         S2=network["pore.charge_source"]["S2"],
                         S_mask=network["pore.micropore"])

    return A, b


def _get_mass_A_and_b(network, cf):

    # get Np
    Np = len(network["pore.coords"])
    # get conductance
    g = network["throat.mass_conductance"]
    # build A
    A = build_A(network, g)
    # build b
    b = build_b(Np)
    # apply value BCs
    A, b = apply_value_BC(A, b,
                          bc_values=cf,
                          bc_mask=network["pore.inlet"])
    # apply outflow BC
    A, b = apply_outflow_BC(A, b,
                            bc_values=network["pore.outflow"],
                            bc_mask=network["pore.outlet"])
    # apply sources
    A, b = apply_sources(A, b,
                         S1=network["pore.mass_source"]["S1"],
                         S2=network["pore.mass_source"]["S2"],
                         S_mask=network["pore.micropore"])

    return A, b


_get_charge_A_and_b = jax.jit(_get_charge_A_and_b)
_get_mass_A_and_b = jax.jit(_get_mass_A_and_b)


#%%  Update Source Terms

def _update_mass_source(network, c, phi):

    # update donnan potential
    donnan_potential = models.electrical_double_layer.donnan_potential
    phi_d0 = network["pore.donnan_potential"]
    phi_d = donnan_potential(phi_d0, phi, c,
                             phi_e=network["pore.electrode_potential"],
                             mu_att=network["pore.attraction_term"],
                             T=network["pore.temperature"],
                             C=network["pore.capacitance"],
                             a=network["pore.surface_area"])

    # calculate micropore concentration
    model = models.electrical_double_layer.micropore_concentration
    c_mi = model(c, phi_d,
                 T=network["pore.temperature"],
                 mu_att=network["pore.attraction_term"])

    # update mass source
    mass_source = models.electrical_double_layer.mass_source
    Rm = mass_source(c_mi,
                     c_mi_i=network["pore.micro_concentration_old"],
                     V_mi=network["pore.effective_volume"],
                     dt=network["time_step"])
    
    return phi_d, c_mi, Rm


def _update_charge_source(network, c, phi):

    # update donnan potential
    donnan_potential = models.electrical_double_layer.donnan_potential
    phi_d0 = network["pore.donnan_potential"]
    phi_d = donnan_potential(phi_d0, phi, c,
                             phi_e=network["pore.electrode_potential"],
                             mu_att=network["pore.attraction_term"],
                             T=network["pore.temperature"],
                             C=network["pore.capacitance"],
                             a=network["pore.surface_area"])

    # update charge source
    charge_source = models.electrical_double_layer.charge_source
    Rc = charge_source(phi_d,
                       phi_d_i=network["pore.donnan_potential_old"],
                       C=network["pore.capacitance"],
                       a=network["pore.surface_area"],
                       V=network["pore.effective_volume"],
                       dt=network["time_step"])
    
    return phi_d, Rc


_update_mass_source = jax.jit(_update_mass_source)
_update_charge_source = jax.jit(_update_charge_source)


#%%  Update Conductance

def _update_charge_conductance(network, c):

    # get conns
    conns = network["throat.conns"]

    # update ionic conductivity (pore and throat)
    conductivity = models.electrical_double_layer.conductivity
    Kp = conductivity(c=c,
                      D=network["pore.diffusivity"],
                      T=network["pore.temperature"])
    Kt = conductivity(c=jnp.average(c[conns], axis=1),
                      D=network["throat.diffusivity"],
                      T=network["throat.temperature"])

    # calculate ionic conductance
    poisson_conductance = models.electrical_double_layer.poisson_conductance
    K = poisson_conductance(network, Kp, Kt,
                            sf=network["throat.diffusive_size_factors"])
    
    return K


_update_charge_conductance = jax.jit(_update_charge_conductance)


#%% Solve


def calc_current(network, throats):

    # get conductance
    g = network["throat.ionic_conductance"]
    phi = network["pore.potential"]
    # calculate current
    current = rate(network, g, phi, throats=throats, mode="group")
    
    return current


calc_current = jax.jit(calc_current)


c_mi = net["pore.micro_concentration"][net["pore.micropore"]]
V_mi = net["pore.micro_volume"][net["pore.micropore"]]
c = net["pore.concentration"]
V = net["pore.effective_volume"]
m0 = jnp.dot(c, V) + jnp.dot(c_mi/2, V_mi)
print(f"Initial Mass: {m0}")

# get c, phi, c_mi, phi_d
c = net["pore.concentration"]
phi = net["pore.potential"]
c_mi = net["pore.micro_concentration"]
phi_d = net["pore.donnan_potential"]

# phi = jnp.where(net["pore.perforated"], 0.0, phi)

# get capacitance
C = net["pore.capacitance"]
a = net["pore.surface_area"]
V = net["pore.effective_volume"]
CaV = jnp.where(net["pore.micropore"], C*a*V, 0.0)


# time stepping
t0 = 0
dt = dt
tf = 500
t_save = jnp.arange(t0, tf + dt*5, dt*5)
# store solution for first time!
y = _np.concatenate((_np.array(c), _np.array(phi), _np.array(c_mi), _np.array(phi_d)))
x = _np.array([t0])
# initialize current
I = []
k = 0
for t in jnp.arange(t0+dt, tf+0.9*dt, dt):
    clear_output(wait=True)
    print(f"Time: {t}s")
    # get initial conditions
    c0 = c.copy()
    phi0 = phi.copy()
    # initialize c_old and phi_old
    c_old = c0.copy()
    phi_old = phi0.copy()
    # define gummel convergence
    g_res = jnp.array([100.0, 100.0])
    g_tol = jnp.array([1e-4, 1e-6])  # jnp.array([1e-6, 1e-6])
    g_max_iter = 40
    w = 0.15
    a = 1.0
    for g_iter in range(g_max_iter):
        print(f"Gummel Iteration No. {g_iter+1}")
        # update charge source, most recent c and phi
        start = time.time()
        phi_d, Rc = _update_charge_source(net, c, phi)
        phi_d.block_until_ready()
        stop = time.time()
        # print(jnp.sum(jnp.abs(Rc["rate"][net["pore.micropore"]])))
        # xx
        print(f'Charge Source Time: {stop - start}s')
        net["pore.donnan_potential"] = phi_d
        net["pore.charge_source"] = Rc
        # build charge A and b
        start = time.time()
        Ac, bc = _get_charge_A_and_b(net)
        bc.block_until_ready()
        stop = time.time()
        print(f'Charge A and b Time: {stop - start}s')
        # solve phi
        start = time.time()
        phi_new, info1 = charge_solve(phi0,
                               phi,
                               dt,
                               args=(Ac, bc, CaV),
                               tol=1e-8,
                               atol=0.0,
                               maxiter=100)
        phi_new.block_until_ready()
        stop = time.time()
        print(f'Charge Solve Time: {stop - start}s')
        # update potential
        # phi_new = jnp.clip(phi_new, -V_cell/2, V_cell/2)
        phi = a * phi_new + (1 - a) * phi_old
        net["pore.potential"] = phi
        # update mass source, most recent c and phi
        start = time.time()
        phi_d, c_mi, Rm = _update_mass_source(net, c, phi)
        phi_d.block_until_ready()
        stop = time.time()
        print(f'Mass Source Time: {stop - start}s')
        net["pore.donnan_potential"] = phi_d
        net["pore.micro_concentration"] = c_mi
        net["pore.mass_source"] = Rm
        # build mass A and b
        start = time.time()
        Am, bm = _get_mass_A_and_b(net, cf)
        Am.block_until_ready()
        stop = time.time()
        print(f'Mass A and b Time: {stop - start}s')
        # solve c
        start = time.time()
        c_new, info2 = mass_solve(c0,
                           c,
                           dt,
                           args=(Am, bm, V),
                           tol=1e-8,
                           atol=0.0,
                           maxiter=100)
        c_new.block_until_ready()
        stop = time.time()
        print(f'Mass Solve Time: {stop - start}s')
        c_new = jnp.clip(c_new, 1e-6, cf)
        c = w * c_new + (1 - w) * c_old
        net["pore.concentration"] = c
        # update charge conductance
        start = time.time()
        K = _update_charge_conductance(net, c)
        K.block_until_ready()
        stop = time.time()
        print(f'Update K Time: {stop - start}s')
        net["throat.ionic_conductance"] = K
        # calculate new residual
        if g_iter > 0:
            g_res = jnp.array([jnp.sum((c - c_old)**2),
                               jnp.sum((phi - phi_old)**2)])
        print(f"  Residual: {g_res}")
        if jnp.all(g_res < g_tol):
            print(f"  Convergence criteria met: {g_res}")
            break
        # break if max no. of iterations reached
        if g_iter == g_max_iter - 1:
            k += 1
            break
        # update old c and phi
        c_old = c.copy()
        phi_old = phi.copy()
    # update properties, most recent c and phi
    phi_d, c_mi, _ = _update_mass_source(net, c, phi)
    net["pore.donnan_potential"] = phi_d
    net["pore.micro_concentration"] = c_mi
    # update old properties
    net["pore.donnan_potential_old"] = net["pore.donnan_potential"].copy()
    net["pore.micro_concentration_old"] = net["pore.micro_concentration"].copy()
    # save data at t_save
    if jnp.any(jnp.isclose(t, t_save, atol=1e-10)):
        # get c, phi, c_mi, phi_d as numpy arrays
        c = _np.array(net["pore.concentration"])
        phi = _np.array(net["pore.potential"])
        c_mi = _np.array(net["pore.micro_concentration"])
        phi_d = _np.array(net["pore.donnan_potential"])
        # save results as y and x, assume that everything gets saved!
        y = _np.vstack((y, _np.concatenate((c, phi, c_mi, phi_d))))
        x = _np.concatenate((x, _np.array([t])))
        # calculate current
        mask = net["throat.separator"]
        mask = mask.at[-1].set(False)  # FIXME: assume last
        # get throat indices
        throats = jnp.where(mask)
        current = calc_current(net, throats)
        I.append(current[0])
    # get c, phi, c_mi, phi_d as jax arrays
    c = net["pore.concentration"]
    phi = net["pore.potential"]
    c_mi = net["pore.micro_concentration"]
    phi_d = net["pore.donnan_potential"]


# calculate final mass
c_mi = c_mi[net["pore.micropore"]]
mf = jnp.dot(c, V) + jnp.dot(c_mi/2, V_mi)

# calculate mass balance 
print(f"Mass Accumulated: {abs(mf-m0)/m0*100}%")  # 41.41290365980112% @ 5s

# calculate "theoretical" charge capacity
V = jnp.sum(net["pore.micro_volume"][net["pore.micropore"]]) / 2
capacity_theory = Ca * V * V_cell / 2 / 96485
print(f'Theoretical Capacity of charge: {capacity_theory} moles of charge')

# calculate "actual" salt capacity
V_mi = net["pore.micro_volume"][net["pore.micropore"]]
c_mi = net["pore.micro_concentration"][net["pore.micropore"]]/2 - cf*jnp.exp(mu_att)
print(f'Total Salt Captured 2: {jnp.sum(c_mi*V_mi)} moles of salt')

# plot discharge
plt.figure(1)
c_out = _np.average(y[:, 0:Np][:, net["pore.outlet"]], axis=1)
plt.plot(x, c_out, label="outlet")
plt.legend(frameon=True)
plt.title("Discharge Curve", fontweight="bold")
plt.xlabel("Time (s)")
plt.ylabel("Concentration (mM)")

# plot current
plt.figure(2)
plt.plot(x[1:], _np.array(I)*f*1e3, label="Discharge")
# plt.ylim([0, 4])
plt.legend(frameon=True)
plt.title("Discharge Curve", fontweight="bold")
plt.xlabel("Time (s)")
plt.ylabel("Current (mA)")

# save
_np.save("data/y" + "_" + str(mu_att) + "_" + str(zeta) + "_" + str(tf) + "s.npy", y)
_np.save("data/x" + "_" + str(mu_att) + "_" + str(zeta) + "_" + str(tf) + "s.npy", x)
_np.save("data/I" + "_" + str(mu_att) + "_" + str(zeta) + "_" + str(tf) + "s.npy", _np.array(I)*f*1e3)