import numpy as np
import openpnm as op
import collection
import algorithms
import models.misc as mods
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import time


# create network
spacing = 1e-5
start = time.time()
net = op.network.BodyCenteredCubic(shape=[10, 10, 10], spacing=spacing)
stop = time.time()
print(f"Time to build network: {stop - start}s")

# this code below is good for 2d!
# shift z-axis of body pores
# mask = net["pore.body"]
# net["pore.coords"][:, 2] -= spacing/2 * mask

# delete corner to corner throats
op.topotools.trim(net, throats=net.throats("body_to_body"))

# label micropores
net.set_label(label="micropore",
              pores=net.pores("body"),
              throats=net.throats("corner_to_body"))

# label macropores
net.set_label(label="macropore",
              pores=net.pores("corner"),
              throats=net.throats("corner_to_corner"))

# label anode, cathode, seperator
x = net["pore.coords"][:, 0]
x_sep = np.round((np.max(x) - np.min(x))/2 + np.min(x), 10)
net["pore.cathode"] = x < x_sep
net["pore.anode"] = x > x_sep
net["pore.separator"] = np.isclose(x, x_sep)

# trim old labels
del net["pore.corner"], net["pore.body"]
del net["throat.corner_to_corner"]
del net["throat.corner_to_body"]
del net["throat.body_to_body"]


# add geometry to macropore
net["pore.diameter@macropore"] = 0.5*spacing
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
del geo_mods["pore.diameter"], geo_mods["pore.seed"], geo_mods["pore.max_size"]
net.add_model_collection(models=geo_mods, domain="macropore")

# add geometry to micropores
net["pore.diameter@micropore"] = 0.25*spacing
del geo_mods["throat.volume"]  # delete so doesn't get added to eff macro vol
net.add_model_collection(models=geo_mods, domain="micropore")

# regenerate models
net.regenerate_models()

# FIXME: will need new model here when we scale!
# calculate effective volume of micropores, this is hard coded!
Vp = net["pore.volume@macropore"][0]
Vt = net["throat.volume@macropore"][0]
net["pore.effective_volume@micropore"] = 0.2*(spacing**3 - Vp - Vt*2)  # FIXME: multiply by porosity
net["throat.volume@micropore"] = 0
eff_vol_mod = op.models.geometry.pore_volume.effective
net.add_model(propname="pore.effective_volume",
              model=eff_vol_mod,
              domain="macropore")

# create phase object
phase = op.phase.Phase(net)

# add properties to phase object
phase["pore.concentration"] = 100
phase["pore.potential@cathode"] = -0.5
phase["pore.potential@anode"] = 0.5
phase["pore.potential@separator"] = 0.0
phase["pore.electrode_potential@cathode"] = -0.5
phase["pore.electrode_potential@anode"] = 0.5
phase["pore.electrode_potential@separator"] = 0.0
phase["pore.attraction_term"] = 0
phase["pore.capacitance"] = 145 * 1e6
phase["pore.surface_area"] = 1
phase["pore.diffusivity@macropore"] = 1.68e-9
phase["pore.diffusivity@micropore"] = 6e-11
phase["pore.temperature"] = 298

# add physics models for macropore
phys_mods_ma = collection.physics.macropore
del phys_mods_ma["throat.diffusive_conductance"]
del phys_mods_ma["throat.hydraulic_conductance"]
del phys_mods_ma["throat.mass_conductance"]
phys_mods_ma["throat.conductivity"]["concentration"] = "throat.concentration"
phys_mods_ma["pore.conductivity"]["concentration"] = "pore.concentration"
phase.add_model_collection(models=phys_mods_ma, domain="macropore")

# add physics models for micropore
phys_mods_mi = collection.physics.micropore
del phys_mods_mi["throat.mass_conductance"]
del phys_mods_mi["throat.hydraulic_conductance"]
phys_mods_mi["throat.conductivity"]["concentration"] = "throat.concentration"
phys_mods_mi["pore.conductivity"]["concentration"] = "pore.concentration"
phase.add_model_collection(models=phys_mods_mi, domain="micropore")

# regenerate models
phase.regenerate_models()

# add donnan potential model
phase["pore.donnan_potential"] = np.zeros(net.Np)  # guess
phase.add_model(propname="pore.donnan_potential",
                model=mods.donnan_potential)

# select time step
dt = 0.01

# add source term model
phase["pore.donnan_potential_old"] = phase["pore.donnan_potential"].copy() 
phase.add_model(propname="pore.charge_source",
                model=mods.charge_source,
                domain="micropore",
                pore_donnan_potential="pore.donnan_potential",
                pore_donnan_potential_old="pore.donnan_potential_old",
                pore_volume="pore.effective_volume",
                pore_capacitance="pore.capacitance",
                pore_surface_area="pore.surface_area",
                time_step=dt)

# create charge transport algorithm
# ct = op.algorithms.TransientReactiveTransport(network=net, phase=phase)
ct = algorithms.CustomTransientReactiveTransport(network=net, phase=phase)
# set settings
ct.settings["conductance"] = "throat.ionic_conductance"
ct.settings["quantity"] = "pore.potential"
ct.settings["pore_volume"] = "pore.effective_volume"
# set source terms
ct.set_source(pores=net.pores("micropore"),
              propname="pore.charge_source")
# Finally, apply BCs and source terms to instantiate A and b
ct._apply_BCs()
ct._apply_sources()

# get A and b matrix
A = ct.A
b = ct.b

# get volume
C = phase["pore.capacitance"]
a = phase["pore.surface_area"]
V = phase["pore.effective_volume"]
V = C*a*V

# get initial condition
phi0 = phase["pore.potential"]

# convert A and b to jax arrays
A = sparse.BCOO.from_scipy_sparse(A)
b = jnp.array(b)
V = jnp.array(V)
phi0 = jnp.array(phi0)

# set device
device = 'cpu'
A = jax.device_put(A, jax.devices(device)[0])
b = jax.device_put(b, jax.devices(device)[0])
V = jax.device_put(V, jax.devices(device)[0])
phi0 = jax.device_put(phi0, jax.devices(device)[0])

# check that everything is on chosen device
print(A.data.device)
print(b.device)
print(V.device)
print(phi0.device)


def dae_solve(u0,
              t_span,
              dt,
              args,
              is_transient,
              tol=1e-6,
              maxiter=50):
        
    
    def linear_map(dt, A, b, V):
        def matvec(u):
            return V/dt*u + A @ u
        return matvec


    def rhs(u_prev, dt, A, b, V):
        return b + V * u_prev / dt 
    
    
    # get A, b, V from args
    A, b, V = args
    # mask out is_transient
    V = jnp.where(is_transient, V, 0)
    args = (A, b, V)
    # get t0 and tf
    t0, tf = t_span
    # get u_prev
    u_prev = u0
    # initialize solution
    us = jnp.array([u_prev])
    ts = jnp.array([t0])
    # time stepping
    for t in jnp.arange(t0+dt, tf+dt, dt):
        # get matvec, can we put this outsie?
        matvec = linear_map(dt, *args)
        # get rhs for new u_prev
        rhs_vec = rhs(u_prev, dt, *args)
        # solve using cg
        u, info = jax.scipy.sparse.linalg.cg(
            matvec,
            rhs_vec,
            tol=tol,
            maxiter=maxiter,
        )
        # update u_prev
        u_prev = u
        # FIXME: make t_eval argument!
        # save u at every time step
        us = jnp.concatenate((us, jnp.array([u_prev])))
        ts = jnp.concatenate((ts, jnp.array([t])))
    
    
    return ts, us


'''
# build I matrix
n = net.Np
row = jnp.arange(n, dtype=jnp.int32)
col = jnp.arange(n, dtype=jnp.int32)
data = jnp.ones(n) * V / dt
indices = jnp.stack([row, col], axis=1)
I = sparse.BCOO((data, indices), shape=(n, n))

# check A_t
dt = 0.01
A_t = I + A
eigvals, _ = jax.numpy.linalg.eigh(A_t.todense())
print(np.sum(eigvals < 0))
print(np.min(eigvals))
'''

static_argnames = ["t_span", "dt"]
dae_solve = jax.jit(dae_solve, static_argnames=static_argnames)

# first solve
dt = 0.01
t_span = (0, 0.01)
args = (A, b, V)
u0 = phi0
start = time.time()
ts, us = dae_solve(u0,
                   t_span,
                   dt,
                   args,
                   is_transient=net["pore.micropore"],
                   tol=1e-6,
                   maxiter=50)
us.block_until_ready()
stop = time.time()
print(f"time: {stop - start}s")  # 0.10451889038085938s

print(us[-1, :])
print(np.average(us[-1, :][net["pore.anode"]]))

# second solve
dt = 0.01
t_span = (0, 0.01)
args = (A, b, V)
u0 = phi0
start = time.time()
ts, us = dae_solve(u0,
                   t_span,
                   dt,
                   args,
                   is_transient=net["pore.micropore"],
                   tol=1e-6,
                   maxiter=50)
us.block_until_ready()
stop = time.time()
print(f"time: {stop - start}s")

print(us[-1, :])
print(np.average(us[-1, :][net["pore.anode"]]))  # 0.38798598548483143
