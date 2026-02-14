"""
v2 does not take jax.jvp. It uses cg to solve Ax=b directly!
"""
import jax
import jax.numpy as jnp
from jax import config
import matplotlib.pyplot as plt
import numpy as np
import openpnm as op
from jax.experimental import sparse
import time

config.update("jax_enable_x64", True)

op.visualization.set_mpl_style()

np.random.seed(1)


#%% Solve diffusion problem on a pore network

# intialize network
net = op.network.Cubic(shape=[100, 100, 50], spacing=1e-5)
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
del geo_mods['pore.diameter']
net['pore.diameter'] = 5e-6
net.add_model_collection(models=geo_mods)
net.regenerate_models()

# initialize phase
phase = op.phase.Water(network=net)
phase["throat.diffusivity"] = 1e-9  # 1.68e-9
phys_mods = op.models.collections.physics.basic.copy()
phase.add_model_collection(models=phys_mods)
phase.regenerate_models()

# set initial concentration
c0 = np.zeros(net.Np)
c0[net.pores("xmin")] = 1.0
net["pore.concentration"] = c0

# initialize mass transport
mt = op.algorithms.TransientFickianDiffusion(network=net,
                                             phase=phase)
mt.set_value_BC(pores=net.pores("xmin"), values=1)
mt.set_value_BC(pores=net.pores("xmax"), values=0)

# Finally, apply BCs and source terms to instantiate A and b
mt._apply_BCs()
mt._apply_sources()

# get A and b
A = mt.A
b = mt.b

# get volume
V = net["pore.volume"]

# convert A and b to jax arrays
A = sparse.BCOO.from_scipy_sparse(A)
b = jnp.array(b)
V = jnp.array(V)
c0 = jnp.array(c0)

# set device
device = 'cpu'
A = jax.device_put(A, jax.devices(device)[0])
b = jax.device_put(b, jax.devices(device)[0])
V = jax.device_put(V, jax.devices(device)[0])
c0 = jax.device_put(c0, jax.devices(device)[0])

# check that everything is on chosen device
print(A.data.device)
print(b.device)
print(V.device)
print(c0.device)


def implicit_solve(u0,
                   t_span,
                   dt,
                   args,
                   tol=1e-6,
                   maxiter=50):
    
    
    def linear_map(dt, A, b, V):
        def matvec(u):
            return u + dt / 2 / V * A @ u
        return matvec


    def rhs(u_prev, dt, A, b, V):
        return u_prev - dt / 2 / V * A @ u_prev + dt / V * b
    
    
    # get A, b, V from args
    # A, b, V = args
    # get t0 and tf
    t0, tf = t_span
    # get u_prev
    u_prev = u0
    # initialize solution
    us = jnp.array([u_prev])
    ts = jnp.array([t0])
    # time stepping
    for t in np.arange(t0+dt, tf+dt, dt):
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


static_argnames = ["t_span", "dt"]
implicit_solve = jax.jit(implicit_solve, static_argnames=static_argnames)


# first solve
dt = 0.25
t_span = (0, 10)
args = (A, b, V)
u0 = c0
start = time.time()
ts, us = implicit_solve(u0,
                        t_span,
                        dt,
                        args,
                        tol=1e-6,
                        maxiter=50)
us.block_until_ready()
stop = time.time()
print(f"First solve: {stop - start}s")


# first solve
dt = 0.25
t_span = (0, 10)
args = (A, b, V)
u0 = c0
start = time.time()
ts, us = implicit_solve(u0,
                        t_span,
                        dt,
                        args,
                        tol=1e-6,
                        maxiter=50)
us.block_until_ready()
stop = time.time()
print(f"Second solve: {stop - start}s")


# print average
print(jnp.average(us[-1, :]))  # 0.11931704574497126

