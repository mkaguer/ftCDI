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


def implicit_solve(fun,
                   t_span,
                   y0,
                   dt,
                   args,
                   method="euler",
                   iters=5,
                   tol=1e-6,
                   newton_maxiter=50,
                   newton_tol=1e-6,
                   check_convergence=False):
    
    
    def residual(u, u_prev, dt, args):
        
        if method == "euler":
            F = u - u_prev - dt * fun(u, args)
        
        elif method == "crank-nicolson":
            F = u - u_prev - dt / 2 * (fun(u, args) + fun(u_prev, args))
        else:
            raise ValueError("method must be either euler or crank-nicolson")
        
            
        return F
    
    
    def Jv(delta, u, u_prev, dt, args):
        F = lambda u: residual(u, u_prev, dt, args)
        return jax.jvp(F, (u,), (delta,))[1]
    
    
    def linear_map(u, u_prev, dt, args):
        def matvec(v):
            return Jv(v, u, u_prev, dt, args)
        return matvec
    
    
    def newton_step(u, u_prev, dt, args, tol=1e-6, maxiter=50):
        
        # calculate residual
        F = residual(u, u_prev, dt, args)
        # get matvec product
        matvec = linear_map(u, u_prev, dt, args)
        # Solve J δ = -F using cg
        delta, info = jax.scipy.sparse.linalg.cg(
            matvec,
            -F,
            tol=tol,
            maxiter=maxiter
        )
        
        return delta
    
    
    def implicit_step(u_prev,
                      dt,
                      args,
                      iters=5,
                      tol=1e-6,
                      newton_maxiter=50,
                      newton_tol=1e-6):
        
        # guess u
        u = u_prev
        
        for i in range(iters):

            # do one newton step
            delta = newton_step(u, u_prev, dt, args,
                                newton_tol, newton_maxiter)

            # update u for next step
            u = u + delta
            
            # set to True if not jitted
            if check_convergence:
                # FIXME: should we calculate residual before or after?
                # calculate residual
                F = residual(u, u_prev, dt, args)
                normF = jnp.linalg.norm(F)
                # break if normF < tolerance 
                if normF < tol:
                    print(f"converged at residual: {F}")
                    break
                
                # raise error if not converged
                if i == iters - 1:
                    raise ValueError("Newton did not converge. Max steps reached!")
            
        return u
    
    # perform time stepping
    t0 = t_span[0]
    tf = t_span[1]
    # get u_prev from initial condition
    u_prev = y0
    # initialize solution
    us = jnp.array([u_prev])
    ts = jnp.array([t0])
    for t in jnp.arange(t0+dt, tf+dt, dt):
        # do full implicit solve
        u = implicit_step(u_prev, dt, args,
                          iters, tol,
                          newton_maxiter, newton_tol)
        # get new u_prev
        u_prev = u
        # FIXME: make t_eval argument!
        # save u at every time step
        us = jnp.concatenate((us, jnp.array([u_prev])))
        ts = jnp.concatenate((ts, jnp.array([t])))

    return ts, us
        


#%% Try solving simple problem

# ode to solve
def f(u, args):
    a, = args
    dudt = u + a
    return dudt


# call implict solve
t0 = 0.0
tf = 1.0
dt = 0.1
y0 = jnp.array([1.0])
ts, ys = implicit_solve(f,
                        t_span=(t0, tf),
                        y0=y0,
                        dt=dt,
                        method="crank-nicolson",
                        args=(5,))


# plot results
ts = np.asarray(ts)
ys = np.asarray(ys)
plt.plot(ts, ys)
plt.xlabel("t")
plt.ylabel("y")
plt.show()


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


def rhs(y, args):
    A, b, V = args
    dcdt = (-A @ y + b) / V
    return dcdt


# jit implicit solver
static_argnames = ["fun",
                   "t_span",
                   "dt",
                   "method",
                   "iters",
                   "check_convergence"]
implicit_solve = jax.jit(implicit_solve, static_argnames=static_argnames)

# call implict solve
t0 = 0.0
tf = 10
dt = 0.25
y0 = c0
start = time.time()
ts, ys = implicit_solve(rhs,
                        t_span=(t0, tf),
                        y0=y0,
                        dt=dt,
                        method="crank-nicolson",
                        args=(A, b, V),
                        iters=1,
                        tol=1e-4)
ys.block_until_ready()
stop = time.time()
print(f"First Solve: {stop - start}s")  # 5.766441345214844s

start = time.time()
ts2, ys2 = implicit_solve(rhs,
                        t_span=(t0, tf),
                        y0=y0,
                        dt=dt,
                        method="crank-nicolson",
                        args=(A, b, V),
                        iters=1,
                        tol=1e-4)
ys2.block_until_ready()
stop = time.time()
print(f"Second Solve: {stop - start}s")  # 3.3895936012268066s

# plot
ts = np.asarray(ts)
ys = np.asarray(ys)
ys_avg = np.average(ys, axis=1)
plt.plot(ts, ys_avg)
plt.xlabel("t")
plt.ylabel("y")
plt.show()

print(ys_avg[-1])  # 0.11931940272169364
