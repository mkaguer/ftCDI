import jax
import jax.numpy as jnp
from jax import config
import matplotlib.pyplot as plt
import numpy as np

config.update("jax_enable_x64", True)


def implicit_solve(fun,
                   t_span,
                   y0,
                   dt,
                   args,
                   method="euler",
                   t_eval=None):
    
    
    def residual(u, u_prev, dt, args):
        
        if method == "euler":
            F = u - u_prev - dt * fun(u, *args)
        
        elif method == "crank-nicolson":
            F = u - u_prev - dt / 2 * (fun(u, *args) + fun(u_prev, *args))
            
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
    
    
    def implicit_step(u_prev, dt, args, iters=5, tol=1e-6):
        
        # guess u
        u = u_prev
        
        for i in range(iters):
            
            # FIXME: should we calculate residual before or after?
            # calculate residual
            F = residual(u, u_prev, dt, args)
            normF = jnp.linalg.norm(F)

            # do one newton step
            delta = newton_step(u, u_prev, dt, args)

            # update u for next step
            u = u + delta

            # break if normF < tolerance 
            if normF < tol:
                print('converged')
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
    us = u_prev.copy()
    ts = jnp.array([t0])
    for t in jnp.arange(t0+dt, tf+dt, dt):
        # do full implicit solve
        u = implicit_step(u_prev, dt, args)
        # get new u_prev
        u_prev = u
        # save u if in t_eval
        if jnp.any(jnp.isclose(t, t_eval, atol=1e-10)):
            us = jnp.concatenate((us, u_prev))
            ts = jnp.concatenate((ts, jnp.array([t])))


    return ts, us
        


# ode to solve
def f(u, a):
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
                        args=(5,),
                        t_eval=jnp.arange(t0, tf+dt, dt))


# plot results
ts = np.asarray(ts)
ys = np.asarray(ys)
plt.plot(ts, ys)