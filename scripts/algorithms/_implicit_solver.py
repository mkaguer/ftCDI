import jax
import jax.numpy as jnp

__all__ = ['implicit_solve']

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
        # FIXME: jax.jvp may be overkill when F which can be solved by Ax = b
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
    
    # FIXME: don't use python loop, although if small OKAY
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

