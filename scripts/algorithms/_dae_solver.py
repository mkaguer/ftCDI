import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse


__all__ = ['dae_solve',
           'dae_solve_v2']


def dae_solve(u0,
              t_span,
              dt,
              args,
              idx_s,
              idx_t,
              tol=1e-6,
              maxiter=50):

    A, b, V = args
    t0, tf = t_span

    # Break A into blocks
    def break_up_A(A, idx_s, idx_t):

        row = A.indices[:, 0]
        col = A.indices[:, 1]

        ss_mask = jnp.isin(row, idx_s) & jnp.isin(col, idx_s)
        st_mask = jnp.isin(row, idx_s) & jnp.isin(col, idx_t)
        ts_mask = jnp.isin(row, idx_t) & jnp.isin(col, idx_s)
        tt_mask = jnp.isin(row, idx_t) & jnp.isin(col, idx_t)

        def build_block(mask, rows_keep, cols_keep):
            new_data = A.data[mask]
            new_indices = A.indices[mask]

            row_map = jnp.searchsorted(rows_keep, new_indices[:, 0])
            col_map = jnp.searchsorted(cols_keep, new_indices[:, 1])
            new_indices = jnp.stack([row_map, col_map], axis=1)

            return sparse.BCOO((new_data, new_indices),
                               shape=(len(rows_keep), len(cols_keep)))

        return (
            build_block(ss_mask, idx_s, idx_s),
            build_block(st_mask, idx_s, idx_t),
            build_block(ts_mask, idx_t, idx_s),
            build_block(tt_mask, idx_t, idx_t),
        )

    A_ss, A_st, A_ts, A_tt = break_up_A(A, idx_s, idx_t)

    # Restrict vectors
    b_t = b[idx_t]
    V_t = V[idx_t]
    u_prev = u0[idx_t]

    # Steady solve
    def solve_ss(rhs):
        z, _ = jax.scipy.sparse.linalg.cg(
            A_ss, 
            rhs, 
            tol=tol, 
            maxiter=maxiter)
        return z

    # Linear operator (constant)
    def matvec(x):
        return x + (dt / 2.0) * (A_tt @ x) / V_t


    # RHS builder
    def build_rhs(u_prev):
        u_s = solve_ss(-A_st @ u_prev)
        return (
            u_prev
            + (dt / 2.0)
            * (2.0 * b_t - 2.0 * A_ts @ u_s - A_tt @ u_prev)
            / V_t
        )

    # One timestep
    def step(u_prev, t):

        rhs = build_rhs(u_prev)

        u_t, _ = jax.scipy.sparse.linalg.cg(
            matvec, 
            rhs, 
            tol=tol,
            maxiter=maxiter)

        u_s = solve_ss(-A_st @ u_t)

        u_full = jnp.zeros_like(b)
        u_full = u_full.at[idx_s].set(u_s)
        u_full = u_full.at[idx_t].set(u_t)

        return u_t, u_full

    # Time grid
    ts = jnp.arange(t0, tf + dt, dt)
    
    # Scan
    def scan_step(u_prev, t):
        u_next, u_full = step(u_prev, t)
        return u_next, u_full

    _, us = lax.scan(scan_step, u_prev, ts[1:])

    # prepend initial condition
    us = jnp.vstack([u0[None, :], us])

    return ts, us


def dae_solve_v2(u0,
                 t_span,
                 dt,
                 args,
                 is_transient,
                 M=None,
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
            M=M,
        )
        # update u_prev
        u_prev = u
        # FIXME: make t_eval argument!
        # save u at every time step
        us = jnp.concatenate((us, jnp.array([u_prev])))
        ts = jnp.concatenate((ts, jnp.array([t])))
    
    
    return ts, us