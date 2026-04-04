import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import cg, gmres, bicgstab

__all__ = ["newton_krylov",
           "newton_krylov_scan",
           "newton_krylov_scan_v2"]


def newton_krylov(fun,
                  x0,
                  max_iters=20,
                  tol=1e-6,
                  newton_maxiter=50,
                  newton_tol=1e-6,
                  alpha=1.0):
    """
    Fully jittable Newton–Krylov solver.

    Solves: fun(x) = 0
    """

    def Jv(x, v):
        return jax.jvp(fun, (x,), (v,))[1]

    def body(state):
        x, k, _ = state

        F = fun(x)

        def matvec(v):
            return Jv(x, v)

        delta, _ = gmres(
            matvec,
            -F,
            tol=newton_tol,
            maxiter=newton_maxiter
        )

        x_new = x + alpha * delta
        F_new = fun(x_new)
        normF = jnp.linalg.norm(F_new)

        return (x_new, k + 1, normF)

    def cond(state):
        _, k, normF = state
        return jnp.logical_and(normF > tol,
                               k < max_iters)

    # initial state
    F0 = fun(x0)
    normF0 = jnp.linalg.norm(F0)

    init_state = (x0, 0, normF0)

    x_final, _, _ = lax.while_loop(cond, body, init_state)

    return x_final


def newton_krylov_scan(fun, x0, max_iters=20, tol=1e-6, newton_maxiter=50, newton_tol=1e-6, alpha=1.0):
    """
    Fully jittable Newton–Krylov solver using lax.scan.
    """

    def Jv(x, v):
        return jax.jvp(fun, (x,), (v,))[1]

    def body_fun(state, _):
        x, converged = state

        # skip computation if already converged
        def compute(x):
            F = fun(x)

            def matvec(v):
                return Jv(x, v)

            delta, _ = bicgstab(
                matvec,
                -F,
                tol=newton_tol,
                maxiter=newton_maxiter
            )

            x_new = x + alpha * delta
            normF = jnp.linalg.norm(fun(x_new))
            converged_new = normF < tol

            return x_new, converged_new

        x_new, converged_new = jax.lax.cond(converged, lambda x: (x, True), compute, x)
        return (x_new, converged_new), None

    # initial state: x0 and not converged
    init_state = (x0, False)

    # scan over max_iters steps
    final_state, _ = lax.scan(body_fun, init_state, xs=None, length=max_iters)

    x_final, _ = final_state
    return x_final


def newton_krylov_scan_v2(fun,
                          x0,
                          max_iters=100,
                          tol=1e-6,
                          newton_maxiter=400,
                          newton_tol=1e-8,
                          alpha=1.0,
                          precond=None,
                          precond_eps=1e-12):
    """
    Fully jittable Newton–Krylov solver using lax.scan.

    If `precond` is provided, it should be a callable of the form
    `precond(x)` that returns a linear preconditioner `M(v)`.
    Otherwise a simple diagonal Jacobian preconditioner is built.
    """

    def Jv(x, v):
        return jax.jvp(fun, (x,), (v,))[1]

    def diagonal_preconditioner(x):
        J = jax.jacfwd(fun)(x)
        diag = jnp.diag(J)
        inv_diag = jnp.where(jnp.abs(diag) > precond_eps,
                             1.0 / diag,
                             1.0)

        def M(v):
            return inv_diag * v

        return M

    def body_fun(state, _):
        x, converged = state

        # skip computation if already converged
        def compute(x):
            F = fun(x)

            def matvec(v):
                return Jv(x, v)

            M = precond(x) if precond is not None else diagonal_preconditioner(x)

            delta, _ = gmres(
                matvec,
                -F,
                tol=newton_tol,
                maxiter=newton_maxiter,
                M=M
            )

            x_new = x + alpha * delta
            normF = jnp.linalg.norm(fun(x_new))
            converged_new = normF < tol

            return x_new, converged_new

        x_new, converged_new = jax.lax.cond(converged,
                                             lambda x: (x, True),
                                             compute,
                                             x)
        return (x_new, converged_new), None

    # initial state: x0 and not converged
    init_state = (x0, False)

    # scan over max_iters steps
    final_state, _ = lax.scan(body_fun, init_state, xs=None, length=max_iters)

    x_final, _ = final_state
    return x_final