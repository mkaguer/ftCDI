import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import gmres

__all__ = ["newton_krylov",
           "newton_krylov_scan"]


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

            delta, _ = gmres(
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
