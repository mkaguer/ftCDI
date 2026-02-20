import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import gmres

__all__ = ["newton_krylov"]


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
