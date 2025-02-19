# modification of https://github.com/kevin-tracy/qpax/blob/main/qpax/pdip_relaxed.py
import jax
import jax.numpy as jnp

from .pdip import (
    factorize_kkt,
    solve_kkt_rhs,
    ort_line_search,
    MAX_ITER,
    EPSILON
)


def pdip_pc_step_relaxed(inputs):
    '''One step of the relaxed predictor-corrector PDIP algorithm.

    Parameters
    ----------
    inputs : tuple
        Tuple of the current state (Q, q, x, s, z, solver_tol, converged, pdip_iter, target_kappa)

    Returns
    -------
    tuple
        Updated state (Q, q, x, s, z, solver_tol, converged, pdip_iter, target_kappa)
    '''
    Q, q, x, s, z, solver_tol, converged, pdip_iter, target_kappa = inputs

    r1 = Q @ x - q - z
    r2 = s * z - target_kappa
    r3 = s - x

    kkt_res = jnp.concatenate((r1, r2, r3))
    converged = jax.lax.select(
        jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol,
        1,
        0
    )

    P_inv_vec, L_H = factorize_kkt(Q, s, z)
    dx, ds, dz = solve_kkt_rhs(s, z, P_inv_vec, L_H, r1, r2, r3)

    alpha = 0.99 * jnp.min(jnp.array([
        1.0,
        0.99 * ort_line_search(s, ds),
        0.99 * ort_line_search(z, dz)
    ]))

    x = x + alpha * dx
    s = s + alpha * ds
    z = z + alpha * dz

    return Q, q, x, s, z, solver_tol, converged, pdip_iter + 1, target_kappa


# MODIFICATION: this function is static, move it outside the solver
# so it does not trigger re-compiles
def converged_check_relaxed(inputs):
    '''Check if the relaxed PDIP algorithm has converged

    Parameters
    ----------
    inputs : tuple
        Tuple of the current state (Q, q, x, s, z, solver_tol, converged, pdip_iter, target_kappa)

    Returns
    -------
    bool
        True if converged or MAX_ITER reached, False otherwise
    '''
    _, _, _, _, _, _, converged, pdip_iter, _ = inputs
    return jnp.logical_and(
        pdip_iter < MAX_ITER,
        converged == 0
    )


def solve_relaxed_nnls(Q, q, x, s, z, target_kappa=1e-3):
    '''Solve the relaxed non-negative least square problem.

    Parameters
    ----------
    Q : jax.numpy.array
        (n, n) positive definite matrix.
    q : jax.numpy.array
        (n,) primal vector
    s : jax.numpy.array
        (n,) slack vector
    z : jax.numpy.array
        (n,) dual vector
    target_kappa : float
        target relaxation parameter

    Returns
    -------
    x : jax.numpy.array
        (n,) relaxed solution x to Qx=q such that x >= 0
    s : jax.numpy.array
        (n,) slack variable at the relaxed solution
    z : jax.numpy.array
        (n,) dual variable at the relaxed solution
    converged : int
        1 if the algorithm converged, 0 otherwise
    pdip_iter : int
        The number of relaxed PDIP iterations taken
    '''
    # MODIFICATION: set the tolerance based on the size of the problem
    # and the precision being used
    solver_tol = Q.shape[0] * EPSILON
    solver_tol = jax.lax.min(solver_tol, 1e-2)
    init_inputs = (Q, q, x, s, z, solver_tol, 0, 0, target_kappa)
    outputs = jax.lax.while_loop(
        converged_check_relaxed,
        pdip_pc_step_relaxed,
        init_inputs
    )
    _, _, x, s, z, _, converged, pdip_iter, _ = outputs
    return x, s, z, converged, pdip_iter
