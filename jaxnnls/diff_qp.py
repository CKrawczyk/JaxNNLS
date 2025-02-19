import jax
import jax.numpy as jnp

from .pdip import factorize_kkt, solve_kkt_rhs, solve_nnls
from .pdip_relaxed import solve_relaxed_nnls


@jax.custom_vjp
def solve_nnls_primal(Q, q, target_kappa=1e-3):
    '''Solve the non-negative least square problem with the ability
    to take a (relaxed) gradient.

    Parameters
    ----------
    Q : jax.numpy.array
        (n, n) positive definite matrix.
    q : jax.numpy.array
        (n,) vector
    target_kappa : float
        target relaxation parameter used for the gradient

    Returns
    -------
    jax.numpy.array
        (n,) solution x to Qx=q such that x >= 0
    '''
    return solve_nnls(Q, q)[0]


def solve_nnls_primal_forward(Q, q, target_kappa=1e-3):
    '''Custom forward pass derivative'''
    # solve nnls as normal and return primal solution
    x, s, z, _, _ = solve_nnls(Q, q)
    # relax this solution by taking vanilla Newton steps on relaxed KKT
    xr, sr, zr, _, _ = solve_relaxed_nnls(Q, q, x, s, z, target_kappa=target_kappa)
    # return real solution x, and save the relaxed variables for backward
    return x, (Q, q, xr, sr, zr)


def diff_nnls(Q, z, s, lam, dl_dz):
    '''Implicit derivatives'''
    P_inv_vec, L_H = factorize_kkt(Q, s, lam)
    dz, _, _ = solve_kkt_rhs(s, lam, P_inv_vec, L_H, dl_dz, 0, 0)
    dl_dQ = 0.5 * (jnp.outer(dz, z) + jnp.outer(z, dz))
    dl_dq = -dz
    return dl_dQ, dl_dq


def solve_nnls_primal_backward(res, input_grad):
    '''Custom backwards pass derivative'''
    # unpack relaxed solution
    Q, _, xr, sr, zr = res
    # return all the normal derivatives, then None's for kwargs
    return (*diff_nnls(Q, xr, sr, zr, input_grad), None)


solve_nnls_primal.defvjp(solve_nnls_primal_forward, solve_nnls_primal_backward)
