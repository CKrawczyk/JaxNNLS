# Modification of https://github.com/kevin-tracy/qpax/blob/main/qpax/pdip.py

# Original license
# Copyright (c) 2023 Kevin Tracy

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modifications by Coleman Krawczyk 2025

import jax
import jax.numpy as jnp
import jax.scipy as jsp

MAX_ITER = 50
# MODIFICATION: base tolerance on the current precision
EPSILON = jnp.finfo(jnp.ones(1).dtype).eps * 5e3

# MODIFICATION: use Cholesky decomposition
# it is significantly faster
factor = jsp.linalg.cho_factor
solve = jsp.linalg.cho_solve

# MODIFICATION: follow qpax but set A = b = h = 0 and G = -I


def initialize(Q, q):
    '''Initialize primal and dual variables

    Parameters
    ----------
    Q : jax.numpy.array
        (n, n) positive definite matrix.
    q : jax.numpy.array
        (n,) vector

    Returns
    -------
    x : jax.numpy.array
        (n,) Initial primal variable
    s : jax.numpy.array
        (n,) Initial slack variable
    z : jax.numpy.array
        (n,) Initial dual variable
    '''
    H = Q + jnp.eye(Q.shape[0])
    L_H = factor(H)

    r1 = q
    x = solve(L_H, r1)
    z = -x

    alpha_p = -jnp.min(-z)
    # MODIFICATION: use `jax.lax.select` directly
    # since `alpha_p` and `alpha_d` are scalars
    s = jax.lax.select(
        alpha_p < 0,
        -z,
        1 + alpha_p - z
    )

    alpha_d = -jnp.min(z)
    z = jax.lax.select(
        alpha_d >= 0,
        1 + alpha_d + z,
        z
    )
    return x, s, z


def factorize_kkt(Q, s, z):
    '''Cache factorize matrix values for solving KKT conditions quickly

    Parameters
    ----------
    Q : jax.numpy.array
        (n, n) positive definite matrix.
    s : jax.numpy.array
        (n,) slack vector
    z : jax.numpy.array
        (n,) dual vector

    Returns
    -------
    P_inv_vec : jax.numpy.array
        (n,)  ratio of z and s (inverse of the diag P matrix)
    L_H : jax.numpy.array
        The Cholesky decomposition of the H matrix (Q with P_inv_vec added down the diag)
    '''
    P_inv_vec = z / s
    L_H = factor(
        Q + jnp.diag(P_inv_vec)
    )
    return P_inv_vec, L_H


# MODIFICATION: move the negatives signs for `v1, v2, v3`
# inside the function (I think it makes the code
# later on look cleaner)
def solve_kkt_rhs(s, z, P_inv_vec, L_H, v1, v2, v3):
    '''Solve the right hand side or the KKT conditions

    Parameters
    ----------
    s : jnp.array
        (n,) slack vector
    z : jnp.array
        (n,) dual vector
    P_inv_vec : jax.numpy.array
        (n,)  inverse of the diag P matrix
    L_H : jax.numpy.array
        The Cholesky decomposition of the H matrix
    v1 : jax.numpy.array
        (n,) negative residual 1
    v2 : jax.numpy.array
        (n,) negative residual 2
    v3 : jax.numpy.array
        (n,) negative residual 3

    Returns
    -------
    dx : jax.numpy.array
        (n,) step size for primal vector
    ds : jax.numpy.array
        (n,) step size for slack vector
    dz : jax.numpy.array
        (n,) step size for dual vector
    '''
    r2 = -v3 + v2 / z
    p1 = -v1 - P_inv_vec * r2
    dx = solve(L_H, p1)
    ds = dx - v3
    dz = -(v2 + z * ds) / s
    return dx, ds, dz


# MODIFICATION: `jnp.where` is already vectorized, no need
# to re-vmap it
def ort_line_search(x, dx):
    '''Maximum alpha <=1 such that x + alpha * dx >= 0

    Parameters
    ----------
    x : jax.numpy.array
        (n,) vector
    dx : jax.numpy.array
        (n,) gradient of vector x

    Returns
    -------
    float
        Maximum alpha <=1 such that x + alpha * dx >= 0
    '''
    min_batch = jnp.min(jnp.where(dx < 0, -x / dx, jnp.inf))
    return jnp.min(jnp.array([1.0, min_batch]))


def centering_params(s, z, ds_a, dz_a):
    '''duality gap + cc term in predictor-corrector PDIP'''
    mu = jnp.dot(s, z) / len(s)
    alpha = jnp.min(jnp.array([
        ort_line_search(s, ds_a),
        ort_line_search(z, dz_a)
    ]))
    sigma = (jnp.dot(s + alpha * ds_a, z + alpha * dz_a) / jnp.dot(s, z))**3
    return sigma, mu


def pdip_pc_step(inputs):
    '''One step of the predictor-corrector PDIP algorithm.

    Parameters
    ----------
    inputs : tuple
        Tuple of the current state (Q, q, x, s, z, solver_tol, converged, pdip_iter)

    Returns
    -------
    tuple
        Updated state (Q, q, x, s, z, solver_tol, converged, pdip_iter)
    '''
    Q, q, x, s, z, solver_tol, converged, pdip_iter = inputs

    r1 = Q @ x - q - z
    r2 = s * z
    r3 = s - x

    kkt_res = jnp.concatenate((r1, r2, r3))
    # MODIFICATION: use `jax.lax.select` directly as the
    # condition is a scalar
    converged = jax.lax.select(
        jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol,
        1,
        0
    )

    P_inv_vec, L_H = factorize_kkt(Q, s, z)
    _, ds_a, dz_a = solve_kkt_rhs(s, z, P_inv_vec, L_H, r1, r2, r3)

    sigma, mu = centering_params(s, z, ds_a, dz_a)
    r2 = r2 - (sigma * mu - (ds_a * dz_a))
    dx, ds, dz = solve_kkt_rhs(s, z, P_inv_vec, L_H, r1, r2, r3)

    alpha = 0.99 * jnp.min(jnp.array([
        ort_line_search(s, ds),
        ort_line_search(z, dz)
    ]))

    x = x + alpha * dx
    s = s + alpha * ds
    z = z + alpha * dz

    return Q, q, x, s, z, solver_tol, converged, pdip_iter + 1


# MODIFICATION: this function is static, move it outside the solver
# so it does not trigger re-compiles
def converged_check(inputs):
    '''Check if the PDIP algorithm has converged

    Parameters
    ----------
    inputs : tuple
        Tuple of the current state (Q, q, x, s, z, solver_tol, converged, pdip_iter)

    Returns
    -------
    bool
        True if converged or MAX_ITER reached, False otherwise
    '''
    _, _, _, _, _, _, converged, pdip_iter = inputs
    return jnp.logical_and(
        pdip_iter < MAX_ITER,
        converged == 0
    )


def solve_nnls(Q, q):
    '''Solve the non-negative least square problem.

    Parameters
    ----------
    Q : jax.numpy.array
        (n, n) positive definite matrix.
    q : jax.numpy.array
        (n,) vector

    Returns
    -------
    x : jax.numpy.array
        (n,) solution x to Qx=q such that x >= 0
    s : jax.numpy.array
        (n,) slack variable at the solution
    z : jax.numpy.array
        (n,) dual variable at the solution
    converged : int
        1 if the algorithm converged, 0 otherwise
    pdip_iter : int
        The number of PDIP iterations taken
    '''
    x, s, z = initialize(Q, q)
    # MODIFICATION: set the tolerance based on the size of the problem
    # and the precision being used
    solver_tol = Q.shape[0] * EPSILON
    solver_tol = jax.lax.min(solver_tol, 1e-2)
    init_inputs = (Q, q, x, s, z, solver_tol, 0, 0)
    outputs = jax.lax.while_loop(
        converged_check,
        pdip_pc_step,
        init_inputs
    )
    _, _, x, s, z, _, converged, pdip_iter = outputs
    return x, s, z, converged, pdip_iter
