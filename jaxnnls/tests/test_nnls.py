import unittest
import jax
import jax.numpy as jnp

from jaxnnls import solve_nnls_primal


Z = jnp.array([
    [73.0, 71.0, 52.0],
    [87.0, 74.0, 46.0],
    [72.0, 2.0, 7.0],
    [80.0, 89.0, 71.0],
])
x = jnp.array([49.0, 67.0, 68.0, 20.0])
Q = Z.T.dot(Z)
q = Z.T.dot(x)


@jax.jit
def loss(Q, q):
    x = solve_nnls_primal(Q, q)
    x_bar = jnp.ones_like(x)
    res = x - x_bar
    return jnp.dot(res, res)


class TestNNLS(unittest.TestCase):
    def test_primal(self):
        expected = jnp.array([0.64953844, 0.0, 0.0])
        result = solve_nnls_primal(Q, q)
        self.assertAlmostEqual(jnp.max(jnp.abs(expected - result)), 0.0, places=6)

    def test_grad(self):
        # expected values calculated using finite difference
        # only the first element of each is non-zero
        expected_dQ = jnp.zeros_like(Q)
        expected_dQ = expected_dQ.at[0, 0].set(1.85963773e-05)

        expected_dq = jnp.zeros_like(q)
        expected_dq = expected_dq.at[0].set(-2.863014e-05)

        result_dQ, result_dq = jax.grad(loss, argnums=(0, 1))(Q, q)
        residual_Q = jnp.max(jnp.abs(expected_dQ - result_dQ))
        residual_q = jnp.max(jnp.abs(expected_dq - result_dq))
        self.assertAlmostEqual(residual_Q, 0, places=6)
        self.assertAlmostEqual(residual_q, 0, places=6)
