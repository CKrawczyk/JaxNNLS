'''
The jaxnnls package provides functions to solve non-negative
least squares problems.
'''

from .pdip import solve_nnls
from .diff_qp import solve_nnls_primal

__version__ = '1.0.0'
