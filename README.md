# JaxNNLS

This package can be used for solving non-negative least square (NNLS) problems of the following form:

```math
\begin{align*}
\underset{x}{\text{minimize}} & \quad \frac{1}{2}x^TQx - q^Tx \\
\text{subject to} & \quad  x \geq 0
\end{align*}
```

where `Q` is positive definite. Or equivalently

```math
\begin{align*}
\underset{x}{{\text{solve}}} & \quad Ax = b \\
\text{subject to} & \quad  x \geq 0
\end{align*}
```

when you set

```math
\begin{align*}
Q&=A^TA \\
q&=A^Tb
\end{align*}
```

This solver can be combined with JAX's `jit` and `vmap` functionality, as well as differentiated with reverse-mode `grad`. 

The NNLS problem is solved with a primal-dual interior point algorithm.  This code is a **modification** on the [qpax](https://github.com/kevin-tracy/qpax/blob/main/README.md) package, but in the special case of NNLS.  Because of the simplifications in this special case the resulting code is significantly faster when $Q$ large in size.

As with the `qpax` code, derivative smoothing can be applied to the gradients.

Link to the [documentation](https://ckrawczyk.github.io/JaxNNLS/).
