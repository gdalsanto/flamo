"""
Autograd-based derivative helpers for z-plane probing.

Provides:
    - ``probe_points``: Vectorized probe over multiple z-plane points.
    - ``probe_with_derivative``: Compute H(z) and dH/dz via Wirtinger calculus.
    - ``wirtinger_derivative``: Generic Wirtinger dF/dz for any callable F(z).
"""

import torch
from typing import Callable, Optional


def probe_points(
    module,
    z_points: torch.Tensor,
    include_shell_io: bool = False,
) -> torch.Tensor:
    r"""
    Evaluate transfer matrix at multiple z-plane points.

    Args:
        module: A FLAMO module with a ``probe(z)`` method.
        z_points: 1-D complex tensor of z-plane evaluation points.
        include_shell_io: Passed to Shell.probe if applicable.

    Returns:
        torch.Tensor: ``(len(z_points), N_out, N_in)`` complex tensor.
    """
    results = []
    for z in z_points:
        if hasattr(module, '_Shell__core'):
            H = module.probe(z, include_shell_io=include_shell_io)
        else:
            H = module.probe(z)
        results.append(H)
    return torch.stack(results, dim=0)


def _compute_element_derivative(u_ij, v_ij, x, y, create_graph):
    """Compute Wirtinger dH/dz for a single (i,j) element."""
    has_u_grad = u_ij.grad_fn is not None
    has_v_grad = v_ij.grad_fn is not None

    if has_u_grad:
        grads_u = torch.autograd.grad(
            u_ij, [x, y],
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
        du_dx = grads_u[0] if grads_u[0] is not None else torch.zeros_like(x)
        du_dy = grads_u[1] if grads_u[1] is not None else torch.zeros_like(y)
    else:
        du_dx = torch.zeros_like(x)
        du_dy = torch.zeros_like(y)

    if has_v_grad:
        grads_v = torch.autograd.grad(
            v_ij, [x, y],
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
        dv_dx = grads_v[0] if grads_v[0] is not None else torch.zeros_like(x)
        dv_dy = grads_v[1] if grads_v[1] is not None else torch.zeros_like(y)
    else:
        dv_dx = torch.zeros_like(x)
        dv_dy = torch.zeros_like(y)

    real_part = 0.5 * (du_dx + dv_dy)
    imag_part = 0.5 * (dv_dx - du_dy)
    return torch.complex(real_part, imag_part)


def wirtinger_derivative(
    eval_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    create_graph: bool = False,
) -> tuple:
    r"""
    Generic Wirtinger derivative dF/dz for any callable ``eval_fn(z) -> Tensor``.

    Constructs differentiable real/imag components of z, evaluates ``eval_fn``,
    and computes:

    .. math::

        \frac{dF}{dz} = \frac{1}{2}\left(
            \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}
        \right) + \frac{j}{2}\left(
            \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
        \right)

    where F = u + jv and z = x + jy.

    Args:
        eval_fn: Callable that takes a complex scalar tensor and returns a
            2-D complex tensor of shape ``(N, M)``.
        z: Scalar complex tensor (the z-plane evaluation point).
        create_graph: Preserve autograd graph for higher-order derivatives.

    Returns:
        tuple: ``(F_val, dF_dz)`` where ``F_val`` is the detached function
            value and ``dF_dz`` is the Wirtinger derivative, both with shape
            ``(N, M)``. When ``create_graph=True``, tensors retain grad_fn.
    """
    x = z.real.detach().clone().to(torch.float64).requires_grad_(True)
    y = z.imag.detach().clone().to(torch.float64).requires_grad_(True)
    z_reconst = torch.complex(x, y)

    F = eval_fn(z_reconst)

    u = F.real
    v = F.imag
    n_rows, n_cols = F.shape

    rows = []
    for i in range(n_rows):
        cols = []
        for j in range(n_cols):
            cols.append(_compute_element_derivative(
                u[i, j], v[i, j], x, y, create_graph,
            ))
        rows.append(torch.stack(cols))
    dF_dz = torch.stack(rows)

    if create_graph:
        return F, dF_dz
    return F.detach(), dF_dz.detach()


def probe_with_derivative(
    module,
    z: torch.Tensor,
    include_shell_io: bool = False,
    create_graph: bool = False,
) -> tuple:
    r"""
    Compute H(z) and dH/dz at a single complex z-plane point using autograd.

    Uses the Wirtinger derivative reconstruction via :func:`wirtinger_derivative`.

    Args:
        module: A FLAMO module with a ``probe(z)`` method.
        z: A scalar complex tensor (the z-plane evaluation point).
        include_shell_io: Passed to Shell.probe if applicable.
        create_graph: Whether to create a computational graph for higher-order
            derivatives. Default: False.

    Returns:
        tuple: ``(H, dH_dz)`` where both are complex tensors of shape
            ``(N_out, N_in)``.
    """
    def _eval(z_val):
        if hasattr(module, '_Shell__core'):
            return module.probe(z_val, include_shell_io=include_shell_io)
        return module.probe(z_val)

    return wirtinger_derivative(_eval, z, create_graph=create_graph)
