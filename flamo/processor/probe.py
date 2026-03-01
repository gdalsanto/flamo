"""
Autograd-based derivative helpers for z-plane probing.

Provides:
    - ``probe_points``: Vectorized probe over multiple z-plane points.
    - ``probe_with_derivative``: Compute H(z) and dH/dz via PyTorch native complex autograd.
    - ``complex_derivative_scalar``: Holomorphic dF/dz for scalar F(z) (native backward + conj).
    - ``complex_derivative``: Holomorphic dF/dz for matrix F(z) (native backward + conj).
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


def _ensure_complex_tensor(z: torch.Tensor) -> torch.Tensor:
    """Return a complex tensor with requires_grad, same device/dtype as z if complex."""
    if not isinstance(z, torch.Tensor):
        z = torch.as_tensor(z, dtype=torch.complex128)
    z = z.detach().clone()
    if not z.is_complex():
        z = torch.complex(z, torch.zeros_like(z))
    return z.to(torch.complex128).requires_grad_(True)


def complex_derivative_scalar(
    eval_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    create_graph: bool = False,
) -> tuple:
    r"""
    Holomorphic derivative dF/dz for a callable ``eval_fn(z) -> complex scalar``.

    Uses PyTorch native complex autograd: backward gives the conjugate Wirtinger
    derivative, so dF/dz = conj(z.grad) for holomorphic F.

    Args:
        eval_fn: Callable that takes a complex scalar tensor and returns a
            complex scalar tensor.
        z: Scalar complex tensor (the z-plane evaluation point).
        create_graph: Preserve autograd graph for higher-order derivatives.

    Returns:
        tuple: ``(F_val, dF_dz)`` where both are complex scalars (0-dim tensors).
    """
    z_var = _ensure_complex_tensor(z)
    F = eval_fn(z_var)
    one = torch.ones((), device=z_var.device, dtype=z_var.dtype)
    dF_dz = torch.autograd.grad(
        F, z_var, grad_outputs=one, create_graph=create_graph, retain_graph=create_graph
    )[0].conj()
    if create_graph:
        return F, dF_dz
    return F.detach(), dF_dz.detach()


def complex_derivative(
    eval_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    create_graph: bool = False,
) -> tuple:
    r"""
    Holomorphic derivative dF/dz for a callable ``eval_fn(z) -> Tensor`` (matrix output).

    Uses PyTorch native complex autograd per element: for each (i,j), grad of F[i,j]
    w.r.t. z is the conjugate Wirtinger derivative; conj gives dF_ij/dz.

    Args:
        eval_fn: Callable that takes a complex scalar tensor and returns a
            2-D complex tensor of shape ``(N, M)``.
        z: Scalar complex tensor (the z-plane evaluation point).
        create_graph: Preserve autograd graph for higher-order derivatives.

    Returns:
        tuple: ``(F_val, dF_dz)`` where ``F_val`` is the detached function
            value and ``dF_dz`` is the holomorphic derivative, both shape ``(N, M)``.
    """
    z_var = _ensure_complex_tensor(z)
    F = eval_fn(z_var)
    n_rows, n_cols = F.shape
    one = torch.ones((), device=z_var.device, dtype=z_var.dtype)
    dF_dz_list = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            g = torch.autograd.grad(
                F[i, j], z_var, grad_outputs=one, retain_graph=True,
                create_graph=create_graph,
            )[0].conj()
            row.append(g)
        dF_dz_list.append(torch.stack(row))
    dF_dz = torch.stack(dF_dz_list)
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

    Uses PyTorch native complex autograd via :func:`complex_derivative`.

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

    return complex_derivative(_eval, z, create_graph=create_graph)
