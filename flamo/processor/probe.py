"""
Autograd-based derivative helpers for z-plane probing.

Provides:
    - ``probe_points``: Vectorized probe over multiple z-plane points.
    - ``probe_with_derivative``: Compute H(z) and dH/dz via Wirtinger calculus.
"""

import torch
from typing import Optional


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


def probe_with_derivative(
    module,
    z: torch.Tensor,
    include_shell_io: bool = False,
    create_graph: bool = False,
) -> tuple:
    r"""
    Compute H(z) and dH/dz at a single complex z-plane point using autograd.

    Uses the Wirtinger derivative reconstruction:

    .. math::

        \frac{dH}{dz} = \frac{1}{2}\left(
            \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}
        \right) + \frac{j}{2}\left(
            \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
        \right)

    where H = u + jv and z = x + jy.

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
    x = z.real.detach().clone().to(torch.float64).requires_grad_(True)
    y = z.imag.detach().clone().to(torch.float64).requires_grad_(True)
    z_reconst = torch.complex(x, y)

    if hasattr(module, '_Shell__core'):
        H = module.probe(z_reconst, include_shell_io=include_shell_io)
    else:
        H = module.probe(z_reconst)

    u = H.real
    v = H.imag

    n_out, n_in = H.shape
    dH_dz = torch.zeros(n_out, n_in, dtype=torch.complex128, device=H.device)

    for i in range(n_out):
        for j in range(n_in):
            if u[i, j].requires_grad or u[i, j].grad_fn is not None:
                du_dx, du_dy = torch.autograd.grad(
                    u[i, j], [x, y],
                    retain_graph=True,
                    create_graph=create_graph,
                )
            else:
                du_dx = torch.zeros_like(x)
                du_dy = torch.zeros_like(y)

            if v[i, j].requires_grad or v[i, j].grad_fn is not None:
                dv_dx, dv_dy = torch.autograd.grad(
                    v[i, j], [x, y],
                    retain_graph=True,
                    create_graph=create_graph,
                )
            else:
                dv_dx = torch.zeros_like(x)
                dv_dy = torch.zeros_like(y)

            real_part = 0.5 * (du_dx + dv_dy)
            imag_part = 0.5 * (dv_dx - du_dy)
            dH_dz[i, j] = torch.complex(real_part, imag_part)

    return H.detach(), dH_dz.detach() if not create_graph else dH_dz
