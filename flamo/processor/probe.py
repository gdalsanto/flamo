"""
Z-plane probing helpers.

Provides:
    - ``probe_points``: Vectorized probe over multiple z-plane points.

Derivatives (e.g. dH/dz, dP/dz) are built by callers (e.g. pyFDN) via
torch.autograd.functional.jvp or grad on module.probe / Recursion._eval_characteristic.
"""

import torch


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
