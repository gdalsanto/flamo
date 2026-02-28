"""Tests for the z-plane probe API."""

import torch
import pytest
import math

from flamo.processor.dsp import (
    Gain, parallelGain, Matrix, Filter, parallelFilter,
    Delay, parallelDelay, SOSFilter, parallelSOSFilter,
    FFT, iFFT, Transform, Biquad, SVF,
)
from flamo.processor.system import Series, Recursion, Parallel, Shell
from flamo.processor.probe import probe_points, probe_with_derivative
from flamo.utils import to_complex

DTYPE = torch.float64
CDTYPE = torch.complex128
NFFT = 256


# ---------------------------------------------------------------------------
# 1. test_gain_probe_constant
# ---------------------------------------------------------------------------
class TestGainProbeConstant:
    def test_gain_probe_is_mapped_param(self):
        g = Gain(size=(3, 2), nfft=NFFT, dtype=DTYPE)
        z = torch.tensor(0.9 + 0.1j, dtype=CDTYPE)
        H = g.probe(z)
        expected = to_complex(g.map(g.param))
        assert H.shape == (3, 2)
        assert torch.allclose(H, expected)

    def test_gain_probe_independent_of_z(self):
        g = Gain(size=(2, 2), nfft=NFFT, dtype=DTYPE)
        z1 = torch.tensor(0.5 + 0.5j, dtype=CDTYPE)
        z2 = torch.tensor(0.9 - 0.3j, dtype=CDTYPE)
        assert torch.allclose(g.probe(z1), g.probe(z2))

    def test_parallel_gain_probe_is_diagonal(self):
        pg = parallelGain(size=(4,), nfft=NFFT, dtype=DTYPE)
        z = torch.tensor(0.8 + 0.2j, dtype=CDTYPE)
        H = pg.probe(z)
        assert H.shape == (4, 4)
        expected_diag = to_complex(pg.map(pg.param))
        assert torch.allclose(torch.diag(H), expected_diag)
        off_diag = H - torch.diag(torch.diag(H))
        assert torch.allclose(off_diag, torch.zeros_like(off_diag))

    def test_matrix_probe(self):
        m = Matrix(size=(3, 3), nfft=NFFT, matrix_type="random", dtype=DTYPE)
        z = torch.tensor(0.7 + 0.1j, dtype=CDTYPE)
        H = m.probe(z)
        expected = to_complex(m.map(m.param))
        assert H.shape == (3, 3)
        assert torch.allclose(H, expected)


# ---------------------------------------------------------------------------
# 2. test_delay_probe_matches_formula
# ---------------------------------------------------------------------------
class TestDelayProbeFormula:
    def test_integer_delay(self):
        d = Delay(
            size=(2, 2), max_len=50, isint=True, nfft=NFFT,
            fs=48000, unit=100, dtype=DTYPE,
        )
        z = torch.tensor(0.95 + 0.1j, dtype=CDTYPE)
        H = d.probe(z)
        m_vals = d.s2sample(d.map(d.param)).round()
        expected = (d.gamma ** m_vals) * (z ** (-m_vals))
        assert H.shape == (2, 2)
        assert torch.allclose(H, expected, atol=1e-12)

    def test_parallel_delay_is_diagonal(self):
        pd = parallelDelay(
            size=(3,), max_len=50, isint=True, nfft=NFFT,
            fs=48000, unit=100, dtype=DTYPE,
        )
        z = torch.tensor(0.9 + 0.05j, dtype=CDTYPE)
        H = pd.probe(z)
        assert H.shape == (3, 3)
        m_vals = pd.s2sample(pd.map(pd.param)).round()
        expected_diag = (pd.gamma ** m_vals) * (z ** (-m_vals))
        assert torch.allclose(torch.diag(H), expected_diag, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. test_filter_probe_matches_poly_eval
# ---------------------------------------------------------------------------
class TestFilterProbePolyEval:
    def test_filter_probe_matches_manual_poly(self):
        K, Nout, Nin = 5, 2, 3
        f = Filter(size=(K, Nout, Nin), nfft=NFFT, dtype=DTYPE)
        z = torch.tensor(0.8 + 0.2j, dtype=CDTYPE)

        H = f.probe(z)
        coeff = f.map(f.param)
        H_manual = torch.zeros(Nout, Nin, dtype=CDTYPE)
        for k in range(K):
            H_manual += to_complex(coeff[k]) * (f.gamma ** k) * (z ** (-k))

        assert H.shape == (Nout, Nin)
        assert torch.allclose(H, H_manual, atol=1e-12)

    def test_parallel_filter_probe_is_diagonal(self):
        K, N = 4, 3
        pf = parallelFilter(size=(K, N), nfft=NFFT, dtype=DTYPE)
        z = torch.tensor(0.7 + 0.3j, dtype=CDTYPE)
        H = pf.probe(z)
        assert H.shape == (N, N)

        coeff = pf.map(pf.param)
        h_manual = torch.zeros(N, dtype=CDTYPE)
        for k in range(K):
            h_manual += to_complex(coeff[k]) * (pf.gamma ** k) * (z ** (-k))

        assert torch.allclose(torch.diag(H), h_manual, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. test_sos_probe_matches_section_formula
# ---------------------------------------------------------------------------
class TestSOSProbeFormula:
    def test_sos_filter_single_section(self):
        sf = SOSFilter(size=(2, 1), n_sections=1, nfft=NFFT, fs=48000, dtype=DTYPE)
        with torch.no_grad():
            sf.param[:, 0, ...] = 1.0
            sf.param[:, 1, ...] = 0.5
            sf.param[:, 2, ...] = 0.2
            sf.param[:, 3, ...] = 1.0
            sf.param[:, 4, ...] = -0.3
            sf.param[:, 5, ...] = 0.1

        z = torch.tensor(0.85 + 0.15j, dtype=CDTYPE)
        H = sf.probe(z)
        assert H.shape == (2, 1)

        mapped = sf.map(sf.param)
        gamma = sf.alias_envelope_dcy
        z_inv = z ** (-1)
        b0, b1, b2 = mapped[0, 0, ...], mapped[0, 1, ...], mapped[0, 2, ...]
        a0, a1, a2 = mapped[0, 3, ...], mapped[0, 4, ...], mapped[0, 5, ...]
        B = to_complex(b0) * gamma[0] + to_complex(b1) * gamma[1] * z_inv + to_complex(b2) * gamma[2] * z_inv**2
        A = to_complex(a0) * gamma[0] + to_complex(a1) * gamma[1] * z_inv + to_complex(a2) * gamma[2] * z_inv**2
        H_expected = B / A
        assert torch.allclose(H, H_expected, atol=1e-10)

    def test_sos_filter_multi_section(self):
        sf = SOSFilter(size=(1, 1), n_sections=3, nfft=NFFT, fs=48000, dtype=DTYPE)
        z = torch.tensor(0.9 + 0.1j, dtype=CDTYPE)
        H = sf.probe(z)
        assert H.shape == (1, 1)

        mapped = sf.map(sf.param)
        gamma = sf.alias_envelope_dcy
        z_inv = z ** (-1)
        H_manual = torch.ones(1, 1, dtype=CDTYPE)
        for k in range(3):
            b0, b1, b2 = mapped[k, 0, ...], mapped[k, 1, ...], mapped[k, 2, ...]
            a0, a1, a2 = mapped[k, 3, ...], mapped[k, 4, ...], mapped[k, 5, ...]
            B_k = to_complex(b0) * gamma[0] + to_complex(b1) * gamma[1] * z_inv + to_complex(b2) * gamma[2] * z_inv**2
            A_k = to_complex(a0) * gamma[0] + to_complex(a1) * gamma[1] * z_inv + to_complex(a2) * gamma[2] * z_inv**2
            H_manual = H_manual * B_k / A_k
        assert torch.allclose(H, H_manual, atol=1e-10)

    def test_parallel_sos_filter_diagonal(self):
        psf = parallelSOSFilter(size=(2,), n_sections=2, nfft=NFFT, fs=48000, dtype=DTYPE)
        z = torch.tensor(0.88 + 0.12j, dtype=CDTYPE)
        H = psf.probe(z)
        assert H.shape == (2, 2)
        off_diag = H - torch.diag(torch.diag(H))
        assert torch.allclose(off_diag, torch.zeros_like(off_diag))


# ---------------------------------------------------------------------------
# 5. test_series_parallel_recursion_probe_matches_manual_matrix_formula
# ---------------------------------------------------------------------------
class TestSystemProbe:
    def test_series_probe(self):
        g1 = Gain(size=(3, 2), nfft=NFFT, dtype=DTYPE)
        g2 = Gain(size=(4, 3), nfft=NFFT, dtype=DTYPE)
        s = Series(g1, g2)
        z = torch.tensor(0.8 + 0.1j, dtype=CDTYPE)
        H = s.probe(z)
        H1 = g1.probe(z)
        H2 = g2.probe(z)
        assert torch.allclose(H, H2 @ H1, atol=1e-12)

    def test_parallel_sum_probe(self):
        g1 = Gain(size=(3, 2), nfft=NFFT, dtype=DTYPE)
        g2 = Gain(size=(3, 2), nfft=NFFT, dtype=DTYPE)
        p = Parallel(g1, g2, sum_output=True)
        z = torch.tensor(0.8 + 0.1j, dtype=CDTYPE)
        H = p.probe(z)
        assert torch.allclose(H, g1.probe(z) + g2.probe(z), atol=1e-12)

    def test_parallel_cat_probe(self):
        g1 = Gain(size=(3, 2), nfft=NFFT, dtype=DTYPE)
        g2 = Gain(size=(4, 2), nfft=NFFT, dtype=DTYPE)
        p = Parallel(g1, g2, sum_output=False)
        z = torch.tensor(0.8 + 0.1j, dtype=CDTYPE)
        H = p.probe(z)
        expected = torch.cat([g1.probe(z), g2.probe(z)], dim=0)
        assert H.shape == (7, 2)
        assert torch.allclose(H, expected, atol=1e-12)

    def test_recursion_probe(self):
        N = 3
        fF = Gain(size=(N, N), nfft=NFFT, dtype=DTYPE)
        fB = Gain(size=(N, N), nfft=NFFT, dtype=DTYPE)
        with torch.no_grad():
            fB.param.mul_(0.1)  # keep feedback small for stability
        rec = Recursion(fF, fB)
        z = torch.tensor(0.9 + 0.05j, dtype=CDTYPE)
        H = rec.probe(z)

        F = fF.probe(z)
        B = fB.probe(z)
        I = torch.eye(N, dtype=CDTYPE)
        H_expected = torch.linalg.solve(I - F @ B, F)
        assert torch.allclose(H, H_expected, atol=1e-10)

    def test_series_with_filter_and_delay(self):
        f = Filter(size=(3, 2, 2), nfft=NFFT, dtype=DTYPE)
        d = Delay(size=(2, 2), max_len=10, isint=True, nfft=NFFT, fs=48000, unit=100, dtype=DTYPE)
        s = Series(d, f)
        z = torch.tensor(0.85 + 0.1j, dtype=CDTYPE)
        H = s.probe(z)
        H_d = d.probe(z)
        H_f = f.probe(z)
        assert torch.allclose(H, H_f @ H_d, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. test_probe_with_derivative_matches_finite_difference
# ---------------------------------------------------------------------------
class TestProbeWithDerivative:
    def _finite_diff_derivative(self, module, z, eps=1e-7):
        """Compute dH/dz via central finite differences."""
        z_xp = z + eps
        z_xn = z - eps
        z_yp = z + 1j * eps
        z_yn = z - 1j * eps

        H_xp = module.probe(z_xp)
        H_xn = module.probe(z_xn)
        H_yp = module.probe(z_yp)
        H_yn = module.probe(z_yn)

        du_dx = (H_xp.real - H_xn.real) / (2 * eps)
        dv_dx = (H_xp.imag - H_xn.imag) / (2 * eps)
        du_dy = (H_yp.real - H_yn.real) / (2 * eps)
        dv_dy = (H_yp.imag - H_yn.imag) / (2 * eps)

        real_part = 0.5 * (du_dx + dv_dy)
        imag_part = 0.5 * (dv_dx - du_dy)
        return torch.complex(real_part.to(torch.float64), imag_part.to(torch.float64))

    def test_gain_derivative_is_zero(self):
        g = Gain(size=(2, 2), nfft=NFFT, dtype=DTYPE)
        z = torch.tensor(0.9 + 0.1j, dtype=CDTYPE)
        _, dH = probe_with_derivative(g, z)
        assert torch.allclose(dH, torch.zeros_like(dH), atol=1e-10)

    def test_delay_derivative_matches_fd(self):
        d = Delay(
            size=(2, 2), max_len=20, isint=True, nfft=NFFT,
            fs=48000, unit=100, dtype=DTYPE,
        )
        z = torch.tensor(0.9 + 0.1j, dtype=CDTYPE)
        _, dH_auto = probe_with_derivative(d, z)
        dH_fd = self._finite_diff_derivative(d, z)
        assert torch.allclose(dH_auto, dH_fd, atol=1e-5)

    def test_filter_derivative_matches_fd(self):
        f = Filter(size=(3, 2, 2), nfft=NFFT, dtype=DTYPE)
        z = torch.tensor(0.85 + 0.15j, dtype=CDTYPE)
        _, dH_auto = probe_with_derivative(f, z)
        dH_fd = self._finite_diff_derivative(f, z)
        assert torch.allclose(dH_auto, dH_fd, atol=1e-5)

    def test_recursion_derivative_matches_fd(self):
        N = 2
        fF = Gain(size=(N, N), nfft=NFFT, dtype=DTYPE)
        fB_delay = Delay(
            size=(N, N), max_len=10, isint=True, nfft=NFFT,
            fs=48000, unit=100, dtype=DTYPE,
        )
        with torch.no_grad():
            fF.param.fill_(0.5)
        rec = Recursion(fF, fB_delay)
        z = torch.tensor(0.9 + 0.05j, dtype=CDTYPE)
        _, dH_auto = probe_with_derivative(rec, z)
        dH_fd = self._finite_diff_derivative(rec, z)
        assert torch.allclose(dH_auto, dH_fd, atol=1e-5)

    def test_sos_derivative_matches_fd(self):
        sf = SOSFilter(size=(1, 1), n_sections=2, nfft=NFFT, fs=48000, dtype=DTYPE)
        z = torch.tensor(0.9 + 0.1j, dtype=CDTYPE)
        _, dH_auto = probe_with_derivative(sf, z)
        dH_fd = self._finite_diff_derivative(sf, z)
        assert torch.allclose(dH_auto, dH_fd, atol=1e-5)

    def test_trainable_gain_derivative_is_zero(self):
        """Trainable Gain (requires_grad=True) has grad_fn from params but
        probe output doesn't depend on z. Must not crash (issue: allow_unused)."""
        g = Gain(size=(2, 2), nfft=NFFT, dtype=DTYPE, requires_grad=True)
        z = torch.tensor(0.9 + 0.1j, dtype=CDTYPE)
        _, dH = probe_with_derivative(g, z)
        assert torch.allclose(dH, torch.zeros_like(dH), atol=1e-10)

    def test_create_graph_preserves_autograd(self):
        """When create_graph=True, dH_dz must retain a grad_fn."""
        d = Delay(
            size=(1, 1), max_len=10, isint=True, nfft=NFFT,
            fs=48000, unit=100, dtype=DTYPE,
        )
        z = torch.tensor(0.9 + 0.1j, dtype=CDTYPE)
        H, dH_dz = probe_with_derivative(d, z, create_graph=True)
        assert dH_dz.grad_fn is not None or all(
            dH_dz[i, j].grad_fn is not None
            for i in range(dH_dz.shape[0])
            for j in range(dH_dz.shape[1])
        )


# ---------------------------------------------------------------------------
# 7. test_shell_probe_core_only_default
# ---------------------------------------------------------------------------
class TestShellProbeCoreOnly:
    def test_shell_core_only(self):
        core = Gain(size=(3, 2), nfft=NFFT, dtype=DTYPE)
        fft_in = FFT(nfft=NFFT, dtype=DTYPE)
        ifft_out = iFFT(nfft=NFFT, dtype=DTYPE)
        shell = Shell(core=core, input_layer=fft_in, output_layer=ifft_out)
        z = torch.tensor(0.8 + 0.2j, dtype=CDTYPE)

        H_shell = shell.probe(z)
        H_core = core.probe(z)
        assert torch.allclose(H_shell, H_core, atol=1e-12)

    def test_shell_core_series(self):
        g1 = Gain(size=(3, 2), nfft=NFFT, dtype=DTYPE)
        g2 = Gain(size=(4, 3), nfft=NFFT, dtype=DTYPE)
        core = Series(g1, g2)
        shell = Shell(core=core, input_layer=FFT(nfft=NFFT, dtype=DTYPE),
                      output_layer=iFFT(nfft=NFFT, dtype=DTYPE))
        z = torch.tensor(0.7 + 0.1j, dtype=CDTYPE)
        H_shell = shell.probe(z)
        H_expected = g2.probe(z) @ g1.probe(z)
        assert torch.allclose(H_shell, H_expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 8. test_shell_probe_include_io_identity_layers
# ---------------------------------------------------------------------------
class TestShellProbeIncludeIO:
    def test_fft_ifft_identity_layers(self):
        """FFT/iFFT layers return None from probe, so include_io should give same result."""
        core = Gain(size=(3, 2), nfft=NFFT, dtype=DTYPE)
        fft_in = FFT(nfft=NFFT, dtype=DTYPE)
        ifft_out = iFFT(nfft=NFFT, dtype=DTYPE)
        shell = Shell(core=core, input_layer=fft_in, output_layer=ifft_out)
        z = torch.tensor(0.8 + 0.2j, dtype=CDTYPE)

        H_default = shell.probe(z, include_shell_io=False)
        H_with_io = shell.probe(z, include_shell_io=True)
        assert torch.allclose(H_default, H_with_io, atol=1e-12)

    def test_gain_io_layers(self):
        """Using gain modules as I/O layers should compose them via probe."""
        core = Gain(size=(3, 3), nfft=NFFT, dtype=DTYPE)
        in_gain = Gain(size=(3, 2), nfft=NFFT, dtype=DTYPE)
        out_gain = Gain(size=(4, 3), nfft=NFFT, dtype=DTYPE)
        shell = Shell(core=core, input_layer=in_gain, output_layer=out_gain)
        z = torch.tensor(0.8 + 0.2j, dtype=CDTYPE)

        H_core_only = shell.probe(z, include_shell_io=False)
        H_with_io = shell.probe(z, include_shell_io=True)

        assert torch.allclose(H_core_only, core.probe(z), atol=1e-12)
        expected = out_gain.probe(z) @ core.probe(z) @ in_gain.probe(z)
        assert torch.allclose(H_with_io, expected, atol=1e-12)

    def test_identity_transform_layer(self):
        """nn.Identity() output layer has no probe method, should not break."""
        core = Gain(size=(3, 2), nfft=NFFT, dtype=DTYPE)
        shell = Shell(core=core, input_layer=FFT(nfft=NFFT, dtype=DTYPE))
        z = torch.tensor(0.8 + 0.2j, dtype=CDTYPE)
        H = shell.probe(z, include_shell_io=True)
        assert torch.allclose(H, core.probe(z), atol=1e-12)


# ---------------------------------------------------------------------------
# Extra: probe_points vectorized helper
# ---------------------------------------------------------------------------
class TestProbePoints:
    def test_probe_points_shape(self):
        g = Gain(size=(3, 2), nfft=NFFT, dtype=DTYPE)
        z_pts = torch.tensor([0.8+0.1j, 0.9+0.2j, 0.7-0.1j], dtype=CDTYPE)
        H_all = probe_points(g, z_pts)
        assert H_all.shape == (3, 3, 2)

    def test_probe_points_matches_individual(self):
        f = Filter(size=(4, 2, 3), nfft=NFFT, dtype=DTYPE)
        z_pts = torch.tensor([0.8+0.1j, 0.9+0.2j], dtype=CDTYPE)
        H_all = probe_points(f, z_pts)
        for i, z in enumerate(z_pts):
            assert torch.allclose(H_all[i], f.probe(z), atol=1e-12)


# ---------------------------------------------------------------------------
# Extra: incompatible subclasses raise NotImplementedError
# ---------------------------------------------------------------------------
class TestIncompatibleSubclassGuard:
    def test_biquad_probe_raises(self):
        b = Biquad(size=(1, 1), n_sections=1, nfft=NFFT, fs=48000, dtype=DTYPE)
        z = torch.tensor(0.9 + 0.1j, dtype=CDTYPE)
        with pytest.raises(NotImplementedError):
            b.probe(z)

    def test_svf_probe_raises(self):
        s = SVF(size=(1, 1), n_sections=1, nfft=NFFT, fs=48000, dtype=DTYPE)
        z = torch.tensor(0.9 + 0.1j, dtype=CDTYPE)
        with pytest.raises(NotImplementedError):
            s.probe(z)


# ---------------------------------------------------------------------------
# Extra: existing forward() is unchanged
# ---------------------------------------------------------------------------
class TestForwardUnchanged:
    def test_gain_forward_unchanged(self):
        g = Gain(size=(2, 3), nfft=NFFT, dtype=DTYPE)
        x = torch.randn(1, NFFT // 2 + 1, 3, dtype=CDTYPE)
        y = g(x)
        expected = torch.einsum("mn,bfn->bfm", to_complex(g.map(g.param)), x)
        assert torch.allclose(y, expected, atol=1e-12)

    def test_filter_forward_unchanged(self):
        f = Filter(size=(3, 2, 2), nfft=NFFT, dtype=DTYPE)
        x = torch.randn(1, NFFT // 2 + 1, 2, dtype=CDTYPE)
        y1 = f(x)
        y2 = torch.einsum("fmn,bfn->bfm", f.freq_response(f.param), x)
        assert torch.allclose(y1, y2, atol=1e-12)
