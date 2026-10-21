"""The fused compiled readout: sum_p P(obs - x_p) s_p without storing a block per point.

``apply_9x9_fortran`` must equal the Python route it replaces (``kernel_9x9_python`` at every point, then
the contraction with the sources) to round-off on both branches of the ladder; its displacement-only form
(rows = 3, which skips the second derivatives) must equal the first three rows of the full one; a source
at an observer must raise as the Python route does; the result must not depend on the thread count; and
``graded_field`` through it must give the field it gave before.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel.kernel import kernel_9x9_python
from cubic_scattering.graded_voxel.kernel_fortran import apply_9x9_fortran

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
OMEGA = 900.0  # k_S = 0.3 /m: separations of 0.2 to 40 m straddle the series switch at k_S r = 0.5


def _case(seed: int = 1, m: int = 5, n: int = 700):
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-20.0, 20.0, size=(m, 3))
    points = rng.uniform(-3.0, 3.0, size=(n, 3))
    src = rng.normal(size=(n, 9)) + 1j * rng.normal(size=(n, 9))
    return obs, points, src


def _reference(obs, points, src, omega):
    return np.stack([np.einsum("nab,nb->a", kernel_9x9_python(x - points, omega, REF), src) for x in obs])


@pytest.mark.parametrize("omega", [OMEGA, OMEGA * (1 + 0.05j)])
def test_fused_readout_equals_the_python_route(omega) -> None:
    obs, points, src = _case()
    got = apply_9x9_fortran(obs, points, src, omega, REF, rows=9)
    want = _reference(obs, points, src, omega)
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-13


def test_displacement_only_is_the_first_three_rows() -> None:
    obs, points, src = _case(2)
    full = apply_9x9_fortran(obs, points, src, OMEGA, REF, rows=9)
    disp = apply_9x9_fortran(obs, points, src, OMEGA, REF, rows=3)
    assert disp.shape == (len(obs), 3)
    assert np.abs(disp - full[:, :3]).max() <= 1e-15 * np.abs(full[:, :3]).max()


def test_a_source_at_an_observer_raises() -> None:
    obs, points, src = _case(3)
    points[17] = obs[2]
    with pytest.raises(ValueError, match="r = 0"):
        apply_9x9_fortran(obs, points, src, OMEGA, REF, rows=3)


def test_result_is_independent_of_thread_count() -> None:
    code = (
        "import numpy as np, sys; sys.path.insert(0, '.');"
        "from cubic_scattering.tests.test_apply_kernel_fortran import _case, OMEGA, REF;"
        "from cubic_scattering.graded_voxel.kernel_fortran import apply_9x9_fortran;"
        "o, p, s = _case(4, m=9, n=2000);"
        "sys.stdout.buffer.write(apply_9x9_fortran(o, p, s, OMEGA, REF, rows=9).tobytes())"
    )
    outs = []
    for threads in ("1", "4"):
        env = dict(os.environ, OMP_NUM_THREADS=threads)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, check=True)
        outs.append(done.stdout)
    assert outs[0] == outs[1]


def test_graded_field_through_the_fused_readout_is_unchanged(monkeypatch) -> None:
    """The scattered field of a small solved graded sphere, by the fused readout and by the former loop."""
    from cubic_scattering.graded_voxel import farfield as FF
    from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft
    from cubic_scattering.tests.test_graded_voxel_fft import CON, KHAT, _profile

    # k_S h = 1/2: within the stored near-series moments
    res = solve_graded_sphere_fft(600.0, 10.0, REF, CON, 4, _profile, KHAT, KHAT, "P", p=1, r=1)
    obs = np.array([[-12.5, x, 0.0] for x in np.linspace(0.0, 30.0, 4)])
    u_fused, e_fused = FF.graded_field(res, obs, n_gauss=4)
    u_disp, e_none = FF.graded_field(res, obs, n_gauss=4, strain=False)
    monkeypatch.setattr(FF, "point_kernel_backend", lambda: "python")
    u_loop, e_loop = FF.graded_field(res, obs, n_gauss=4)
    assert np.abs(u_fused - u_loop).max() / np.abs(u_loop).max() < 1e-13
    assert np.abs(e_fused - e_loop).max() / np.abs(e_loop).max() < 1e-13
    assert e_none is None and np.abs(u_disp - u_loop).max() / np.abs(u_loop).max() < 1e-13
