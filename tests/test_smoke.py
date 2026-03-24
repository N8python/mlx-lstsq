from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import unittest
import venv
from collections.abc import Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def built_wheel() -> Path | None:
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheels = sorted(DIST.glob(f"mlx_lstsq-*-{py_tag}-{py_tag}-*.whl"))
    if not wheels:
        wheels = sorted(DIST.glob("mlx_lstsq-*.whl"))
    if not wheels:
        return None
    return max(wheels, key=lambda path: path.stat().st_mtime)


def run_in_fresh_env(code: str, *, extra_packages: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
    wheel = built_wheel()
    if wheel is None:
        raise unittest.SkipTest("No built wheel found in dist/. Run `python -m build` first.")

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    with tempfile.TemporaryDirectory(prefix="mlx-lstsq-wheel-test-") as tmpdir:
        env_dir = Path(tmpdir) / "venv"
        venv.EnvBuilder(with_pip=True, symlinks=False).create(env_dir)
        python = env_dir / "bin" / "python"

        subprocess.run(
            [str(python), "-m", "pip", "install", str(wheel), *extra_packages],
            check=True,
            cwd=tmpdir,
            env=env,
        )
        return subprocess.run(
            [str(python), "-c", code],
            check=True,
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
        )


class SmokeTests(unittest.TestCase):
    def test_built_wheel_installs_and_solves_scalar_system(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise unittest.SkipTest("mlx-lstsq only supports macOS on Apple Silicon.")

        code = textwrap.dedent(
            """
            import mlx.core as mx
            import mlx_lstsq

            a = mx.array([[2.0]], dtype=mx.float32)
            b = mx.array([8.0], dtype=mx.float32)

            x = mlx_lstsq.solve(a, b)
            mx.eval(x)
            print(x.shape, float(x.item()))
            """
        )

        completed = run_in_fresh_env(code)

        self.assertEqual(completed.stdout.strip(), "(1,) 4.0")

    def test_built_wheel_matches_numpy_on_random_matrices(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise unittest.SkipTest("mlx-lstsq only supports macOS on Apple Silicon.")

        code = textwrap.dedent(
            """
            import numpy as np
            import mlx.core as mx
            import mlx_lstsq

            rng = np.random.default_rng(0)
            worst_rel_error = 0.0
            worst_case = None

            for case_idx in range(100):
                n = int(rng.integers(1, 33))
                m = int(rng.integers(n, n + 65))
                a = rng.normal(size=(m, n)).astype(np.float32)
                b = rng.normal(size=(m,)).astype(np.float32)

                expected = np.linalg.lstsq(a, b, rcond=None)[0]
                actual = np.asarray(mlx_lstsq.solve(mx.array(a), mx.array(b)).tolist(), dtype=np.float32)

                denominator = np.maximum(np.abs(expected), 1e-6)
                rel_error = np.abs(actual - expected) / denominator
                case_worst = float(rel_error.max(initial=0.0))

                if case_worst > worst_rel_error:
                    worst_rel_error = case_worst
                    worst_case = (case_idx, m, n)

            print(f"max_rel_error={worst_rel_error:.6g} worst_case={worst_case}")
            if worst_rel_error > 1e-3:
                raise AssertionError(
                    f"max relative error {worst_rel_error:.6g} exceeded 1e-3 on case {worst_case}"
                )
            """
        )

        completed = run_in_fresh_env(code, extra_packages=("numpy",))

        self.assertIn("max_rel_error=", completed.stdout)


if __name__ == "__main__":
    unittest.main()
