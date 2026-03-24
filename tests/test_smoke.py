from __future__ import annotations

import os
import platform
import subprocess
import tempfile
import textwrap
import unittest
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def built_wheel() -> Path | None:
    wheels = sorted(DIST.glob("mlx_lstsq-*.whl"))
    if not wheels:
        return None
    return max(wheels, key=lambda path: path.stat().st_mtime)


class SmokeTests(unittest.TestCase):
    def test_built_wheel_installs_and_solves_scalar_system(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise unittest.SkipTest("mlx-lstsq only supports macOS on Apple Silicon.")

        wheel = built_wheel()
        if wheel is None:
            raise unittest.SkipTest("No built wheel found in dist/. Run `python -m build` first.")

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

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

        with tempfile.TemporaryDirectory(prefix="mlx-lstsq-wheel-smoke-") as tmpdir:
            env_dir = Path(tmpdir) / "venv"
            venv.EnvBuilder(with_pip=True, symlinks=False).create(env_dir)
            python = env_dir / "bin" / "python"

            subprocess.run(
                [str(python), "-m", "pip", "install", str(wheel)],
                check=True,
                cwd=tmpdir,
                env=env,
            )
            completed = subprocess.run(
                [str(python), "-c", code],
                check=True,
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.stdout.strip(), "(1,) 4.0")


if __name__ == "__main__":
    unittest.main()
