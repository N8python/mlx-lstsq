from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

__version__ = "0.1.1"

_RUNTIME_EXPORTS = {
    "cpu_triangular_solve",
    "mps_cholesky_factor",
    "mps_syrk_gram_rhs",
}


@lru_cache(maxsize=1)
def _load_runtime():
    import mlx.core as mx

    from . import _ext

    return mx, _ext


def __getattr__(name: str):
    if name in _RUNTIME_EXPORTS:
        _, ext = _load_runtime()
        return getattr(ext, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def solve(A: "mx.array", b: "mx.array", *, synchronize: bool = True) -> "mx.array":
    mx, ext = _load_runtime()
    gram, rhs = ext.mps_syrk_gram_rhs(A, b)
    return _solve_from_gram_rhs(gram, rhs, synchronize=synchronize, mx=mx, ext=ext)


def solve_ridge(
    A: "mx.array",
    b: "mx.array",
    ridge,
    *,
    synchronize: bool = True,
) -> "mx.array":
    mx, ext = _load_runtime()
    gram, rhs = ext.mps_syrk_gram_rhs(A, b)
    if hasattr(ridge, "shape"):
        gram = gram + ridge
    elif ridge != 0.0:
        gram = gram + ridge * mx.eye(gram.shape[0], dtype=gram.dtype)
    return _solve_from_gram_rhs(gram, rhs, synchronize=synchronize, mx=mx, ext=ext)


def _solve_from_gram_rhs(
    gram: "mx.array",
    rhs: "mx.array",
    *,
    synchronize: bool,
    mx=None,
    ext=None,
) -> "mx.array":
    if mx is None or ext is None:
        mx, ext = _load_runtime()
    factor = ext.mps_cholesky_factor(gram)
    y = ext.cpu_triangular_solve(factor, rhs, upper=True, transpose=True)
    x = ext.cpu_triangular_solve(factor, y, upper=True)
    if synchronize:
        mx.eval(x)
        mx.synchronize()
        mx.synchronize(mx.default_stream(mx.cpu))
    return x


__all__ = [
    "solve",
    "solve_ridge",
    "mps_syrk_gram_rhs",
    "mps_cholesky_factor",
    "cpu_triangular_solve",
]
