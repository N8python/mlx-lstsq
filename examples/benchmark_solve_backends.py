import argparse
import gc
import math
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import torch
from scipy import linalg as sla

from mlx_lstsq import cpu_triangular_solve, mps_cholesky_factor, solve


CPU_STREAM = mx.default_stream(mx.cpu)
TORCH_MPS = torch.device("mps")
TORCH_CPU = torch.device("cpu")


@dataclass
class BenchmarkResult:
    mean_s: float
    stderr_s: float
    samples_s: list[float]


def sync_mlx() -> None:
    mx.synchronize()
    mx.synchronize(CPU_STREAM)


def sync_torch_mps() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def solve_torch_hybrid(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    gram = A.T @ A
    rhs = A.T @ b
    L = torch.linalg.cholesky(gram)
    sync_torch_mps()
    L_cpu = L.to(TORCH_CPU)
    rhs_cpu = rhs.to(TORCH_CPU)
    y_cpu = torch.linalg.solve_triangular(
        L_cpu, rhs_cpu.unsqueeze(1), upper=False
    )
    return torch.linalg.solve_triangular(L_cpu.T, y_cpu, upper=True).squeeze(1)


def solve_scipy_cpu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    gram = A.T @ A
    rhs = A.T @ b
    L = sla.cholesky(gram, lower=True, check_finite=False, overwrite_a=False)
    y = sla.solve_triangular(
        L, rhs, lower=True, check_finite=False, overwrite_b=False
    )
    x = sla.solve_triangular(
        L.T, y, lower=False, check_finite=False, overwrite_b=False
    )
    return x.astype(np.float32, copy=False)


def solve_numpy_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    gram = A.T @ A
    rhs = A.T @ b
    L = np.linalg.cholesky(gram)
    y = np.linalg.solve(L, rhs)
    x = np.linalg.solve(L.T, y)
    return x.astype(np.float32, copy=False)


def benchmark(fn, warmup: int, trials: int) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    samples_s = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        samples_s.append(time.perf_counter() - t0)
    mean_s = sum(samples_s) / len(samples_s)
    stderr_s = 0.0
    if len(samples_s) > 1:
        var = sum((t - mean_s) ** 2 for t in samples_s) / (len(samples_s) - 1)
        stderr_s = math.sqrt(var) / math.sqrt(len(samples_s))
    return BenchmarkResult(mean_s=mean_s, stderr_s=stderr_s, samples_s=samples_s)


def format_result(result: BenchmarkResult) -> str:
    return f"{result.mean_s:8.3f} +/- {result.stderr_s:6.3f}"


def run_shape(m: int, n: int, warmup: int, trials: int, seed: int) -> dict[str, BenchmarkResult]:
    rng = np.random.default_rng(seed)
    A_np = rng.standard_normal((m, n), dtype=np.float32)
    b_np = rng.standard_normal((m,), dtype=np.float32)

    results: dict[str, BenchmarkResult] = {}

    results["numpy_chol"] = benchmark(
        lambda: solve_numpy_cholesky(A_np, b_np),
        warmup=warmup,
        trials=trials,
    )
    results["scipy_cpu"] = benchmark(
        lambda: solve_scipy_cpu(A_np, b_np),
        warmup=warmup,
        trials=trials,
    )

    A_mx = mx.array(A_np)
    b_mx = mx.array(b_np)
    mx.eval(A_mx, b_mx)
    sync_mlx()
    results["mlx_current"] = benchmark(
        lambda: _run_mlx(A_mx, b_mx),
        warmup=warmup,
        trials=trials,
    )
    del A_mx, b_mx
    sync_mlx()
    gc.collect()

    A_torch = torch.from_numpy(A_np).to(TORCH_MPS)
    b_torch = torch.from_numpy(b_np).to(TORCH_MPS)
    sync_torch_mps()
    results["torch_hybrid"] = benchmark(
        lambda: _run_torch(A_torch, b_torch),
        warmup=warmup,
        trials=trials,
    )
    del A_torch, b_torch
    sync_torch_mps()
    gc.collect()

    del A_np, b_np
    gc.collect()
    return results


def _run_mlx(A: mx.array, b: mx.array) -> None:
    x = solve(A, b, synchronize=False)
    mx.eval(x)
    sync_mlx()


def _run_torch(A: torch.Tensor, b: torch.Tensor) -> None:
    with torch.no_grad():
        _ = solve_torch_hybrid(A, b)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark mlx_lstsq against Torch hybrid, SciPy CPU, and NumPy "
            "Cholesky baselines on random least-squares systems."
        )
    )
    parser.add_argument("--n-exp", type=int, default=10, help="Use n = 2**n_exp.")
    parser.add_argument(
        "--m-start-exp",
        type=int,
        default=14,
        help="Start the m sweep at 2**m_start_exp.",
    )
    parser.add_argument(
        "--m-end-exp",
        type=int,
        default=22,
        help="End the m sweep at 2**m_end_exp.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("Torch MPS is required for the torch_hybrid benchmark.")

    mx.set_default_device(mx.gpu)

    n = 2**args.n_exp
    exponents = range(args.m_start_exp, args.m_end_exp + 1)

    print("benchmark: least-squares backends")
    print(f"n = {n}")
    print(f"m sweep = {[2**exp for exp in exponents]}")
    print(f"warmup = {args.warmup}, trials = {args.trials}")
    print()
    print(
        f"{'m':>10} {'mlx_current (s)':>20} {'torch_hybrid (s)':>20} "
        f"{'scipy_cpu (s)':>20} {'numpy_chol (s)':>20}"
    )
    print("-" * 100)

    for exp in exponents:
        m = 2**exp
        results = run_shape(
            m=m,
            n=n,
            warmup=args.warmup,
            trials=args.trials,
            seed=args.seed + exp,
        )
        print(
            f"{m:>10} "
            f"{format_result(results['mlx_current']):>20} "
            f"{format_result(results['torch_hybrid']):>20} "
            f"{format_result(results['scipy_cpu']):>20} "
            f"{format_result(results['numpy_chol']):>20}"
        )


if __name__ == "__main__":
    main()
