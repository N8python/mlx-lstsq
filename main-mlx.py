import time

import mlx.core as mx

from mlx_lstsq import solve

mx.set_default_device(mx.gpu)

CPU_STREAM = mx.default_stream(mx.cpu)

ms = [5000]
ns = [1000]
warmup = 1
trials = 5


def force(x):
    mx.eval(x)
    mx.synchronize(CPU_STREAM)


print("solver: specialized_gpu_gram_factor_cpu_triangular")
print(f"{'m':>8} {'n':>8} {'ms(sync)':>10}")
print("-" * 30)

for m in ms:
    for n in ns:
        if n > m:
            continue
        A = mx.random.normal((m, n))
        b = mx.random.normal((m,))
        s = None
        for _ in range(warmup):
            s = solve(A, b, synchronize=False)
            force(s)

        t0 = time.perf_counter()
        for _ in range(trials):
            force(solve(A, b, synchronize=False))
        elapsed = (time.perf_counter() - t0) / trials * 1000

        print(f"{m:>8} {n:>8} {elapsed:>10.2f}")
