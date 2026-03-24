import argparse
import struct
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lstsq import solve_ridge


def read_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected image magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected label magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.shape[0] != count:
        raise ValueError(f"label count mismatch in {path}")
    return data


def load_mnist(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images = read_idx_images(root / "train-images-idx3-ubyte")
    train_labels = read_idx_labels(root / "train-labels-idx1-ubyte")
    test_images = read_idx_images(root / "t10k-images-idx3-ubyte")
    test_labels = read_idx_labels(root / "t10k-labels-idx1-ubyte")
    return train_images, train_labels, test_images, test_labels


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_data = repo_root / "data" / "MNIST" / "raw"

    parser = argparse.ArgumentParser(
        description="Train one-vs-rest MNIST logistic classifiers with IRLS using mlx_lstsq."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"MNIST raw directory not found: {args.data_dir}")

    mx.set_default_device(mx.gpu)
    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    X_train = np.concatenate(
        [X_train, np.ones((X_train.shape[0], 1), dtype=np.float32)], axis=1
    )
    X_test = np.concatenate(
        [X_test, np.ones((X_test.shape[0], 1), dtype=np.float32)], axis=1
    )

    X_train_mx = mx.array(X_train)
    X_test_mx = mx.array(X_test)
    ridge_eye = args.ridge * mx.eye(X_train.shape[1], dtype=mx.float32)
    weights = []
    start = time.perf_counter()
    for cls in range(10):
        beta = mx.zeros((X_train.shape[1],), dtype=mx.float32)
        target = mx.array((y_train == cls).astype(np.float32))
        for _ in range(args.iterations):
            logits = X_train_mx @ beta
            probs = 1.0 / (1.0 + mx.exp(-logits))
            weights_diag = mx.maximum(probs * (1.0 - probs), 1e-6)
            working_response = logits + (target - probs) / weights_diag
            sqrt_weights = mx.sqrt(weights_diag)
            A = X_train_mx * sqrt_weights[:, None]
            b = working_response * sqrt_weights
            beta = solve_ridge(A, b, ridge_eye)
        weights.append(beta)

    W = mx.stack(weights, axis=1)
    logits = X_test_mx @ W
    predictions = np.array(mx.argmax(logits, axis=1))
    elapsed = time.perf_counter() - start
    accuracy = float((predictions == y_test).mean())

    print(f"data_dir: {args.data_dir}")
    print(f"iterations: {args.iterations}")
    print(f"ridge: {args.ridge}")
    print(f"train_seconds: {elapsed:.6f}")
    print(f"val_accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
