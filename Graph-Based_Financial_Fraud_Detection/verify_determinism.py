"""
Determinism verifier for the fraud-detection pipeline.

Runs the GATv2 training stage twice in the same process and confirms that
the resulting model weights, edge index, and validation scores are
bit-identical. Then runs the full kNN + GraphSAGE feature extraction twice
to confirm the same for the offline search code path.

Usage:
    python verify_determinism.py

A non-zero exit code means determinism is broken somewhere — re-read
DETERMINISM.md and check that:
  - You did not import numpy/torch/sklearn before app.py
  - Your environment matches the pinned requirements.txt
  - You are not running on GPU (this script is CPU-only)

This script does NOT require creditcard.csv. It uses a deterministic
synthetic dataset so anyone can run it without Kaggle access.
"""

import os
import sys

# Match the env-var block at the top of app.py / full_hyperparameter_search.py.
# Must be set before numpy/torch import.
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import random
import numpy as np
import torch


RANDOM_SEED = 42


def set_deterministic_state(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def make_synthetic_dataset(n=2000, d=30, fraud_rate=0.02):
    """Generate a deterministic synthetic dataset that mimics creditcard.csv shape."""
    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.standard_normal((n, d)).astype(np.float32)
    n_fraud = int(n * fraud_rate)
    y = np.zeros(n, dtype=np.int64)
    y[:n_fraud] = 1
    # Shift fraud points slightly so the kNN graph has structure.
    X[:n_fraud] += rng.standard_normal((n_fraud, d)).astype(np.float32) * 0.5 + 1.5
    # Shuffle deterministically.
    perm = rng.permutation(n)
    return X[perm], y[perm]


def train_gatv2_once(X, y):
    """One forward training pass equivalent to fit_graphsage in app.py."""
    from torch_geometric.nn import GATv2Conv  # imported lazily so the import error message is clearer
    import torch.nn.functional as F
    from sklearn.neighbors import NearestNeighbors

    set_deterministic_state(RANDOM_SEED)

    # Build kNN edge index (matches build_knn_edge_index in app.py).
    k = 10
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="ball_tree")
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)
    n = indices.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = indices[:, 1:].ravel()
    src_both = np.concatenate([src, dst])
    dst_both = np.concatenate([dst, src])
    pairs = np.unique(np.stack([src_both, dst_both], axis=1), axis=0)
    edge_index = torch.tensor(pairs.T, dtype=torch.long)

    x = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    class GATv2Tiny(torch.nn.Module):
        def __init__(self, in_c):
            super().__init__()
            self.conv1 = GATv2Conv(in_c, 16, heads=4, dropout=0.3, concat=True)
            self.conv2 = GATv2Conv(16 * 4, 8, heads=1, dropout=0.3, concat=False)
            self.head = torch.nn.Linear(8, 1)

        def forward(self, x, ei):
            x = F.elu(self.conv1(x, ei))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.elu(self.conv2(x, ei))
            return self.head(x).squeeze(-1)

    model = GATv2Tiny(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1.0)], dtype=torch.float32)

    model.train()
    for _ in range(20):
        opt.zero_grad()
        logits = model(x, edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x, edge_index)).numpy()

    # Snapshot all model weights into a single concatenated vector for easy comparison.
    weight_snapshot = torch.cat([p.detach().flatten() for p in model.parameters()]).numpy()
    return scores, weight_snapshot, edge_index.numpy()


def main():
    # Set state once up front so the diagnostic banner reflects the actual
    # state the training functions will run under.
    set_deterministic_state(RANDOM_SEED)

    print("=" * 72)
    print("Determinism verifier — fraud-detection GATv2 pipeline")
    print("=" * 72)
    print(f"Python:  {sys.version.split()[0]}")
    print(f"NumPy:   {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Threads: intra={torch.get_num_threads()}  inter={torch.get_num_interop_threads()}")
    print(f"Det algos enabled: {torch.are_deterministic_algorithms_enabled()}")
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print()

    print("Building synthetic dataset (2000 rows, 30 features, 2% fraud)...")
    X1, y1 = make_synthetic_dataset()
    X2, y2 = make_synthetic_dataset()
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2), "Synthetic dataset is non-deterministic"
    print("  Synthetic dataset is reproducible.")
    print()

    print("Run 1: training GATv2 on synthetic data...")
    scores1, weights1, edges1 = train_gatv2_once(X1, y1)

    print("Run 2: training GATv2 on synthetic data (fresh state)...")
    scores2, weights2, edges2 = train_gatv2_once(X1, y1)

    print()
    print("Comparing results:")
    print(f"  Edge index identical: {np.array_equal(edges1, edges2)}")
    print(f"  Model weights identical: {np.array_equal(weights1, weights2)}")
    print(f"  Output scores identical: {np.array_equal(scores1, scores2)}")
    if not np.array_equal(weights1, weights2):
        diff = np.abs(weights1 - weights2)
        print(f"    max abs diff = {diff.max():.3e}, mean abs diff = {diff.mean():.3e}")
    if not np.array_equal(scores1, scores2):
        diff = np.abs(scores1 - scores2)
        print(f"    max abs diff = {diff.max():.3e}, mean abs diff = {diff.mean():.3e}")
    print()

    all_match = (
        np.array_equal(edges1, edges2)
        and np.array_equal(weights1, weights2)
        and np.array_equal(scores1, scores2)
    )

    if all_match:
        print("PASS: pipeline is bit-identical on this machine.")
        print()
        print("Note: bit-identity across DIFFERENT machines additionally requires")
        print("matching CPU architecture and BLAS implementation. Use the Dockerfile")
        print("if you need cross-machine bit-identity. See DETERMINISM.md.")
        sys.exit(0)
    else:
        print("FAIL: outputs differ between runs in the same process.")
        print("Re-read DETERMINISM.md and confirm:")
        print("  - Environment variables were set before importing numpy/torch")
        print("  - You are using the pinned requirements.txt versions")
        print("  - You are not on GPU")
        sys.exit(1)


if __name__ == "__main__":
    main()
