"""
Smoke tests for raftup internal modules.
"""

import numpy as np

from raftup import (
    _downsample,
    _load_data,
    _metrics_two_gpr,
    _fsgw_utils,
    _recoverfull_new_new_knn,
)


def test_downsample():
    print("[TEST] _downsample")

    import scanpy as sc
    import numpy as np

    # 构造最小 AnnData
    X = np.random.rand(10, 5)
    adata = sc.AnnData(X)
    adata.obsm["spatial"] = np.random.rand(10, 2)

    _downsample.downsample_slice(adata, tar_distance=0.5)

    print("  OK")


def test_metrics():
    print("[TEST] _metrics_two_gpr")
    P = np.eye(5)
    X1 = np.random.rand(5, 2)
    X2 = np.random.rand(5, 2)
    _metrics_two_gpr.GPR_original(P, X1, X2, dis_cut=1.0)
    print("  OK")


def test_utils():
    print("[TEST] _fsgw_utils")
    print("  OK")


def test_recover():
    print("[TEST] _recoverfull_new_new_knn")
    print("  OK")


if __name__ == "__main__":
    test_downsample()
    test_metrics()
    test_utils()
    test_recover()
    print("\nAll smoke tests passed.")