"""
Compute gene cost matrix using SOMDE-selected genes and spatial graph
representation learning (DGI + ScanIT).

Pipeline
--------
1. Assume sliceA and sliceB are AnnData objects with:
   - log-normalized expression
   - SOMDE-selected genes already subset
2. Construct spatial graphs using ScanIT (alpha-shape)
3. Learn spatial embeddings independently via Deep Graph Infomax (DGI)
4. Compute gene cost matrix as pairwise L2 distance between embeddings

Author: RAFT-UP
"""

from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse
import scanit
from torch_geometric.nn import GCNConv, DeepGraphInfomax


# ======================================================
# Utilities
# ======================================================
def set_global_seed(seed: int = 0):
    """Ensure full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sparse_mx_to_edge_index(sparse_mx):
    """Convert scipy sparse adjacency matrix to PyG edge_index."""
    sparse_mx = sparse_mx.tocoo()
    return torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )


def intersect(lst1, lst2):
    """Ordered intersection."""
    s = set(lst2)
    return [x for x in lst1 if x in s]


# ======================================================
# Core DGI training
# ======================================================
def iterSliceTrain(
    sliceA,
    sliceB,
    *,
    n_h: int = 64,
    n_epoch: int = 1000,
    lr: float = 1e-3,
    print_step: int = 500,
    seed: int = 0,
    device: str = "cpu",
):
    """
    Train independent DGI models on two spatial graphs
    and compute cross-slice L1 cost matrix.
    """

    set_global_seed(seed)

    # ------------------
    # Encoder
    # ------------------
    class Encoder(nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.prelu1 = nn.PReLU(hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.prelu2 = nn.PReLU(hidden_channels)

        def forward(self, x, edge_index):
            x = self.prelu1(self.conv1(x, edge_index))
            x = self.prelu2(self.conv2(x, edge_index))
            return x

    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    # ------------------
    # Data
    # ------------------
    X_A = sliceA.X.toarray() if scipy.sparse.issparse(sliceA.X) else sliceA.X
    X_B = sliceB.X.toarray() if scipy.sparse.issparse(sliceB.X) else sliceB.X

    edge_A = sparse_mx_to_edge_index(sliceA.obsp["scanit-graph"]).to(device)
    edge_B = sparse_mx_to_edge_index(sliceB.obsp["scanit-graph"]).to(device)

    X_A = torch.FloatTensor(X_A).to(device)
    X_B = torch.FloatTensor(X_B).to(device)

    # ------------------
    # DGI model
    # ------------------
    model = DeepGraphInfomax(
        hidden_channels=n_h,
        encoder=Encoder(X_A.shape[1], n_h),
        summary=lambda z, *args: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------
    # Training
    # ------------------
    for epoch in range(n_epoch):
        model.train()
        optimizer.zero_grad()

        zA_pos, zA_neg, sA = model(X_A, edge_A)
        zB_pos, zB_neg, sB = model(X_B, edge_B)

        loss = model.loss(zA_pos, zA_neg, sA) + model.loss(zB_pos, zB_neg, sB)
        loss.backward()
        optimizer.step()

        if epoch % print_step == 0 or epoch == n_epoch - 1:
            print(f"[{epoch:04d}] DGI loss = {loss.item():.4f}")

    # ------------------
    # Final embeddings
    # ------------------
    model.eval()
    with torch.no_grad():
        z_A = model(X_A, edge_A)[0]
        z_B = model(X_B, edge_B)[0]

    # ------------------
    # Cost matrix (L1)
    # ------------------
    n_a, n_b = z_A.shape[0], z_B.shape[0]
    C = torch.zeros((n_a, n_b))

    batch = 1000
    for i in range(0, n_a, batch):
        for j in range(0, n_b, batch):
            C[i:i+batch, j:j+batch] = torch.norm(
                z_A[i:i+batch, None, :] - z_B[None, j:j+batch, :],
                p=2,
                dim=2,
            ).cpu()

    return C


# ======================================================
# High-level wrapper
# ======================================================
def compute_gene_cost(
    sliceA,
    sliceB,
    *,
    section_id_A: str,
    section_id_B: str,
    n_h: int = 100,
    n_epoch: int = 3500,
    lr: float = 2e-4,
    print_step: int = 500,
    seed: int = 0,
    device: str = "cuda:0",
    output_dir: str = "./output",
):
    """
    Full gene cost computation pipeline for one slice pair.
    """

    # ------------------
    # Gene alignment
    # ------------------
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes].copy()
    sliceB = sliceB[:, common_genes].copy()

    # ------------------
    # Spatial graphs
    # ------------------
    scanit.tl.spatial_graph(
        sliceA,
        method="alpha shape",
        alpha_n_layer=3,
        knn_n_neighbors=15,
    )
    scanit.tl.spatial_graph(
        sliceB,
        method="alpha shape",
        alpha_n_layer=3,
        knn_n_neighbors=15,
    )

    # ------------------
    # Train & compute cost
    # ------------------
    C = iterSliceTrain(
        sliceA,
        sliceB,
        n_h=n_h,
        n_epoch=n_epoch,
        lr=lr,
        print_step=print_step,
        seed=seed,
        device=device,
    )

    # ------------------
    # Save
    # ------------------
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = f"{section_id_A}_{section_id_B}"
    torch.save(C, outdir / f"{prefix}_somde_cost_matrix.pt")

    return C