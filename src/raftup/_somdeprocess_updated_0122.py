"""
Compute SOMDE spatially variable genes for individual ST slices.

This script:
1. Loads raw spatial transcriptomics data (.h5ad or Visium)
2. Applies normalize_total + log1p preprocessing
3. Runs SOMDE on spatial coordinates and gene expression
4. Saves ranked spatially variable genes to CSV

Author: RAFT-UP
"""

import os
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse
from somde import SomNode


# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_DIR = "./data4train"            # where *.h5ad are stored
OUTPUT_DIR = "./data4train/somde_updated_0122"    # output folder
SOM_K = 5                            # SOM granularity (as in original SOMDE)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Core function
# --------------------------------------------------
# for loading DLPFC12 data
def load_DLPFC(root_dir='/home/salovjade/0324_raftupdata/DLPFC12', section_id='151507'):
    # 151507, ..., 151676 12 in total
    ad = sc.read_visium(path=os.path.join(root_dir, section_id), count_file=section_id+'_filtered_feature_bc_matrix.h5')
    ad.var_names_make_unique()

    gt_dir = os.path.join(root_dir, section_id, 'gt')
    gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'), sep=',', header=None, index_col=0)
    ad.obs['original_clusters'] = gt_df.loc[:, 6]
    keep_bcs = ad.obs.dropna().index
    ad = ad[keep_bcs].copy()
    ad.obs['original_clusters'] = ad.obs['original_clusters'].astype(int).astype(str)
    print(f"{section_id} loading done")

    return ad


def compute_and_save_somde_csv(
    adata: sc.AnnData,
    section_id: str,
    output_dir: str = OUTPUT_DIR,
    som_k: int = SOM_K,
):
    """
    Compute SOMDE spatially variable genes for one slice.

    Parameters
    ----------
    adata : AnnData
        Raw spatial transcriptomics data (counts).
    section_id : str
        Slice identifier, used for naming output file.
    output_dir : str
        Directory to save SOMDE results.
    som_k : int
        Controls SOM resolution (same meaning as in SOMDE paper).

    Output
    ------
    Saves:
        somde_{section_id}.csv
    """

    print(f"[SOMDE] Processing slice {section_id}")

    # -----------------------------
    # 1. Expression preprocessing
    # -----------------------------
    # IMPORTANT: SOMDE does NOT normalize internally. 
    # sc.pp.normalize_total(adata) # could do, but for simplicity, we won't.
    # sc.pp.log1p(adata) # will have bug

    # -----------------------------
    # 2. Prepare inputs
    # -----------------------------
    # spatial coordinates (n_spots × 2)
    pts = adata.obsm["spatial"].astype(np.float32)

    # expression matrix (spots × genes)
    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X

    # convert to DataFrame: genes × spots (SOMDE expects this)
    df_expr = pd.DataFrame(X, columns=adata.var_names)

    # -----------------------------
    # 3. Run SOMDE
    # -----------------------------
    som = SomNode(pts, som_k)
    som.mtx(df_expr.T)     # aggregate expression on SOM nodes
    som.norm()             # stabilize + regress out library size
    result, _ = som.run()  # spatial DE test

    # -----------------------------
    # 4. Save result
    # -----------------------------
    output_path = os.path.join(output_dir, f"somde_{section_id}.csv")
    result.to_csv(output_path, index=False)

    print(f"[SOMDE] Saved result to {output_path}")


# --------------------------------------------------
# Example main (batch processing)
# --------------------------------------------------
if __name__ == "__main__":

    # Example: process multiple slices
    # section_ids = [
    #     "151507",
    #     "151508",
    #     "151509",
    #     "151510", 
    # ]

    section_ids = [
        "151669",
        "151670",
        "151671",
        "151672", 
        "151673",
        "151674",
        "151675",
        "151676", 
    ]


    for sid in section_ids:
        print(f"\n=== SOMDE for {sid} ===")

        # load pre-saved h5ad
        adata = load_DLPFC(section_id=sid)

        compute_and_save_somde_csv(
            adata=adata,
            section_id=sid
        )

        print(f'{sid} somde done')

    print("\nAll SOMDE computations finished.")