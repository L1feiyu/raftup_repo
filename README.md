# RAFT-UP

RAFT-UP is a Python package for robust alignment of spatial transcriptomics data.  
It provides tools for preprocessing, gene-cost construction, downsampling-based matching, full-resolution recovery, and visualization of alignment results.

## Basic usage

A typical workflow in RAFT-UP includes:

1. Data preprocessing and filtering out highly variable genes.
2. Construct the gene cost matrix.
3. Compute downsampled alignment.
4. Recover full-resolution alignment.
5. Visualize the matching results.

## Key parameters

Some important parameters used in RAFT-UP include:

- `rho_f1`: gene-expression cutoff used in the downsampled alignment stage.
- `rho_s`: spatial distance cutoff used in the downsampled alignment stage.
- `rho_f2`: gene-expression cutoff used in the full-resolution recovery stage.
- `rho_t`: spatial distance cutoff used in the full-resolution recovery stage.
- `k1`, `k2`: nearest-neighbor parameters used for construction of spatial cost in the full-resolution recovery stage.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/L1feiyu/raftup_repo.git
cd raftup_repo
conda activate raftup
pip install -e .
