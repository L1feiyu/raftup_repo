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

### Main RAFT-UP environment

Clone the repository and install RAFT-UP in editable mode:

```bash
git clone https://github.com/L1feiyu/raftup_repo.git
cd raftup_repo
conda create -n raftup python=3.10
conda activate raftup
pip install -e .
```
To verify that RAFT-UP is imported from the local repository, run:
```bash
python -c "import raftup; print(raftup.__file__)"
```

### Optional Mayavi environment

For Mayavi-based 3D visualization of RAFT-UP alignment, we recommend using a separate environment to avoid dependency conflicts with the main RAFT-UP environment.

Create the Mayavi environment for RAFT-UP visualization with:

```bash
conda create -n mayavi_env -c conda-forge python=3.11 mayavi pyqt scanpy ipykernel openpyxl "numpy<2.4"
conda activate mayavi_env
```

