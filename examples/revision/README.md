# Revision notebooks

Install the revision package before opening these notebooks. Do not add another checkout's `src` to `sys.path`. Print `raftup.__file__` to confirm which installation is active.

## Installation

From the repository root:

```sh
conda env create -f environment.yml
conda activate raftup_revision
python -m pip install -e '.[revision]'
python -m ipykernel install --user --name raftup_revision --display-name 'Python (raftup_revision)'
```

The environment definition is based on the existing verification environment; a new cross-platform Conda solve and Linux GPU run have not yet been validated. SCAN-IT is pinned to a commit. PyTorch 2.1.0 is retained for reproducibility. GPU training requires a compatible Linux/CUDA installation; report `torch.__version__`, `torch.version.cuda`, and `torch.cuda.is_available()` before training. Saved feature-cost alignment and plotting do not require GPU training. Competing alignment methods keep their separate environments.

## Package-call tutorials

| Notebook | Task |
|---|---|
| 00_installation_smoke_test | Tiny synthetic solver check; no external data |
| 01_automated_DLPFC | DLPFC automatic selection |
| 02_automated_MERFISH | MERFISH automatic selection |
| 03_known_overlap | Experimental partial-overlap routing and exact correspondence evaluation |
| 04_3d_reconstruction | Consecutive Deep-STARmap alignments and rigid transform composition |
| 05_HT206B1_nonbrain | Manual HT206B1 settings and spatial-MNN purity evaluation |
| 06_downsampling | Explicit fixed-cutoff sampling sweep |
| 07_spatial_distortion | Exact final-support spatial audit |

Each real-data tutorial takes prepared inputs, not bundled raw datasets. Copy `pair.example.json` to `pair.json` next to your prepared data and edit it. Set `RAFTUP_TUTORIAL_DATA` to that directory. File paths inside the manifest resolve relative to the manifest itself. Set `RAFTUP_OUTPUT_ROOT` to a fresh output directory. Outputs do not modify the source data. Do not reuse output directories for changed configurations.

Required files:
- `slice_a.h5ad`, `slice_b.h5ad`: unique observation IDs, expression data and finite two-dimensional `obsm['spatial']`.
- `feature_cost.npy`: nonnegative finite A-by-B cost matrix from the intended feature pipeline.
- `cell_ids_a.json`, `cell_ids_b.json`: lists of IDs in the EXACT order used when constructing the cost matrix. The loader checks these against the slices. Never construct these lists from reordered slices to bypass a mismatch.
- `pair.json`: input paths, assay and coordinate calibration; see example.

For serial reconstruction, `series.json` is an ordered JSON list such as `["00_01/pair.json", "01_02/pair.json"]`. Shared sections must have identical IDs and original coordinates. No fitting to ground truth is performed by the alignment tutorial. Paper nRMSE uses a specific ground-truth reference frame and tissue diagonal; use the archived figure notebook for the exact paper evaluation.

Historical calibration: DLPFC 138 input coordinate units/100 µm; MERFISH 100. HT206B1 tutorials expect preconverted µm coordinates. Deep-STARmap's recorded 100 convention is historical numerical calibration, not independently verified physical calibration. Automatic tissue selection is not claimed for HT206B1, development or CCC. The `test` router is calibrated on nine research cases.

## Exact paper figures from archived results

`figures/01` through `figures/05` preserve the latest large-font, separate-cell plots. Set `RAFTUP_EXPERIMENT_ROOT` to the original revision data archive root (the directory containing `all_exp/`, `2_3d_reconstruction_exp/`, etc.). Launch Jupyter in `figures/`. The archive is NOT included in this source package. These notebooks verify/read saved results and do not rerun competing methods. The downsampling figure uses the archived 17 runs plus the supplemental 10% run; the package-call tutorial instead performs a fresh six-ratio sweep.

The exact input paths are visible in each figure's data cells. `source_manifest.json` records original notebook checksums. `protocol_sources/` preserves original experimental protocol code with outputs removed; its machine-specific paths are retained for provenance, and it is not the portable entry point.

## Output compatibility

The public API remains `import raftup`. Old private modules are retained. `run_automated_alignment` and `run_manual_alignment` return `(P, decision)`. The revision helper functions live in `raftup.experiments`. Raw alignment matrices, fitted coordinates and high-dimensional embeddings are different objects: use spatial coordinates for the purity metric.
