# Revision notebooks

Install the revision package before opening these notebooks. Do not add another checkout's `src` to `sys.path`. Print `raftup.__file__` to confirm which installation is active.

## Installation

Clone the `revision` branch as described in the [repository README](../../README.md), then run the following in a terminal from the repository root. Git and Conda/Miniforge must already be installed:

```sh
conda env create -f environment.yml
conda activate raftup_revision
python -m pip install -e '.[revision]'
python -m pip check
python -m ipykernel install --user --name raftup_revision --display-name 'Python (raftup_revision)'
```

The revision environment uses PyTorch 2.2.2; the previous 2.1.0 macOS wheel had incorrect architecture metadata. SCAN-IT is pinned to a commit. See VALIDATION.md for the fresh-environment execution report; Linux GPU execution is a separate check. GPU training requires a compatible Linux/CUDA installation; report `torch.__version__`, `torch.version.cuda`, and `torch.cuda.is_available()` before training. Saved feature-cost alignment and plotting do not require GPU training. Competing alignment methods keep their separate environments.

Before GPU feature training, check the active installation:

```sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

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

Download links and preparation commands are in [DATA.md](DATA.md). Each real-data tutorial takes prepared inputs, not bundled raw datasets. Use `scripts/prepare_revision_pair.py` to create the prepared inputs and manifest (see DATA.md). Alternatively, copy [pair.example.json](pair.example.json) to `pair.json` next to already prepared files and edit every field; copying the example alone does not create the data. Set `RAFTUP_TUTORIAL_DATA` to that directory. File paths inside the manifest resolve relative to the manifest itself. Set `RAFTUP_OUTPUT_ROOT` to a fresh output directory. Outputs do not modify the source data. Do not reuse output directories for changed configurations.

Required files:
- `slice_a.h5ad`, `slice_b.h5ad`: unique observation IDs, expression data and finite two-dimensional `obsm['spatial']`.
- `feature_cost.npy`: nonnegative finite A-by-B cost matrix from the intended feature pipeline.
- `cell_ids_a.json`, `cell_ids_b.json`: lists of IDs in the EXACT order used when constructing the cost matrix. The loader checks these against the slices. Never construct these lists from reordered slices to bypass a mismatch.
- `pair.json`: input paths, assay and coordinate calibration; see example.

For serial reconstruction, `series.json` is an ordered JSON list such as `["00_01/pair.json", "01_02/pair.json"]`. Shared sections must have identical IDs and original coordinates. No fitting to ground truth is performed by the alignment tutorial. Paper nRMSE uses a specific ground-truth reference frame and tissue diagonal; use the archived figure notebook for the exact paper evaluation.

Historical calibration: DLPFC 138 input coordinate units/100 µm; MERFISH 100. HT206B1 tutorials expect preconverted µm coordinates. Deep-STARmap's recorded 100 convention is historical numerical calibration, not independently verified physical calibration. Automatic tissue selection is not claimed for HT206B1, development or CCC. The `test` router is calibrated on nine research cases.

## Exact paper figures from archived results

`figures/01` through `figures/05` preserve the latest large-font, separate-cell plots. Set `RAFTUP_EXPERIMENT_ROOT` to the original revision data archive root (the directory containing `all_exp/`, `2_3d_reconstruction_exp/`, etc.). Launch Jupyter from the repository root and open a notebook in `examples/revision/figures/`. Use absolute environment-variable paths, set before starting Jupyter, to avoid dependence on the kernel working directory. The archive is NOT included in this source package. These notebooks verify/read saved results and do not rerun competing methods. The downsampling figure uses the archived 17 runs plus the supplemental 10% run; the package-call tutorial instead performs a fresh six-ratio sweep.

The exact input paths are visible in each figure's data cells. `source_manifest.json` records original notebook checksums. `protocol_sources/` preserves original experimental protocol code with outputs removed; its machine-specific paths are retained for provenance, and it is not the portable entry point.

## Output compatibility

The public API remains `import raftup`. Old private modules are retained. `run_automated_alignment` and `run_manual_alignment` return `(P, decision)`. The revision helper functions live in `raftup.experiments`. Raw alignment matrices, fitted coordinates and high-dimensional embeddings are different objects: use spatial coordinates for the purity metric.

## Execute in the active environment

```sh
python scripts/execute_revision_notebooks.py --mode tutorials --notebook 01 \
  --input-root /path/to/prepared_inputs --output-dir .validation/tutorials
python scripts/execute_revision_notebooks.py --mode figures --notebook 03 \
  --archive-root /path/to/revision_archive --output-dir .validation/figures
```

The runner creates a temporary kernel specification for the active Python, executes every code cell, and writes executed copies plus per-notebook pass/fail reports outside the source notebooks. Add `--notebook 01` to select one notebook. Use a new output directory for each run. Historical records in `protocol_sources/` are provenance copies, not supported portable tutorial entry points; their original machine paths are intentionally not used by this runner.

## Recommended first real-data run

1. Run notebook 00 or the smoke-check command in the repository README. No data or `pair.json` is needed for this check.
2. Download DLPFC and prepare `data/revision/dlpfc_0708/pair.json` using [DATA.md](DATA.md). Confirm that the feature-cost IDs exactly match the slice IDs.
3. Run only tutorial 01 first:

```sh
python scripts/execute_revision_notebooks.py --mode tutorials --notebook 01 \
  --input-root data/revision --output-dir outputs/dlpfc_run_01
```

The runner maps `--input-root` to the folders in DATA.md; for this example it reads `data/revision/dlpfc_0708/pair.json`. It writes `executed.ipynb` and `result.json` under `outputs/dlpfc_run_01/01_automated_DLPFC/`, and alignment files under that directory's `outputs/dlpfc/<name>/`. `<name>` is the manifest's `name` value. Successful execution produces `alignment_matrix.npz` and `decision.json`.

Run individual notebooks before a full batch. Omitting `--notebook` executes every notebook in the selected mode and stops on the first failure. Tutorial 07 requires an alignment to be placed in its input folder first; the runner does not copy tutorial 01 outputs into that folder automatically. The default timeout is 7,200 seconds **per code cell**; use `--timeout` to change it for a large run.

## Interactive execution and path overrides

From the repository root, replace the example absolute paths with your own:

```sh
conda activate raftup_revision
export RAFTUP_TUTORIAL_DATA="/absolute/path/to/prepared_inputs/dlpfc_0708"
export RAFTUP_OUTPUT_ROOT="/absolute/path/to/new_outputs/dlpfc_run_01"
jupyter lab
```

Open `examples/revision/01_automated_DLPFC.ipynb`, choose **Python (raftup_revision)** and run the cells from top to bottom. `RAFTUP_TUTORIAL_DATA` names one pair folder (or the folder containing `series.json` for tutorial 04), whereas the runner's `--input-root` names the parent of all case folders. A previously set variable overrides the notebook default; clear it with `unset RAFTUP_TUTORIAL_DATA RAFTUP_OUTPUT_ROOT` before relying on defaults. Changing terminal variables after Jupyter starts does not update existing kernels; relaunch Jupyter with the desired variables.

For a paper figure, use the same environment but set `RAFTUP_EXPERIMENT_ROOT` to the complete revision archive, and `RAFTUP_OUTPUT_ROOT` to a new output directory. Alternatively:

```sh
python scripts/execute_revision_notebooks.py --mode figures --notebook 03 \
  --archive-root /absolute/path/to/revision_archive \
  --output-dir outputs/ht206_figure_run_01
```

The example redraws HT206B1 from saved results. The public derived-archive URL is still pending; raw-data downloads cannot replace that archive. Figure exports can overwrite files with the same names, so always choose a fresh output root to preserve earlier figures.

## If a run stops

| Symptom | What to check |
|---|---|
| `ModuleNotFoundError` or an unexpected package version | Activate the environment, install from this checkout and select its registered kernel. Inspect `raftup.__file__`. |
| Missing `pair.json`, `.h5ad` or `.npy` | Prepare inputs first; check the printed input folder and any environment-variable overrides. |
| Cell-order mismatch | Use the IDs recorded when the cost was generated. Correct the input ordering/provenance rather than replacing IDs to bypass validation. |
| Missing archive table, mask or reconstructed coordinates | Obtain the derived revision archive. The original raw-data links do not supply these results. |
| `FileExistsError` | Choose a new output directory; saved alignment runs are not overwritten. |
| `KeyError` for a downsampling case | Tutorial 06 accepts only the three documented protocol names; check `pair.json` and the notebook's prerequisites. |
| Missing audit settings or `alignment_matrix.npz` | Complete tutorial 07's input checklist; the batch runner does not prepare these automatically. |

Notebook 05 prints purity metrics, notebook 06 displays coverage and notebook 07 prints its audit report. These cells do not write separate metric tables; retain the executed notebook if you need those outputs. See each notebook for the exact saved-file layout. Do not commit local datasets or executed notebooks to GitHub.
