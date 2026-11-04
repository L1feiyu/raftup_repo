# Validation of 0.2.0.dev0

Validated on 2026-11-04 in a newly created Conda environment from `environment.yml`, followed by `pip install -e ".[revision]"`: Python 3.10, macOS arm64, CPU, PyTorch 2.2.2. `pip check` passes. The previous Torch 2.1.0 macOS distribution had incorrect wheel architecture metadata; the revision environment now pins 2.2.2. Exact observed versions are recorded in [observed-verification-environment.json](../../environments/observed-verification-environment.json).

## Executed notebooks

All code cells of the following 13 supported notebooks ran in a real kernel using the new environment. Inputs and executed copies remain external to Git. Times are wall-clock seconds, with some independent checks run concurrently; they are not performance benchmarks.

| Notebook | Seconds | Input scope |
|---|---:|---|
| [00_installation_smoke_test.ipynb](00_installation_smoke_test.ipynb) | 15.61 | Four-cell synthetic pair; actual manual solver. |
| [01_automated_DLPFC.ipynb](01_automated_DLPFC.ipynb) | 437.68 | Full DLPFC 151507–151508 pair (4,221 × 4,381 spots), archived feature cost; automatic selection and recovery. |
| [02_automated_MERFISH.ipynb](02_automated_MERFISH.ipynb) | 308.73 | Full MERFISH -0.04/-0.09 pair (5,488 × 5,557 cells), archived feature cost; automatic selection and recovery. |
| [03_known_overlap.ipynb](03_known_overlap.ipynb) | 32.89 | One same-section DLPFC cropped-window pair (166 × 199 spots), experimental routing and exact correspondence evaluation; 87 shared original barcodes verified by array positions. |
| [04_3d_reconstruction.ipynb](04_3d_reconstruction.ipynb) | 847.68 | One real Deep-STARmap adjacent pair 13–14 (9,059 × 8,223 cells), automatic selection, rigid placement and output saving; not the complete 15-section rerun. |
| [05_HT206B1_nonbrain.ipynb](05_HT206B1_nonbrain.ipynb) | 196.53 | Full HT206B1 U2–U3 pair (3,949 × 4,311 spots), manual alignment, weighted rigid placement and spatial-MNN purity with 100 µm distance gate. |
| [06_downsampling.ipynb](06_downsampling.ipynb) | 1181.53 | Fresh DLPFC 151507–151508 sweep at 2%, 4%, 5%, 6.7%, 8%, 10%; all six full alignments with fixed costs/cutoffs. |
| [07_spatial_distortion.ipynb](07_spatial_distortion.ipynb) | 8.95 | Exact audit of the newly computed DLPFC final support, including pairs sharing endpoints; recorded 200 µm recovery cutoff. |
| [01_known_overlap.ipynb](figures/01_known_overlap.ipynb) | 57.58 | All figure cells using the archived partial-overlap results; no competitor retraining. |
| [02_3d_reconstruction.ipynb](figures/02_3d_reconstruction.ipynb) | 173.43 | All figure cells including nRMSE, mean bars and full saved 3D reconstruction; no new full-volume alignment. |
| [03_HT206B1_nonbrain.ipynb](figures/03_HT206B1_nonbrain.ipynb) | 50.75 | All figure cells using saved histology, purity and method coordinates; 27 internal checks passed. |
| [04_downsampling.ipynb](figures/04_downsampling.ipynb) | 58.48 | All figure cells using the historical 17 runs plus supplemental run; 15 plots generated. |
| [05_spatial_distortion.ipynb](figures/05_spatial_distortion.ipynb) | 86.69 | All figure cells using the archived audit; 51 source-table checks passed. |

The newly computed DLPFC, MERFISH, Deep-STARmap and HT206B1 alignment matrices match their saved historical counterparts exactly: identical positive support and maximum absolute difference of zero. The cropped-window test recovered all 87 known correspondences with precision and recall of 1. These are regression checks on the stated inputs, not independent validation of selector generalization.

## Additional checks

- Built a wheel, installed it into an isolated target directory and verified imports resolve to that wheel. Four functional tests and the actual manual-solver smoke notebook passed both with the editable package and with the installed wheel.
- The four tests cover rigid-transform recovery, distance-gated spatial MNN purity, exact support-pair auditing including shared endpoints, and rejection of inconsistent cost/observation ordering.
- SCAN-IT feature construction and two CPU training epochs passed on a small synthetic input; this is a dependency/integration check, not a rerun of 3,500-epoch feature training.
- The prepared-input CLI passed with an archived cost and explicit matching IDs. The public notebook runner passed its smoke-notebook execution check.
- All 13 supported notebooks compile. The five preserved automatic/manual/experimental algorithm files remain identical to the pre-packaging snapshot; see `algorithm_source_sha256.json`.

## Limits and reproduction

Linux/CUDA has not been executed here. Full benchmark collections and competing-method training were not rerun. The 3D package-call check uses one adjacent pair; the figure notebook reads the complete saved reconstruction. The partial-overlap routing check uses one window pair; the downsampling rerun uses DLPFC only. Historical notebooks in `protocol_sources/` preserve provenance and are not portable entry points; legacy tutorials are outside this revision validation.

[DATA.md](DATA.md) provides raw-data download links and prepared-input instructions. The exact five paper figures additionally require the authors' derived-result archive, whose public URL is not yet supplied. Data, feature costs, matrices and executed notebooks are not newly committed.

Run `python scripts/validate_revision.py` for the small checks, or use `scripts/execute_revision_notebooks.py` as described in [README.md](README.md) for real notebook execution. [validation_results.json](validation_results.json) records the scope, runtime and code hash of each successful execution.
