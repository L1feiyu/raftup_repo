# Data downloads and directory layout

**New revision datasets, cached feature costs, alignment matrices and generated results are not bundled with these tutorials.** Older tutorial assets may already be tracked in the repository; they do not constitute the complete revision inputs. Download them separately; keep them in the ignored `data/` directory or another local disk. All notebooks support environment-variable path overrides. Do not commit executed notebooks containing data or large output images.

## Original source data

| Data | Download source | Use in this repository |
|---|---|---|
| DLPFC | [DLPFC12 benchmark archive](https://zenodo.org/records/10698880), [original spatialLIBD resource](https://research.libd.org/spatialLIBD/) | 151507–151508 tutorial; additional pairs and partial-overlap inputs derive from DLPFC sections |
| MERFISH | [benchmark archive](https://zenodo.org/records/10698909), [original Dryad record](https://doi.org/10.5061/dryad.8t8s248) | -0.04 and -0.09 slice tutorial |
| Deep-STARmap | [publication data record](https://doi.org/10.5281/zenodo.16783354) | Computational sections derived from the measured 3D volume; the paper uses 15 sections and a saved rotation manifest |
| HT206B1 | [Mo et al. source repository and Data Access instructions](https://github.com/ding-lab/ST_subclone_publication), [sample lookup table](https://github.com/ding-lab/ST_subclone_publication/blob/main/Data_access/Sample_ID_Lookup_table_v1.xlsx), [HTAN portal](https://humantumoratlas.org/) | Select HT206B1 U2–U5, including expression matrices, spatial positions, scale factors, histology and tumour-purity annotations |

The Zenodo API records were checked during environment validation: DLPFC uses `DLPFC12.zip`; MERFISH uses `mHypothalamus.zip`; Deep-STARmap includes `Brain_Deep_STARmap_expression_matrix.csv` and `Brain_Deep_STARmap_spatial.csv` (use gene-expression features and measured spatial coordinates, not UMAP coordinates).

HTAN/Synapse access may require an account and acceptance of the provider's terms. The software does not include credentials or bypass access controls. Use the sample lookup and the portal file identifiers; an expression matrix alone does not include histology or tumour-purity estimates.

## Raw data are not the complete paper reproduction archive

The five `figures/` notebooks additionally require derived files: fixed feature costs, model outputs, reconstructed coordinates, the rotation ground truth, purity metadata, H&E masks and evaluation summaries. These are not contained in the original raw-data links above. A public download URL for the RAFT-UP revision-result archive has not yet been supplied in this repository. Until it is supplied, the exact paper figures require the authors' local archive. Do not substitute new computed results while claiming exact reproduction of the saved figures.

Set `RAFTUP_EXPERIMENT_ROOT` to the directory containing the archive's original `all_exp/`, `1_simulated_data_exp/`, `2_3d_reconstruction_exp/`, and other referenced subdirectories. Each figure notebook lists its exact input paths and verifies saved metrics where applicable. Set `RAFTUP_OUTPUT_ROOT` to a separate writable output folder. No raw datasets are downloaded automatically.

## Prepared package-call inputs

Suggested layout, independent of the current working directory:

```text
raftup_repo/
  data/revision/                 # ignored by Git
    dlpfc_0708/pair.json
    merfish_0409/pair.json
    known_overlap/pair.json
    deep_starmap_series/series.json
    ht206b1_u2_u3/pair.json
  outputs/revision/              # ignored by Git
```

Each `pair.json` points to two `.h5ad` slices, an A-by-B `.npy` feature-cost matrix and two JSON lists recording the cost's exact observation order. Paths inside a manifest resolve relative to the manifest, not the notebook. Every slice must have unique `obs_names`, counts/expression in `X`, and finite two-column `obsm['spatial']`. The loader rejects ordering mismatches. Coordinates must retain the declared calibration.

For DLPFC, the existing loader accepts the downloaded archive directory explicitly:

```python
from raftup import _load_data
a = _load_data.load_DLPFC(root_dir="/your/download/DLPFC12", section_id="151507")
b = _load_data.load_DLPFC(root_dir="/your/download/DLPFC12", section_id="151508")
a.write_h5ad("A.h5ad")
b.write_h5ad("B.h5ad")
```

For other providers, prepare AnnData using their expression matrices and cell/spot coordinates; the legacy assay tutorials document preprocessing. For HT206B1, match barcodes to tissue-position rows, retain the intended in-tissue spots and convert full-resolution image coordinates to µm using the recorded calibration. Add `Tumor_purity_estimate` in the same observation order for evaluation; never infer purity from expression just to fill this field. Deep-STARmap sections and simulated overlap are derived inputs, not direct raw-download filenames; preserve their preprocessing/rotation manifests from the revision archive for exact paper reproduction.

### Create an input manifest and a new feature cost

From the repository root, after activating the revision environment:

```sh
python scripts/prepare_revision_pair.py --a A.h5ad --b B.h5ad \
  --output data/revision/dlpfc_0708 --name dlpfc_0708 \
  --technology dlpfc --pixels-per-100um 138 --device cpu
```

This computes a new SCAN-IT cost (3,500 epochs by default) and saves the IDs used to construct it. On a tested GPU environment, `--device cuda` selects GPU feature training. The resulting cost is not guaranteed to match an archived paper matrix. The alignment tutorials themselves use the saved cost.

To reuse an existing matrix, add `--cost feature_cost.npy --ids-a cell_ids_a.json --ids-b cell_ids_b.json`. These ID lists must come from the matrix-generation step, not be fabricated from subsequently reordered inputs. No matrix is silently reordered.

For MERFISH use `--technology merfish --pixels-per-100um 100`. HT206B1 expects prepared coordinates already in µm, so use `--technology dlpfc --pixels-per-100um 100`. The historical Deep-STARmap numerical convention also uses 100; its physical calibration has not been independently validated. Partial-overlap tutorials require an explicit actual assay in the manifest and select the experimental `test` workflow only when called.

For 3D reconstruction create an ordered JSON list of relative pair manifests, e.g. `["00_01/pair.json", "01_02/pair.json"]`, in `series.json`. Consecutive pairs must reuse exactly the same shared-section IDs and coordinates.

For spatial auditing, additionally place the saved `alignment_matrix.npz` in the pair folder and set `audit_coordinate_scale` (two multipliers) and `audit_spatial_cutoff` in `pair.json` using the original run's recorded units. The example manifest values are illustrative, not universally correct thresholds.

## Existing-result reproducibility

The paper's downsampling figure combines 17 historical runs and one supplemental 10% run. A fresh six-ratio sweep in notebook 06 is a new computation and is labeled accordingly. A short adjacent-pair execution check of notebook 04 is not a rerun of the full 15-section experiment. Validation reports distinguish these scopes explicitly.

## Manifest fields and experiment-specific prerequisites

The preparation script creates `pair.json`, the two `.h5ad` files, the feature-cost `.npy` and ID JSON files in the requested output folder. Use an empty folder. It does not create ground truth, purity annotations, serial sections or audit settings. Keep a unique `name` for each pair, especially in a reconstruction series.

| Field | Meaning |
|---|---|
| `name` | Case name used in output paths; tutorial 06 requires one of the three names below |
| `slice_a`, `slice_b` | Paths to prepared `.h5ad` inputs |
| `feature_cost` | A-by-B `.npy` matrix in the recorded observation order |
| `cell_ids_a`, `cell_ids_b` | Paths to JSON lists of observation IDs in cost-matrix order |
| `data_technology` | `dlpfc` or `merfish`; the partial-overlap router still needs the actual assay |
| `pixels_per_100um` | Number of input coordinate units per 100 µm; 100 for coordinates already in µm |

The same pair manifest can be reused for different notebooks, but each run should have its own output directory. No download or preparation happens automatically when a real-data notebook is opened.

- **03, known overlap:** optionally add `"ground_truth": "ground_truth.npy"` to `pair.json`. This must be an A-by-B boolean/0–1 array in the same order as the cost, derived from known correspondence (for example retained IDs in a simulated crop). Without this field the final precision/recall cell is skipped. The tutorial does not generate the five overlap conditions.
- **04, reconstruction:** place `series.json` in `deep_starmap_series/`, containing an ordered list such as `["00_01/pair.json", "01_02/pair.json"]`. Pair paths resolve relative to `series.json`; each pair's input paths resolve relative to its own `pair.json`. Shared sections must have identical IDs, order and original XY coordinates. Use distinct pair names. The existing notebook sets `Z_SPACING = 10`; verify that this matches your intended z units and spacing. The notebook does not create sections or random rotations from the raw 3D volume.
- **05, HT206B1:** both slices must contain finite `obs['Tumor_purity_estimate']` values aligned by barcode, with XY coordinates in µm and `pixels_per_100um=100`. The notebook processes one pair per run and prints Spearman correlation and MAE for spatial mutual nearest neighbours within 100 µm. Repeat manually for the three adjacent pairs if needed; it does not aggregate all pairs automatically.
- **06, downsampling:** `name` must be exactly `dlpfc_0708`, `merfish_0409` or `deep_starmap_13_14`, matching the input and fixed protocol. It runs six ratios for one case per invocation. The batch runner maps 06 to `dlpfc_0708`; use the interactive `RAFTUP_TUTORIAL_DATA` override for either other case.
- **07, spatial audit:** copy the desired run's `alignment_matrix.npz` into the same directory as `pair.json`. Verify its A/B order against those inputs. Add `audit_coordinate_scale` as two coordinate multipliers and `audit_spatial_cutoff` from that run's recorded recovery cutoff. This notebook also loads the feature cost through the shared loader, even though the audit itself uses coordinates and alignment support. It does not regenerate a missing alignment.

For the documented DLPFC example (original coordinates at 138 units per 100 µm), the multipliers are `[100/138, 100/138]` numerically, as in `pair.example.json`. Use cutoff 200 only if the selected run actually recorded a 200-µm recovery cutoff. Check `decision.json`; do not assume the example cutoff applies to a different selected template.

The raw-data preparation snippets are starting points. Exact paper reproduction also depends on preprocessing, cell filtering, feature costs, rotations and saved run settings. Newly generated costs may give different results. No public processed-input/result archive has yet been linked, so an external user cannot currently reproduce every saved paper figure from a raw download alone.
