# Data downloads and directory layout

**Datasets, cached feature costs, alignment matrices and generated results are not uploaded to GitHub.** Download them separately; keep them in the ignored `data/` directory or another local disk. All notebooks support environment-variable path overrides. Do not commit executed notebooks containing data or large output images.

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
