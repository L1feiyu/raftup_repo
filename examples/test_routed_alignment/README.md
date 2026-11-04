# Experimental `technology="test"` interface

The public entry point defaults to `technology="dlpfc"` (also when explicitly passed `None`), without inferring the workflow from annotation columns. Use `technology="merfish"` explicitly for the MERFISH workflow. Only explicit `technology="test"` activates the experimental workflow router. The DLPFC/MERFISH cutoff classifiers remain active within their respective workflows.

Use the existing entry point with two slices and their feature-cost matrix:

```python
from raftup._cutoff_selection import run_automated_alignment

alignment, decision = run_automated_alignment(
    sliceA, sliceB, feature_cost,
    technology="test",
    output_dir=OUTPUT_DIR,
)
print(decision["selected_workflow"])
print(decision["routing"]["successes"])
```

For an uninstalled local checkout, put the checkout's `src` directory on `sys.path` before importing raftup. Restart a notebook kernel if it has already imported an older module.

The experimental router checks geometric containment support using three random seeds, two candidate directions, a two-spacing spatial radius and a 98% threshold. Two passing seeds choose `simulated_localized`; otherwise it uses `synthetic_window`. These are development rules from nine research cases, not an independently validated biological classifier. The negative branch is a workflow fallback, not proof of bilateral partial overlap. No ground-truth labels, G, LAA or GPR enter routing.

- `synthetic_window`: the previously benchmarked **fixed normalized-cost** window rules, full input points, direct hard alignment. These differ from the legacy small-window branch of `technology="dlpfc"`/`"merfish"`.
- `simulated_localized`: the frozen centered-cost placement, original downsampling, feature cutoffs 0.5/0.5, spatial cutoffs 358/274 **coordinate units**, k=4, compact active anchors and final rigid-distance filtering. Empty/insufficient pilot anchors refuse recovery. These spatial settings were benchmarked for DLPFC-derived simulated data, not established as cross-technology physical distances.

`technology="test"` is a workflow mode, not a sequencing assay. The actual assay is inferred using the existing metadata-column convention (`cell_types`/`Neuron_cluster_ID` on both inputs => MERFISH; otherwise DLPFC). If that metadata is unavailable, pass `data_technology="merfish"` or `"dlpfc"` explicitly. MERFISH localized sampling uses the original max-min sampler; no DLPFC grid sampler is used for that assay. The nine-case references use a supplied cost and all MERFISH windows route to the window path.

`feature_cost=None` preserves the existing cost-generation policy: for both slices <=800, construct simple PCA and SCAN-IT costs, use the existing label-free cost-candidate selector, then route and rerun the selected cost through the new workflow; larger inputs generate SCAN-IT only. SCAN-IT receives the actual assay, never the string `"test"`. The nine-case numerical reproduction uses supplied matrices; generated-cost dispatch is separately tested with mocks and is not claimed to reproduce those cached matrices.

Outputs:

- `alignment_matrix.npy`: final matrix, input A rows / input B columns.
- `cutoff_decision.json`: mode, actual assay, selected workflow, route diagnostics, feature-cost source and workflow parameters.
- `routing_decision.json`: route candidates, vote counts and thresholds (also available if later computation fails).
- `workflow/`: localized pilot, indices, inferred transformed coordinates and matrix before distance filtering, when applicable.
- `selected_feature_cost.npy`: selected generated cost when `feature_cost=None`.

LAA, GPR and ground-truth-based bounds remain external evaluations. Existing `technology="dlpfc"` and `"merfish"` behavior is unchanged.
