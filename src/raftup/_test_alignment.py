"""Experimental geometry routing and the two previously benchmarked workflows.
No evaluation metrics or anatomical label values enter decisions.
"""
import json
import tempfile
import time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from . import _cutoff_selection as cs
from . import _fsgw_utils_test_sparse
from ._test_overlap_router import route_slices
from ._test_localized_alignment import run as run_localized


# Fixed normalized-cost thresholds per class, not per-pair quantiles.
# The existing full-cost positive-entry p99 normalization is retained.
WINDOW_FEATURE_CUTOFFS = {
    'same_structure': 0.10,
    'merfish_partial_window': 0.58,
    'dlpfc_partial_window': 0.45,
}


def _window(slice_a, slice_b, cost, technology):
    normalized = cs._normalize_cost_percentile(cost, 99)
    row, col = normalized.argmin(axis=1), normalized.argmin(axis=0)
    mutual = float(np.mean(col[row] == np.arange(len(slice_a))))
    if mutual >= .35:
        kind, multiplier = 'same_structure', .365
    elif technology == 'merfish':
        kind, multiplier = 'merfish_partial_window', 2.
    else:
        kind, multiplier = 'dlpfc_partial_window', 2.
    xa, xb = np.asarray(slice_a.obsm['spatial']), np.asarray(slice_b.obsm['spatial'])
    da, db = cdist(xa, xa), cdist(xb, xb)
    nn_a, nn_b = da.copy(), db.copy()
    np.fill_diagonal(nn_a, np.inf)
    np.fill_diagonal(nn_b, np.inf)
    spacing = float(np.median([np.median(nn_a.min(axis=1)), np.median(nn_b.min(axis=1))]))
    spatial = float(round(max(1, spacing * multiplier)))
    feature = WINDOW_FEATURE_CUTOFFS[kind]
    state = np.random.get_state()
    try:
        np.random.seed(0)
        p = _fsgw_utils_test_sparse.fsgw_mvc_exp_sparse_heap(
            da, db, normalized, gw_cutoff=spatial, w_cutoff=feature, show_progress=False)
    finally:
        np.random.set_state(state)
    alignment = cs._harden_rows(p)
    return alignment, dict(selected_class=kind, feature_mutual_nn=mutual,
        ds_feature_quantile=None, ds_feature=feature, ds_spatial=spatial,
        feature_cutoff_mode='fixed_normalized_value',
        feature_cost_normalization='full_cost_divided_by_positive_entry_percentile99',
        cutoff_rule_version='window_fixed_v1',
        rf_feature=None, rf_spatial=None, k1=None, k2=None,
        selected_points={'A': len(slice_a), 'B': len(slice_b)},
        downsample_ratio={'A': 1., 'B': 1.}, small_window_mode=True)


def run_test_alignment(slice_a, slice_b, feature_cost=None, *, output_dir=None,
                       data_technology=None, router_seeds=(11, 29, 47),
                       scanit_kwargs=None, stability_runs=1, scanit_min_gain=.50):
    """Return (alignment, decision) using a geometry-selected experimental path.

    data_technology describes the actual assay for window rules/cost generation.
    If omitted, use the existing metadata-column technology inference. Large
    localized inputs retain benchmark spatial cutoffs358/274 in coordinate units.
    """
    actual = (data_technology or cs._infer_technology(slice_a, slice_b)).lower()
    if actual not in {'dlpfc', 'merfish'}:
        raise ValueError("data_technology must be 'dlpfc' or 'merfish'")
    for obj in (slice_a, slice_b):
        xy = np.asarray(obj.obsm['spatial'], dtype=float)
        if xy.shape != (len(obj), 2) or len(obj) < 3 or not np.isfinite(xy).all():
            raise ValueError('test mode requires at least three finite 2D spatial coordinates per slice')
    started = time.perf_counter()
    source = 'provided'
    cost_decision = None
    if feature_cost is None:
        options = dict(scanit_kwargs or {})
        options.setdefault('technology', actual)
        if options['technology'] != actual:
            raise ValueError('scanit_kwargs technology conflicts with data_technology')
        if max(len(slice_a), len(slice_b)) <= 800:
            costs = {'simple_pca': cs.compute_simple_pca_cost(slice_a, slice_b),
                     'scanit_dgi': cs.compute_scanit_cost(slice_a, slice_b, **options)}
            _, cost_decision, _ = cs._run_small_cost_candidates(
                slice_a, slice_b, costs, actual, stability_runs=stability_runs,
                scanit_min_gain=scanit_min_gain)
            source = cost_decision['feature_cost_source']
            feature_cost = costs[source]
        else:
            source = 'scanit_dgi'
            feature_cost = cs.compute_scanit_cost(slice_a, slice_b, **options)
    cost = np.asarray(cs._as_numpy(feature_cost), dtype=float)
    if cost.shape != (len(slice_a), len(slice_b)) or not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError('feature_cost must be a finite nonnegative matrix matching both slices')
    routing = route_slices(slice_a, slice_b, cost, router_seeds)
    route = routing['route']
    # A temporary directory supports the same function without output_dir.
    with tempfile.TemporaryDirectory(prefix='raftup_test_') as temporary:
        folder = Path(output_dir) if output_dir is not None else Path(temporary)
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'routing_decision.json').write_text(json.dumps(routing, indent=2))
        if route == 'synthetic_window':
            alignment, workflow = _window(slice_a, slice_b, cost, actual)
        elif route == 'simulated_localized':
            indices = None
            if actual == 'merfish':
                points = min(len(slice_a), len(slice_b), max(150, int(np.ceil(.067*min(len(slice_a),len(slice_b))))))
                _, _, ia, ib = cs._merfish_downsample_pair(slice_a, slice_b, points, 0)
                indices = (ia, ib)
            alignment, workflow = run_localized(slice_a, slice_b, cost, folder/'workflow', sampling_indices=indices)
        else:
            raise RuntimeError(f'Unknown route: {route}')
        decision = dict(technology='test', data_technology=actual, selected_workflow=route,
            routing=routing, workflow_decision=workflow, feature_cost_source=source,
            feature_cost_selection=cost_decision, selection_uses_ground_truth=False,
            experimental=True, total_seconds=time.perf_counter()-started,
            **cs._coverage(alignment))
        if output_dir is not None:
            np.save(folder/'alignment_matrix.npy', alignment)
            if source != 'provided':
                np.save(folder/'selected_feature_cost.npy', cost)
            (folder/'cutoff_decision.json').write_text(json.dumps(decision, indent=2))
    return alignment, decision
