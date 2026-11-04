"""Explicit, single-attempt RAFT-UP alignment for reproducible experiments."""
from contextlib import contextmanager
from pathlib import Path
import json
import random
import time
import warnings
import numpy as np


@contextmanager
def _seeded(seed):
    """Restore caller Python/NumPy/Torch RNG state, also when a run fails."""
    import torch
    py, npstate = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            yield
        finally:
            random.setstate(py); np.random.set_state(npstate)


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def run_manual_alignment(
    slice_a, slice_b, feature_cost=None, *,
    downsampling_ratio,
    feature_cutoff_downsample=None,
    spatial_cutoff_downsample_um,
    feature_cutoff_recovery=None,
    spatial_cutoff_recovery_um=None,
    k=None, k1=None, k2=None,
    technology="dlpfc", pixels_per_100um=None, random_state=0,
    min_downsampled_points=150, max_downsampled_points=None,
    cost_method=None, cost_kwargs=None, distance_step_um=25, output_dir=None,
    alignment_mode="two_stage", window_class=None,
):
    """Run one downsampling/alignment/recovery attempt, without classification.

    alignment_mode='window' runs all-point direct alignment with the auto window's
    full-cost positive-entry p99 normalization and no recovery. Require ratio=1;
    omit recovery cutoffs and k. Supply a numerical feature cutoff or explicitly
    select window_class (same_structure=0.10, dlpfc_partial_window=0.45,
    merfish_partial_window=0.58). No classifier runs. The spatial cutoff is in um;
    to reproduce the DLPFC auto window's canonical 50/274 pixel thresholds use
    50/1.38 or 274/1.38 um. random_state=42 matches that solver's default.
    min_downsampled_points/distance_step_um apply only to two_stage; the maximum
    count, when explicitly supplied, also guards direct window graphs. No automatic mode switch by size.

    For the default two_stage mode the ratio and four cutoffs are required. Supply either k (scalar or A/B
    pair), or both k1 and k2. All spatial thresholds and distance_step_um are
    micrometers. Coordinates are copied and calibrated with the same helper as
    run_automated_alignment (defaults: 138 DLPFC, 100 MERFISH units/100 um).

    DLPFC targets the specified fraction of EACH slice with one shared grid
    distance. MERFISH retains the existing maxmin sampler: round(ratio * min
    slice size) points on EACH side, so the larger side has a lower ratio.
    The target is approximate; actual counts, unique counts and ratios are
    recorded. max_downsampled_points=None (default) imposes no additional upper
    limit. Above 800 actual points per slice a runtime notice is printed. An
    explicitly supplied maximum remains a hard limit. Conflicts with point
    bounds raise, never silently clip. A ratio
    of 1 uses all points (subject to the bounds) and still performs recovery.

    Cutoffs are absolute normalized-cost values, NOT quantiles: step 1 divides
    the sampled cost by its maximum; recovery uses its existing full-matrix
    99th-percentile cap/rescale (+1e-10). No cost-based routing or fallback.

    cost_method=None resolves to 'provided' when a matrix is supplied, otherwise
    'scanit'. Alternatives: 'joint_pca', 'scanit', 'provided'. cost_kwargs are
    passed to the selected existing cost builder; seed/technology are controlled
    by this interface. Supplied matrices require row/column input-cell order.

    random_state controls Python, NumPy, Torch and generated costs; the DLPFC
    grid sampler is deterministic. Seeds do not guarantee GPU determinism.
    The caller RNG state is restored. k is never reduced for too few anchors:
    invalid/empty anchors raise before recovery. Returns dense P and decision.
    """
    if alignment_mode not in {"two_stage", "window"}:
        raise ValueError("alignment_mode must be two_stage or window")
    if alignment_mode == "window":
        if any(v is not None for v in (feature_cutoff_recovery, spatial_cutoff_recovery_um, k, k1, k2)):
            raise ValueError("Direct window mode has no recovery; omit recovery cutoffs and k/k1/k2")
        return _run_manual_window(slice_a, slice_b, feature_cost,
            downsampling_ratio=downsampling_ratio, feature_cutoff=feature_cutoff_downsample,
            spatial_cutoff_um=spatial_cutoff_downsample_um, window_class=window_class,
            technology=technology, pixels_per_100um=pixels_per_100um, random_state=random_state,
            max_points=max_downsampled_points, cost_method=cost_method, cost_kwargs=cost_kwargs,
            output_dir=output_dir)
    if window_class is not None:
        raise ValueError("window_class applies only to alignment_mode='window'")
    if any(v is None for v in (feature_cutoff_downsample,feature_cutoff_recovery,spatial_cutoff_recovery_um)):
        raise ValueError("two_stage mode requires all four cutoffs")
    from . import _cutoff_selection as cs
    technology = (technology or "dlpfc").lower()
    if technology not in {"dlpfc", "merfish"}:
        raise ValueError("Manual mode requires the sampling technology 'dlpfc' or 'merfish'; no 'test' router is run")
    ratio = float(downsampling_ratio)
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("downsampling_ratio must be in (0, 1]")
    dsf, rff = float(feature_cutoff_downsample), float(feature_cutoff_recovery)
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (dsf, rff)):
        raise ValueError("Feature cutoffs must be finite normalized thresholds in [0, 1], not quantiles")
    dss, rfs = float(spatial_cutoff_downsample_um), float(spatial_cutoff_recovery_um)
    if not all(np.isfinite(v) and v > 0 for v in (dss, rfs)):
        raise ValueError("Spatial cutoffs must be positive finite micrometers")
    lower = _positive_int("min_downsampled_points", min_downsampled_points)
    upper = max(len(slice_a), len(slice_b)) if max_downsampled_points is None else _positive_int("max_downsampled_points", max_downsampled_points)
    step = _positive_int("distance_step_um", distance_step_um)
    if lower > upper:
        raise ValueError("min_downsampled_points must not exceed max_downsampled_points")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)) or not 0 <= random_state < 2**32:
        raise ValueError("random_state must be an integer in [0, 2**32)")
    seed = int(random_state)
    if k is not None:
        if k1 is not None or k2 is not None:
            raise ValueError("Supply k OR k1/k2, not both")
        if isinstance(k, (tuple, list)) and len(k) == 2:
            k1, k2 = k
        else:
            k1 = k2 = k
    k1, k2 = _positive_int("k1", k1), _positive_int("k2", k2)
    sizes = np.array([len(slice_a), len(slice_b)], dtype=int)
    if np.any(sizes < 2):
        raise ValueError("Each slice needs at least two observations")
    desired = (np.repeat(int(round(ratio * min(sizes))), 2) if technology == "merfish" and ratio < 1
               else np.ceil(ratio * sizes).astype(int))
    if np.any(desired < lower) or np.any(desired > upper):
        raise ValueError(f"Requested counts {desired.tolist()} conflict with bounds [{lower}, {upper}]; change ratio or explicitly change bounds")
    if np.any(desired < [k1, k2]):
        raise ValueError("Requested downsample counts are smaller than k1/k2")
    method = cost_method or ("provided" if feature_cost is not None else "scanit")
    if method not in {"provided", "scanit", "joint_pca"}:
        raise ValueError("cost_method must be provided, scanit, or joint_pca")
    if (method == "provided") != (feature_cost is not None):
        raise ValueError("Supply feature_cost only with cost_method='provided'; generated costs require feature_cost=None")
    opts = dict(cost_kwargs or {})
    if method == "provided" and opts:
        raise ValueError("cost_kwargs cannot be used with provided cost")
    if {"seed", "random_state", "technology"}.intersection(opts):
        raise ValueError("Set random_state and technology on run_manual_alignment, not cost_kwargs")
    started = time.perf_counter()
    a, b, units = cs._prepare_spatial_units(slice_a, slice_b, technology, pixels_per_100um)
    with _seeded(seed):
        t = time.perf_counter()
        if method == "scanit":
            cost = cs.compute_scanit_cost(a, b, technology=technology, seed=seed, **opts)
        elif method == "joint_pca":
            cost = cs.compute_simple_pca_cost(a, b, random_state=seed, **opts)
        else:
            cost = cs._as_numpy(feature_cost)
        cost = cs._as_numpy(cost)
        if cost.shape != tuple(sizes) or not np.isfinite(cost).all() or np.any(cost < 0):
            raise ValueError(f"feature_cost must be finite, nonnegative, shape {tuple(sizes)}")
        cost_seconds = time.perf_counter() - t
        t = time.perf_counter()
        tar = None
        if ratio == 1:
            da, db, ia, ib = a, b, np.arange(len(a)), np.arange(len(b))
        elif technology == "merfish":
            da, db, ia, ib = cs._merfish_downsample_pair(a, b, int(desired[0]), seed)
        else:
            tar, sampled = cs.choose_tar_distance(a, b, ratio, max_ratio=max(.10, ratio),
                min_points=lower, max_points=upper, distance_step=step, seed=seed)
            da, db, ia, ib, _ = sampled
        counts = np.array([len(ia), len(ib)])
        if np.any(counts < lower) or np.any(counts > upper):
            raise RuntimeError("Actual sample count violates requested bounds")
        if np.any(counts > 800):
            print("it may take longer time since the size of downsampling is greater than 800.", flush=True)
        sampling_seconds = time.perf_counter() - t
        if np.any(np.abs(counts - desired) / desired > .20):
            warnings.warn("Actual sample counts differ from targets by >20%; inspect decision['sampling']", RuntimeWarning)
        t = time.perf_counter()
        template = cs.CutoffTemplate(ratio, dsf, dss, dss, rff, rfs, rfs, k1, k2)
        p, sampled_cost = cs._run_downsampled_alignment(da, db, ia, ib, cost, template, seed=seed)
        step1_seconds = time.perf_counter() - t
        hard = cs._harden_rows(np.asarray(p))
        valid, matched_a, matched_b = cs._has_recovery_anchors(hard, k1, k2)
        # Mirror recovery's delta=0.5 mass pruning before authorizing its kNN step.
        ra = hard.sum(1) >= .5 / len(ia)
        rb = hard.sum(0) >= .5 / len(ib)
        eligible = hard[np.ix_(ra, rb)]
        valid_pruned, pruned_a, pruned_b = cs._has_recovery_anchors(eligible, k1, k2)
        if not valid or not valid_pruned or not np.isfinite(hard).all() or np.any(hard < 0):
            raise RuntimeError(f"Recovery rejected: hardened anchors={matched_a}/{matched_b}, after mass pruning={pruned_a}/{pruned_b}, required k1/k2={k1}/{k2}")
        t = time.perf_counter()
        alignment = cs._recoverfull_new_new_knn.recover_full_mapping_knn(
            M=cost, X1=a.obsm["spatial"], X2=b.obsm["spatial"], P=hard,
            idx1=ia, idx2=ib, k1=k1, k2=k2, thresh_CGW=rfs, thresh_CCC=rff, eps=.01).toarray()
        step2_seconds = time.perf_counter() - t
    decision = dict(mode="manual", alignment_mode="two_stage", technology=technology, random_state=seed,
        feature_cost_source=method, cost_kwargs=opts, spatial_units=units,
        parameters=dict(downsampling_ratio=ratio,feature_cutoff_downsample=dsf,
            spatial_cutoff_downsample_um=dss,feature_cutoff_recovery=rff,
            spatial_cutoff_recovery_um=rfs,k1=k1,k2=k2,
            min_downsampled_points=lower,max_downsampled_points=max_downsampled_points,distance_step_um=step),
        sampling=dict(method="hexagonal_grid" if technology=="dlpfc" else "maxmin",
            ratio_definition="each_slice_target" if technology=="dlpfc" or ratio==1 else "equal_counts_from_smaller_slice",
            requested_counts=desired.tolist(),actual_counts=counts.tolist(),
            unique_counts=[len(np.unique(ia)),len(np.unique(ib))],
            ratio_a=float(len(ia)/len(a)),ratio_b=float(len(ib)/len(b)),tar_distance_um=tar,
            indices_a=ia.tolist(),indices_b=ib.tolist()),
        normalization=dict(downsample="sampled_cost / max(sampled_cost)",
            recovery="min(full_cost, percentile99(full_cost)) / percentile99(full_cost) + 1e-10",
            cutoff_type="absolute_normalized_value_not_quantile"),
        anchors=dict(hard_a=matched_a,hard_b=matched_b,pruned_a=pruned_a,pruned_b=pruned_b),
        classifier_used=False, fallback_used=False, recovery_status="completed",
        cost_seconds=cost_seconds,sampling_seconds=sampling_seconds,
        step1_seconds=step1_seconds,step2_seconds=step2_seconds,
        total_seconds=time.perf_counter()-started, **cs._coverage(alignment))
    if output_dir is not None:
        folder = Path(output_dir); folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "alignment_matrix.npy", alignment)
        np.save(folder / "downsampled_alignment.npy", p)
        np.savez_compressed(folder / "downsample_indices.npz", A=ia, B=ib)
        (folder / "cell_order.json").write_text(json.dumps(dict(A=slice_a.obs_names.tolist(),B=slice_b.obs_names.tolist())))
        (folder / "cutoff_decision.json").write_text(json.dumps(decision, indent=2))
    return alignment, decision



def _run_manual_window(slice_a, slice_b, feature_cost, *, downsampling_ratio,
                       feature_cutoff, spatial_cutoff_um, window_class, technology,
                       pixels_per_100um, random_state, max_points, cost_method,
                       cost_kwargs, output_dir):
    """Direct window alignment with the fixed-cost convention of the auto window."""
    from . import _cutoff_selection as cs
    from ._test_alignment import WINDOW_FEATURE_CUTOFFS
    from scipy.spatial.distance import cdist
    technology = (technology or "dlpfc").lower()
    if technology not in {"dlpfc", "merfish"}:
        raise ValueError("Window manual technology must identify the assay: dlpfc or merfish")
    if float(downsampling_ratio) != 1.0:
        raise ValueError("alignment_mode='window' uses all points; require downsampling_ratio=1")
    if window_class is not None and window_class not in WINDOW_FEATURE_CUTOFFS:
        raise ValueError(f"window_class must be one of {tuple(WINDOW_FEATURE_CUTOFFS)}")
    if feature_cutoff is None:
        if window_class is None:
            raise ValueError("Supply feature_cutoff_downsample or an explicit window_class")
        feature_cutoff = WINDOW_FEATURE_CUTOFFS[window_class]
    feature, spatial = float(feature_cutoff), float(spatial_cutoff_um)
    if not np.isfinite(feature) or not 0 <= feature <= 1:
        raise ValueError("feature_cutoff_downsample must be in [0, 1]")
    if not np.isfinite(spatial) or spatial <= 0:
        raise ValueError("spatial_cutoff_downsample_um must be positive and finite")
    limit = max(len(slice_a), len(slice_b)) if max_points is None else _positive_int("max_downsampled_points", max_points)
    if min(len(slice_a), len(slice_b)) < 3 or max(len(slice_a), len(slice_b)) > limit:
        raise ValueError("Direct window mode requires 3..max_downsampled_points per slice")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)) or not 0 <= random_state < 2**32:
        raise ValueError("random_state must be an integer in [0, 2**32)")
    if max(len(slice_a), len(slice_b)) > 800:
        print("it may take longer time since the size of downsampling is greater than 800.", flush=True)
    seed = int(random_state)
    method = cost_method or ("provided" if feature_cost is not None else "scanit")
    if method not in {"provided", "scanit", "joint_pca"} or ((method == "provided") != (feature_cost is not None)):
        raise ValueError("Use provided with feature_cost, or scanit/joint_pca without feature_cost")
    opts = dict(cost_kwargs or {})
    if (method == "provided" and opts) or {"seed", "random_state", "technology"}.intersection(opts):
        raise ValueError("Invalid cost_kwargs: control seed/technology on the main interface")
    started = time.perf_counter()
    # Match auto/test's calibrated frame exactly; the public spatial cutoff is still um.
    a, b, units = cs._prepare_spatial_units(slice_a, slice_b, "test", pixels_per_100um, technology)
    internal_cutoff = spatial * units["internal_units_per_um"]
    with _seeded(seed):
        t = time.perf_counter()
        if method == "provided": cost = cs._as_numpy(feature_cost)
        elif method == "scanit": cost = cs.compute_scanit_cost(a, b, technology=technology, seed=seed, **opts)
        else: cost = cs.compute_simple_pca_cost(a, b, random_state=seed, **opts)
        cost = cs._as_numpy(cost)
        if cost.shape != (len(a), len(b)) or not np.isfinite(cost).all() or (cost < 0).any():
            raise ValueError("Invalid feature_cost shape or entries")
        normalized = cs._normalize_cost_percentile(cost, 99)
        cost_seconds = time.perf_counter() - t
        t = time.perf_counter()
        p = cs._fsgw_utils_test_sparse.fsgw_mvc_exp_sparse_heap(
            cdist(a.obsm["spatial"], a.obsm["spatial"]),
            cdist(b.obsm["spatial"], b.obsm["spatial"]), normalized,
            gw_cutoff=internal_cutoff, w_cutoff=feature, seed=seed, show_progress=False)
        alignment = cs._harden_rows(np.asarray(p))
        step1_seconds = time.perf_counter() - t
    ia, ib = np.arange(len(a)), np.arange(len(b))
    decision = dict(mode="manual", alignment_mode="window", technology=technology,
        window_class=window_class, classifier_used=False, fallback_used=False,
        random_state=seed, feature_cost_source=method, cost_kwargs=opts, spatial_units=units,
        parameters=dict(downsampling_ratio=1., feature_cutoff_downsample=feature,
            spatial_cutoff_downsample_um=spatial, feature_cutoff_recovery=None,
            spatial_cutoff_recovery_um=None, k1=None, k2=None, max_downsampled_points=max_points),
        sampling=dict(method="all_points", actual_counts=[len(a), len(b)],
            unique_counts=[len(a), len(b)], ratio_a=1., ratio_b=1.,
            indices_a=ia.tolist(), indices_b=ib.tolist(), tar_distance_um=None),
        normalization=dict(downsample="full_cost / percentile99(positive_full_cost)",
            recovery=None, cutoff_type="absolute_normalized_value_not_quantile"),
        ds_feature_quantile=None, ds_feature=feature, ds_spatial=internal_cutoff,
        recovery_status="not_needed_all_spots_used", cost_seconds=cost_seconds,
        sampling_seconds=0., step1_seconds=step1_seconds, step2_seconds=0.,
        total_seconds=time.perf_counter()-started, **cs._coverage(alignment))
    if output_dir is not None:
        folder=Path(output_dir); folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/"alignment_matrix.npy",alignment)
        np.save(folder/"downsampled_alignment.npy",p)
        np.savez_compressed(folder/"downsample_indices.npz",A=ia,B=ib)
        (folder/"cutoff_decision.json").write_text(json.dumps(decision,indent=2))
        (folder/"cell_order.json").write_text(json.dumps(dict(A=slice_a.obs_names.tolist(),B=slice_b.obs_names.tolist())))
    return alignment, decision


__all__ = ["run_manual_alignment"]
