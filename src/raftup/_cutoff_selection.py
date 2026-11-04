"""Automated DLPFC and MERFISH cutoff selection for RAFT-UP.

The public interface converts each input spatial coordinate system using
pixels_per_100um (defaults: DLPFC 138, MERFISH 100). Standard workflows use
micrometers internally; experimental test rules retain their calibrated frame. Anatomical labels and final
evaluation metrics are never used during selection.
"""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from . import _downsample, _fsgw_utils_test_sparse, _recoverfull_new_new_knn


@dataclass(frozen=True)
class CutoffTemplate:
    downsample_ratio: float
    ds_feature: float
    ds_spatial_um: int
    ds_spatial_px: int
    rf_feature: float
    rf_spatial_um: int
    rf_spatial_px: int
    k1: int
    k2: int


TEMPLATES = {
    "similar": CutoffTemplate(0.067, 0.10, 400, 552, 0.20, 200, 276, 20, 20),
    "partial_similar": CutoffTemplate(0.067, 0.10, 400, 552, 0.20, 200, 276, 20, 20),
    "far": CutoffTemplate(0.050, 0.20, 150, 206, 0.40, 150, 206, 10, 10),
}

MERFISH_TEMPLATES = {
    "standard": {
        "downsample_ratio": 0.050, "ds_feature": 0.20, "ds_spatial": 100,
        "rf_feature": 0.20, "rf_spatial": 60, "k1": 10, "k2": 10,
    },
    "coverage_adjusted": {
        "downsample_ratio": 0.050, "ds_feature": 0.20, "ds_spatial": 100,
        "rf_feature": 0.25, "rf_spatial": 60, "k1": 10, "k2": 10,
    },
    "distant": {
        "downsample_ratio": 0.067, "ds_feature": 0.20, "ds_spatial": 100,
        "rf_feature": 0.25, "rf_spatial": 60, "k1": 6, "k2": 6,
    },
}

FEATURE_NAMES = (
    "delta_coverage", "delta_collision", "delta_reciprocal",
    "delta_feature_cost", "delta_local_distortion", "delta_margin",
    "feature_low_cost_fraction", "feature_row_min_median",
    "feature_mutual_nn", "geometry_quantile_gap",
)

# Frozen StandardScaler and logistic-regression parameters fitted on the 18
# DLPFC table-labelled pairs. Keeping them here makes the runtime self-contained.
_SCALER_MEAN = np.array([
    -0.23692741216910854, -0.027418977639165707, -0.027418977639165707,
    0.03325511784189277, -0.004532197754462009, 0.028655918929467186,
    0.08876560938901652, 0.055018541403114796, 0.3317354281200245,
    0.011187695294949927,
])
_SCALER_SCALE = np.array([
    0.13063977494902415, 0.024557408027636632, 0.024557408027636632,
    0.019113203789396422, 0.007447858104518393, 0.016803106510708068,
    0.060094058432019294, 0.016782722853712174, 0.0630062873933192,
    0.005805259546778711,
])
_LOGISTIC_COEF = np.array([
    0.3982531535239037, -0.15349880831355228, -0.15349880831355228,
    -0.5160456283440669, 0.10109551721512909, -0.4985793302516191,
    -0.00790839974787421, -0.0796936359311848, 0.4519366143502603,
    -0.3294285223497757,
])
_LOGISTIC_INTERCEPT = 0.22508440278505662


def _as_numpy(matrix):
    if hasattr(matrix, "detach"):
        matrix = matrix.detach()
    if hasattr(matrix, "cpu"):
        matrix = matrix.cpu()
    if hasattr(matrix, "numpy"):
        matrix = matrix.numpy()
    return np.asarray(matrix, dtype=float)


def _downsample_pair(slice_a, slice_b, tar_distance):
    ds_a, idx_a = _downsample.downsample_slice(slice_a, int(tar_distance))
    ds_b, idx_b = _downsample.downsample_slice(slice_b, int(tar_distance))
    if ds_a is None or ds_b is None:
        raise RuntimeError(f"Empty downsample at tar_distance={tar_distance}")
    idx_a, idx_b = np.asarray(idx_a, dtype=int), np.asarray(idx_b, dtype=int)
    ratios = len(idx_a) / len(slice_a), len(idx_b) / len(slice_b)
    return ds_a, ds_b, idx_a, idx_b, ratios


def choose_tar_distance(
    slice_a,
    slice_b,
    target_ratio,
    *,
    max_ratio=0.10,
    max_points=800,
    min_points=150,
    distance_step=25,
    seed=0,
):
    """Choose a shared distance while keeping each downsample reliable.

    ``min_points`` takes precedence over ``max_ratio`` for a small slice.  This
    is needed for partial-overlap inputs, where applying the same percentage to
    a much smaller slice can otherwise leave too few points to estimate an
    anchor alignment.
    """
    if min_points < 1 or max_points < min_points:
        raise ValueError("Require 1 <= min_points <= max_points")
    tested = {}

    def evaluate(distance):
        distance = int(distance)
        if distance not in tested:
            state = np.random.get_state()
            try:
                np.random.seed(int(seed) + distance)
                tested[distance] = _downsample_pair(slice_a, slice_b, distance)
            finally:
                np.random.set_state(state)
        return tested[distance]

    sizes = np.array([len(slice_a), len(slice_b)], dtype=int)
    required = np.minimum(sizes, int(min_points))
    desired = np.minimum(
        sizes,
        np.maximum(required, np.ceil(target_ratio * sizes).astype(int)),
    )
    allowed = np.minimum(
        sizes,
        np.minimum(
            int(max_points),
            np.maximum(
                np.ceil(1.25 * required).astype(int),
                np.floor(max_ratio * sizes).astype(int),
            ),
        ),
    )
    # With one shared spatial distance, a small partial slice may need a much
    # denser sample to reach ``min_points``.  That same distance can push the
    # larger slice above its percentage cap, so in this asymmetric case the
    # absolute ``max_points`` guard is the meaningful upper bound.
    if np.any(np.floor(max_ratio * sizes).astype(int) < required):
        allowed = np.minimum(sizes, int(max_points))

    # Locate the distance corresponding to each slice's desired count.  The
    # union of neighborhoods covers the asymmetric case without an exhaustive
    # scan over every possible distance.
    centers = []
    for side in (0, 1):
        low, high = 1, 500
        while len(evaluate(high)[2 + side]) > desired[side]:
            low, high = high + 1, int(np.ceil(high * 1.5))
            if high > 10000:
                raise RuntimeError("Could not bracket target downsampling size")
        for _ in range(16):
            if low > high:
                break
            middle = (low + high) // 2
            if len(evaluate(middle)[2 + side]) > desired[side]:
                low = middle + 1
            else:
                high = middle - 1
        centers.append(int(round(((low + high) / 2) / distance_step) * distance_step))
    for center in centers:
        for offset in range(-5, 6):
            evaluate(max(distance_step, center + offset * distance_step))

    valid = {
        distance: value for distance, value in tested.items()
        if distance % distance_step == 0
        and np.all(np.array([len(value[2]), len(value[3])]) >= required)
        and np.all(np.array([len(value[2]), len(value[3])]) <= allowed)
    }
    if not valid:
        raise RuntimeError(
            "No shared tar_distance satisfies the downsampled point-count "
            f"bounds; required={required.tolist()}, allowed={allowed.tolist()}"
        )
    return min(
        valid.items(),
        key=lambda item: float(np.max(np.abs(
            np.array([len(item[1][2]), len(item[1][3])]) - desired
        ) / np.maximum(desired, 1))),
    )


def _downsampled_anchor_counts(p, tolerance=0.0):
    support = np.asarray(p) > tolerance
    return int(np.any(support, axis=1).sum()), int(np.any(support, axis=0).sum())


def _has_recovery_anchors(p, k1, k2, tolerance=0.0):
    """Return whether a downsampled plan can support kNN recovery."""
    rows, cols = _downsampled_anchor_counts(p, tolerance=tolerance)
    return rows >= int(k1) and cols >= int(k2), rows, cols


def _run_downsampled_alignment(ds_a, ds_b, idx_a, idx_b, feature_cost, template, *, seed=42):
    cost = feature_cost[np.ix_(idx_a, idx_b)].astype(float, copy=True)
    cost /= max(float(cost.max()), 1e-12)
    alignment = _fsgw_utils_test_sparse.fsgw_mvc_exp_sparse_heap(
        cdist(ds_a.obsm["spatial"], ds_a.obsm["spatial"]),
        cdist(ds_b.obsm["spatial"], ds_b.obsm["spatial"]),
        cost,
        gw_cutoff=template.ds_spatial_um,
        w_cutoff=template.ds_feature,
        seed=seed,
        show_progress=False,
    )
    return alignment, cost


def _pilot_quality(p, cost, xa, xb, local_k=5):
    support = p > 0
    rows, cols = support.sum(axis=1), support.sum(axis=0)
    covered = rows > 0
    row_choice, col_choice = p.argmax(axis=1), p.argmax(axis=0)
    reciprocal = covered & (col_choice[row_choice] == np.arange(len(rows)))
    unique_targets = len(np.unique(row_choice[covered])) if covered.any() else 0
    matched_cost = float(np.mean(cost[np.arange(len(rows))[covered], row_choice[covered]])) if covered.any() else 1.0
    if covered.any() and cost.shape[1] > 1:
        alternatives = cost.copy()
        alternatives[np.arange(len(rows)), row_choice] = np.inf
        margin = float(np.mean(
            alternatives[covered].min(axis=1)
            - cost[np.arange(len(rows))[covered], row_choice[covered]]
        ))
    else:
        margin = 0.0
    source_dist, target_dist = cdist(xa, xa), cdist(xb, xb)
    positive_target = target_dist[target_dist > 0]
    target_scale = np.median(positive_target) if len(positive_target) else 1.0
    distortion = []
    for i in np.where(covered)[0]:
        neighbors = np.argsort(source_dist[i])[1:local_k + 1]
        neighbors = neighbors[covered[neighbors]]
        if len(neighbors):
            d_source = source_dist[i, neighbors]
            d_target = np.linalg.norm(xb[row_choice[neighbors]] - xb[row_choice[i]], axis=1)
            distortion.extend((np.abs(d_source - d_target) / max(target_scale, 1e-12)).tolist())
    return {
        "coverage": float(min((rows > 0).mean(), (cols > 0).mean())),
        "collision": 1.0 - unique_targets / max(int(covered.sum()), 1),
        "reciprocal": float(reciprocal.sum() / max(covered.sum(), 1)),
        "feature_cost": matched_cost,
        "local_distortion": float(np.median(distortion)) if distortion else 1.0,
        "margin": margin,
    }


def _classifier_features(candidate_alignments, pilot_cost, xa, xb):
    quality = {
        name: _pilot_quality(candidate_alignments[name], pilot_cost, xa, xb)
        for name in ("similar", "far")
    }
    row_best, col_best = pilot_cost.argmin(axis=1), pilot_cost.argmin(axis=0)
    mutual_feature = np.mean(col_best[row_best] == np.arange(pilot_cost.shape[0]))
    da, db = cdist(xa, xa), cdist(xb, xb)
    qa = np.quantile(da[da > 0] / max(da.max(), 1e-12), [0.1, 0.25, 0.5, 0.75, 0.9])
    qb = np.quantile(db[db > 0] / max(db.max(), 1e-12), [0.1, 0.25, 0.5, 0.75, 0.9])
    values = np.array([
        quality["similar"]["coverage"] - quality["far"]["coverage"],
        quality["far"]["collision"] - quality["similar"]["collision"],
        quality["similar"]["reciprocal"] - quality["far"]["reciprocal"],
        quality["far"]["feature_cost"] - quality["similar"]["feature_cost"],
        quality["far"]["local_distortion"] - quality["similar"]["local_distortion"],
        quality["similar"]["margin"] - quality["far"]["margin"],
        float(np.mean(pilot_cost <= 0.1)),
        float(np.median(pilot_cost.min(axis=1))),
        float(mutual_feature),
        float(np.mean(np.abs(qa - qb))),
    ])
    return quality, values


def _predict_similar_probability(features):
    logit = float(((features - _SCALER_MEAN) / _SCALER_SCALE) @ _LOGISTIC_COEF + _LOGISTIC_INTERCEPT)
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -700, 700))))


def _harden_rows(p):
    hard = np.zeros_like(p)
    rows = np.arange(p.shape[0])
    hard[rows, p.argmax(axis=1)] = p[rows, p.argmax(axis=1)]
    return hard


def _coverage(p):
    return {
        "coverage_a": float(np.any(p != 0, axis=1).mean()),
        "coverage_b": float(np.any(p != 0, axis=0).mean()),
    }


def _run_dlpfc_automated_alignment(
    slice_a,
    slice_b,
    feature_cost,
    *,
    coverage_floor=0.80,
    pilot_ratio=0.05,
    max_ratio=0.10,
    max_points=800,
    min_points=150,
    distance_step=25,
    class_override=None,
):
    """Select DLPFC cutoffs, run downsampled alignment, and recover full mapping.

    Returns
    -------
    alignment : numpy.ndarray
        Dense full-resolution alignment matrix, shape ``(len(slice_a), len(slice_b))``.
    decision : dict
        Classifier output, selected cutoffs, downsampling information, coverage,
        timing, and every attempted fallback.
    """
    cost = _as_numpy(feature_cost)
    expected = (len(slice_a), len(slice_b))
    if cost.shape != expected:
        raise ValueError(f"feature_cost shape {cost.shape} != {expected}")
    if not np.isfinite(cost).all():
        raise ValueError("feature_cost contains non-finite values")
    if class_override not in (None, "similar", "partial_similar", "far"):
        raise ValueError("class_override must be None, 'similar', 'partial_similar', or 'far'")

    pilot_tar, pilot = choose_tar_distance(
        slice_a, slice_b, pilot_ratio,
        max_ratio=max_ratio, max_points=max_points, min_points=min_points,
        distance_step=distance_step,
    )
    pilot_a, pilot_b, pilot_idx_a, pilot_idx_b, pilot_ratios = pilot
    pilot_alignments, pilot_cost, pilot_seconds = {}, None, {}
    for name in ("similar", "far"):
        started = time.perf_counter()
        pilot_alignments[name], pilot_cost = _run_downsampled_alignment(
            pilot_a, pilot_b, pilot_idx_a, pilot_idx_b, cost, TEMPLATES[name]
        )
        pilot_seconds[name] = time.perf_counter() - started
    pilot_quality, features = _classifier_features(
        pilot_alignments, pilot_cost,
        pilot_a.obsm["spatial"], pilot_b.obsm["spatial"],
    )
    p_similar = _predict_similar_probability(features)
    base_prediction = "similar" if p_similar >= 0.5 else "far"
    field_size_ratio = min(len(slice_a), len(slice_b)) / max(len(slice_a), len(slice_b))
    predicted_class = "partial_similar" if field_size_ratio <= 0.82 else base_prediction
    initial_class = class_override or predicted_class

    attempts = []

    def run_attempt(pair_class, target_ratio, tag):
        template = TEMPLATES[pair_class]
        tar_distance, sampled = choose_tar_distance(
            slice_a, slice_b, target_ratio,
            max_ratio=max_ratio, max_points=max_points, min_points=min_points,
            distance_step=distance_step,
        )
        ds_a, ds_b, idx_a, idx_b, ratios = sampled
        started = time.perf_counter()
        p_ds, _ = _run_downsampled_alignment(ds_a, ds_b, idx_a, idx_b, cost, template)
        step1_seconds = time.perf_counter() - started
        valid_anchors, matched_a, matched_b = _has_recovery_anchors(
            p_ds, template.k1, template.k2
        )
        if valid_anchors:
            started = time.perf_counter()
            p_full = _recoverfull_new_new_knn.recover_full_mapping_knn(
                M=cost,
                X1=slice_a.obsm["spatial"],
                X2=slice_b.obsm["spatial"],
                P=_harden_rows(p_ds),
                idx1=idx_a,
                idx2=idx_b,
                k1=template.k1,
                k2=template.k2,
                thresh_CGW=template.rf_spatial_um,
                thresh_CCC=template.rf_feature,
                eps=0.01,
            ).toarray()
            step2_seconds = time.perf_counter() - started
            recovery_status = "completed"
        else:
            p_full = np.zeros((len(slice_a), len(slice_b)), dtype=float)
            step2_seconds = 0.0
            recovery_status = "rejected_insufficient_downsampled_anchors"
        record = {
            "tag": tag,
            "pair_class": pair_class,
            "target_ratio": target_ratio,
            "tar_distance": int(tar_distance),
            "anchors_a": int(len(idx_a)),
            "anchors_b": int(len(idx_b)),
            "matched_anchors_a": matched_a,
            "matched_anchors_b": matched_b,
            "minimum_recovery_anchors_a": int(template.k1),
            "minimum_recovery_anchors_b": int(template.k2),
            "recovery_status": recovery_status,
            "ratio_a": float(ratios[0]),
            "ratio_b": float(ratios[1]),
            "step1_seconds": step1_seconds,
            "step2_seconds": step2_seconds,
            **asdict(template),
            "downsample_ratio": target_ratio,
            **_coverage(p_full),
        }
        attempts.append(record)
        return p_full, record

    if initial_class == "partial_similar":
        if base_prediction == "similar":
            alignment, final = run_attempt(
                "partial_similar", TEMPLATES["partial_similar"].downsample_ratio,
                "01_partial_similar_067",
            )
        else:
            alignment, final = run_attempt(
                "far", TEMPLATES["far"].downsample_ratio,
                "01_partial_far_050",
            )
    elif initial_class == "similar":
        alignment, final = run_attempt("similar", TEMPLATES["similar"].downsample_ratio, "01_similar_067")
        if min(final["coverage_a"], final["coverage_b"]) < coverage_floor:
            alignment, final = run_attempt("similar", TEMPLATES["far"].downsample_ratio, "02_similar_050_fallback")
        if min(final["coverage_a"], final["coverage_b"]) < coverage_floor:
            alignment, final = run_attempt("far", TEMPLATES["far"].downsample_ratio, "03_far_050_fallback")
    else:
        alignment, final = run_attempt("far", TEMPLATES["far"].downsample_ratio, "01_far_050")

    decision = {
        "classifier_prediction": predicted_class,
        "base_classifier_prediction": base_prediction,
        "p_similar": p_similar,
        "field_size_ratio": field_size_ratio,
        "initial_class": initial_class,
        "selected_attempt": final["tag"],
        "coverage_floor": coverage_floor,
        "min_downsampled_points": int(min_points),
        "max_downsampled_points": int(max_points),
        "pilot_tar_distance": int(pilot_tar),
        "pilot_ratio_a": float(pilot_ratios[0]),
        "pilot_ratio_b": float(pilot_ratios[1]),
        "pilot_seconds": pilot_seconds,
        "pilot_quality": pilot_quality,
        "classifier_features": dict(zip(FEATURE_NAMES, features.tolist())),
        "attempts": attempts,
        "final": final,
    }
    return alignment, decision


def _normalize_cost_percentile(cost, percentile=99):
    cost = _as_numpy(cost).astype(float, copy=True)
    positive = cost[cost > 0]
    if not len(positive):
        raise ValueError("feature_cost has no positive entries")
    return cost / max(float(np.percentile(positive, percentile)), 1e-12)


def compute_simple_pca_cost(slice_a, slice_b, n_components=50, *, random_state=0):
    """Reproduce the cropped-window PCA cost used in the original notebooks."""
    import scanpy as sc
    from sklearn.metrics import pairwise_distances

    common = slice_a.var_names.intersection(slice_b.var_names)
    if not len(common):
        raise ValueError("The slices have no common genes")
    a, b = slice_a[:, common].copy(), slice_b[:, common].copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
    joined = a.concatenate(b, batch_key="batch", batch_categories=["a", "b"])
    sc.pp.scale(joined, max_value=10)
    n_components = min(n_components, joined.n_obs - 1, joined.n_vars - 1)
    sc.tl.pca(joined, n_comps=n_components, random_state=random_state)
    z = joined.obsm["X_pca"]
    cost = pairwise_distances(
        z[joined.obs["batch"] == "a"], z[joined.obs["batch"] == "b"],
        metric="euclidean",
    )
    return _normalize_cost_percentile(cost, 99)


def compute_scanit_cost(slice_a, slice_b, *, technology="dlpfc", n_top_genes=None,
                        n_epoch=3500, n_h=None, lr=2e-4, seed=0, device="cpu"):
    """Compute a technology-specific library SCAN-IT/DGI feature cost."""
    import scanpy as sc
    from . import _gene_cost_somde

    common = slice_a.var_names.intersection(slice_b.var_names)
    if not len(common):
        raise ValueError("The slices have no common genes")
    a, b = slice_a[:, common].copy(), slice_b[:, common].copy()
    spatial_a = np.asarray(a.obsm["spatial"]).copy()
    spatial_b = np.asarray(b.obsm["spatial"]).copy()
    for obj in (a, b):
        sc.pp.normalize_total(obj, target_sum=1e4); sc.pp.log1p(obj)
    technology = technology.lower()
    if technology not in {"dlpfc", "merfish"}:
        raise ValueError("technology must be 'dlpfc' or 'merfish'")
    if n_h is None:
        n_h = 32 if technology == "merfish" else 100
    if n_top_genes is None and technology == "dlpfc":
        n_top_genes = 1000
    joined = a.concatenate(b, batch_key="batch", batch_categories=["a", "b"])
    if n_top_genes is not None and joined.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(joined, n_top_genes=n_top_genes)
        genes = joined.var_names[joined.var["highly_variable"]]
        a, b = a[:, genes].copy(), b[:, genes].copy()
    joined = a.concatenate(b, batch_key="batch", batch_categories=["a", "b"])
    sc.pp.scale(joined, max_value=10)
    a = joined[joined.obs["batch"] == "a"].copy()
    b = joined[joined.obs["batch"] == "b"].copy()
    a.obsm["spatial"], b.obsm["spatial"] = spatial_a, spatial_b
    with tempfile.TemporaryDirectory(prefix="raftup_scanit_") as tmp:
        cost = _gene_cost_somde.compute_gene_cost(
            a, b, section_id_A="A", section_id_B="B", n_h=n_h,
            n_epoch=n_epoch, lr=lr, print_step=max(n_epoch, 1), seed=seed,
            device=device, output_dir=tmp,
        )
    return _normalize_cost_percentile(cost, 99)


def _run_small_window_alignment(slice_a, slice_b, feature_cost, technology, *, seed=0):
    """Align an already-cropped window without reducing it to an unusably small pilot."""
    cost = _as_numpy(feature_cost)
    expected = (len(slice_a), len(slice_b))
    if cost.shape != expected:
        raise ValueError(f"feature_cost shape {cost.shape} != {expected}")
    normalized = _normalize_cost_percentile(cost, 99)
    rows = np.arange(len(slice_a))
    row_best, col_best = normalized.argmin(axis=1), normalized.argmin(axis=0)
    mutual = float(np.mean(col_best[row_best] == rows))
    xa = np.asarray(slice_a.obsm["spatial"], dtype=float)
    xb = np.asarray(slice_b.obsm["spatial"], dtype=float)
    da, db = cdist(xa, xa), cdist(xb, xb)
    nn_a = np.min(np.where(da > 0, da, np.inf), axis=1)
    nn_b = np.min(np.where(db > 0, db, np.inf), axis=1)
    spatial_scale = float(np.median(np.r_[nn_a[np.isfinite(nn_a)], nn_b[np.isfinite(nn_b)]]))
    strong_feature_identity = mutual >= 0.35
    if technology == "merfish":
        spatial_cutoff = max(1.0, 5.0 * np.ceil((2.0 * spatial_scale) / 5.0))
    else:
        spatial_cutoff = max(1.0, (
            np.round(0.365 * spatial_scale) if strong_feature_identity
            else np.ceil(spatial_scale)
        ))
    if strong_feature_identity:
        feature_cutoff = 0.10
    else:
        raw_feature_cutoff = np.quantile(normalized.min(axis=1), 0.90) + 0.10
        feature_cutoff = float(np.clip(0.05 * np.ceil(raw_feature_cutoff / 0.05), 0.20, 0.55))
    started = time.perf_counter()
    np.random.seed(seed)
    alignment = _fsgw_utils_test_sparse.fsgw_mvc_exp_sparse_heap(
        da, db, normalized, gw_cutoff=spatial_cutoff,
        w_cutoff=feature_cutoff, show_progress=False,
    )
    elapsed = time.perf_counter() - started
    decision = {
        "technology": technology,
        "classifier_prediction": "partial_similar",
        "selected_class": "partial_similar",
        "small_window_mode": True,
        "classifier_features": {
            "feature_mutual_nn": mutual,
            "median_nearest_spatial_distance": spatial_scale,
        },
        "selected_points": {"A": len(slice_a), "B": len(slice_b)},
        "downsample_ratio": {"A": 1.0, "B": 1.0},
        "ds_feature": feature_cutoff,
        "ds_spatial": spatial_cutoff,
        "rf_feature": None,
        "rf_spatial": None,
        "k1": None, "k2": None,
        "step1_seconds": elapsed,
        "step2_seconds": 0.0,
        "total_seconds": elapsed,
        **_coverage(alignment),
    }
    return alignment, decision


def _alignment_support(alignment):
    covered = np.any(alignment > 0, axis=1)
    return {(int(i), int(alignment[i].argmax())) for i in np.where(covered)[0]}


def _small_candidate_score(alignment, normalized_cost, xa, xb, stability):
    quality = _pilot_quality(alignment, normalized_cost, xa, xb)
    score = (
        1.0 * quality["coverage"]
        + 0.8 * quality["reciprocal"]
        - 0.8 * quality["collision"]
        - 0.7 * quality["local_distortion"]
        - 0.8 * quality["feature_cost"]
        + 0.8 * quality["margin"]
        + (0.8 * stability if stability is not None else 0.0)
    )
    return float(score), quality


def _run_small_cost_candidates(slice_a, slice_b, costs, technology, stability_runs=1,
                               scanit_min_gain=0.50):
    candidates = {}
    xa, xb = np.asarray(slice_a.obsm["spatial"]), np.asarray(slice_b.obsm["spatial"])
    for name, cost in costs.items():
        p0, d0 = _run_small_window_alignment(slice_a, slice_b, cost, technology, seed=0)
        stability = None
        if stability_runs >= 2:
            p1, _ = _run_small_window_alignment(slice_a, slice_b, cost, technology, seed=1)
            s0, s1 = _alignment_support(p0), _alignment_support(p1)
            stability = len(s0 & s1) / max(len(s0 | s1), 1)
        normalized = _normalize_cost_percentile(cost, 99)
        score, quality = _small_candidate_score(p0, normalized, xa, xb, stability)
        candidates[name] = {
            "alignment": p0, "decision": d0, "selection_score": score,
            "stability": stability, "quality": quality,
        }
    selected = max(candidates, key=lambda name: candidates[name]["selection_score"])
    if {"simple_pca", "scanit_dgi"}.issubset(candidates):
        gain = (candidates["scanit_dgi"]["selection_score"]
                - candidates["simple_pca"]["selection_score"])
        if gain < scanit_min_gain:
            selected = "simple_pca"
    else:
        gain = None
    result = candidates[selected]
    decision = dict(result["decision"])
    decision.update({
        "feature_cost_source": selected,
        "feature_cost_selection_uses_ground_truth": False,
        "scanit_score_gain_over_simple_pca": gain,
        "scanit_min_gain": scanit_min_gain,
        "feature_cost_candidates": {
            name: {k: v for k, v in item.items() if k not in {"alignment", "decision"}}
            for name, item in candidates.items()
        },
    })
    return result["alignment"], decision, candidates


def _merfish_downsample_pair(slice_a, slice_b, points, random_seed):
    np.random.seed(random_seed)
    ds_a, idx_a = _downsample.maxmin_downsample_slice_merfish(slice_a, int(points))
    ds_b, idx_b = _downsample.maxmin_downsample_slice_merfish(slice_b, int(points))
    return ds_a, ds_b, np.asarray(idx_a, dtype=int), np.asarray(idx_b, dtype=int)


def _run_merfish_downsampled_alignment(ds_a, ds_b, idx_a, idx_b, feature_cost, template):
    cost = feature_cost[np.ix_(idx_a, idx_b)].astype(float, copy=True)
    cost /= max(float(cost.max()), 1e-12)
    alignment = _fsgw_utils_test_sparse.fsgw_mvc_exp_sparse_heap(
        cdist(ds_a.obsm["spatial"], ds_a.obsm["spatial"]),
        cdist(ds_b.obsm["spatial"], ds_b.obsm["spatial"]),
        cost,
        gw_cutoff=template["ds_spatial"],
        w_cutoff=template["ds_feature"],
        show_progress=False,
    )
    return alignment, cost


def _merfish_classifier_features(pilot_alignment, pilot_cost):
    rows = np.arange(pilot_alignment.shape[0])
    covered = np.any(pilot_alignment > 0, axis=1)
    selected = pilot_alignment.argmax(axis=1)
    if covered.any() and pilot_cost.shape[1] > 1:
        alternatives = pilot_cost.copy()
        alternatives[rows, selected] = np.inf
        feature_margin = float(np.mean(
            alternatives[covered].min(axis=1)
            - pilot_cost[rows[covered], selected[covered]]
        ))
    else:
        feature_margin = 0.0
    row_best, col_best = pilot_cost.argmin(axis=1), pilot_cost.argmin(axis=0)
    feature_mutual_nn = float(np.mean(col_best[row_best] == rows))
    return {
        "feature_mutual_nn": feature_mutual_nn,
        "feature_margin": feature_margin,
    }


def _predict_merfish_template(classifier_features):
    mutual = classifier_features["feature_mutual_nn"]
    margin = classifier_features["feature_margin"]
    if mutual <= 0.3475:
        return "distant", float(0.3475 - mutual)
    if mutual >= 0.3675 and margin <= -0.035013:
        return "coverage_adjusted", float(min(mutual - 0.3675, -0.035013 - margin))
    distance = min(
        mutual - 0.3475,
        max(0.3675 - mutual, margin - (-0.035013)),
    )
    return "standard", float(max(distance, 0.0))


def _run_merfish_automated_alignment(
    slice_a,
    slice_b,
    feature_cost,
    *,
    max_ratio=0.10,
    max_points=800,
    min_points=150,
    pilot_points=200,
    random_seed=23,
    class_override=None,
):
    """Select MERFISH cutoffs and return a full-resolution alignment."""
    cost = _as_numpy(feature_cost)
    expected = (len(slice_a), len(slice_b))
    if cost.shape != expected:
        raise ValueError(f"feature_cost shape {cost.shape} != {expected}")
    if not np.isfinite(cost).all():
        raise ValueError("feature_cost contains non-finite values")
    if class_override not in (None, *MERFISH_TEMPLATES):
        raise ValueError(f"class_override must be one of {tuple(MERFISH_TEMPLATES)}")

    total_started = time.perf_counter()
    pilot_a, pilot_b, pilot_idx_a, pilot_idx_b = _merfish_downsample_pair(
        slice_a, slice_b, min(pilot_points, len(slice_a), len(slice_b)), random_seed
    )
    pilot_started = time.perf_counter()
    pilot_alignment, pilot_cost = _run_merfish_downsampled_alignment(
        pilot_a, pilot_b, pilot_idx_a, pilot_idx_b, cost,
        MERFISH_TEMPLATES["standard"],
    )
    pilot_seconds = time.perf_counter() - pilot_started
    classifier_features = _merfish_classifier_features(pilot_alignment, pilot_cost)
    predicted_class, boundary_distance = _predict_merfish_template(classifier_features)
    selected_class = class_override or predicted_class
    template = MERFISH_TEMPLATES[selected_class]

    smaller_slice = min(len(slice_a), len(slice_b))
    required_points = min(int(min_points), smaller_slice)
    points = max(required_points, int(round(template["downsample_ratio"] * smaller_slice)))
    ratio_cap = int(np.floor(max_ratio * smaller_slice))
    allowed_points = min(int(max_points), max(required_points, ratio_cap))
    points = min(points, allowed_points, smaller_slice)
    if points < 2:
        raise ValueError("Slices are too small for MERFISH downsampling")
    ds_a, ds_b, idx_a, idx_b = _merfish_downsample_pair(
        slice_a, slice_b, points, random_seed
    )
    step1_started = time.perf_counter()
    downsample_alignment, _ = _run_merfish_downsampled_alignment(
        ds_a, ds_b, idx_a, idx_b, cost, template
    )
    step1_seconds = time.perf_counter() - step1_started
    valid_anchors, matched_a, matched_b = _has_recovery_anchors(
        downsample_alignment, template["k1"], template["k2"]
    )
    if valid_anchors:
        step2_started = time.perf_counter()
        alignment = _recoverfull_new_new_knn.recover_full_mapping_knn(
            M=cost,
            X1=slice_a.obsm["spatial"], X2=slice_b.obsm["spatial"],
            P=_harden_rows(downsample_alignment), idx1=idx_a, idx2=idx_b,
            k1=template["k1"], k2=template["k2"],
            thresh_CGW=template["rf_spatial"],
            thresh_CCC=template["rf_feature"], eps=0.01,
        ).toarray()
        step2_seconds = time.perf_counter() - step2_started
        recovery_status = "completed"
    else:
        alignment = np.zeros((len(slice_a), len(slice_b)), dtype=float)
        step2_seconds = 0.0
        recovery_status = "rejected_insufficient_downsampled_anchors"
    decision = {
        "technology": "merfish",
        "classifier_prediction": predicted_class,
        "classifier_boundary_distance": boundary_distance,
        "selected_class": selected_class,
        "classifier_features": classifier_features,
        "pilot_points": int(len(pilot_idx_a)),
        "pilot_seconds": pilot_seconds,
        "selected_points": int(points),
        "minimum_downsampled_points": int(required_points),
        "matched_anchors_a": matched_a,
        "matched_anchors_b": matched_b,
        "recovery_status": recovery_status,
        "ratio_a": float(points / len(slice_a)),
        "ratio_b": float(points / len(slice_b)),
        **template,
        "step1_seconds": step1_seconds,
        "step2_seconds": step2_seconds,
        "total_seconds": time.perf_counter() - total_started,
        **_coverage(alignment),
    }
    return alignment, decision


def _infer_technology(slice_a, slice_b):
    merfish_columns = {"cell_types", "Neuron_cluster_ID"}
    if merfish_columns.intersection(slice_a.obs.columns) and merfish_columns.intersection(slice_b.obs.columns):
        return "merfish"
    return "dlpfc"


def _run_automated_alignment_coordinates(slice_a, slice_b, feature_cost=None, *, technology="dlpfc",
                            output_dir=None, scanit_kwargs=None, stability_runs=1,
                            scanit_min_gain=0.50, **kwargs):
    """Run automated alignment, optionally constructing the feature cost.

    Omitted technology (or None) defaults to "dlpfc", without metadata inference.
    The experimental workflow router runs only for explicit technology="test".
    DLPFC/MERFISH cutoff selectors are unchanged.

    If no cost is supplied, large pairs use SCAN-IT/DGI only. Pairs with at
    most 800 observations per slice compare simple-PCA and SCAN-IT/DGI costs
    using alignment-only quality and stability features (never labels).

    technology="test" selects the experimental geometric router followed by
    the benchmarked window or localized simulated workflow. Pass
    data_technology="dlpfc"/"merfish" for the actual assay when metadata does
    not identify it. This mode is calibrated on the nine research cases.
    """
    technology = ("dlpfc" if technology is None else technology).lower()
    if technology not in {"dlpfc", "merfish", "test"}:
        raise ValueError("technology must be 'dlpfc', 'merfish', or 'test'")
    if technology == "test":
        from ._test_alignment import run_test_alignment
        return run_test_alignment(
            slice_a, slice_b, feature_cost, output_dir=output_dir,
            scanit_kwargs=scanit_kwargs, stability_runs=stability_runs,
            scanit_min_gain=scanit_min_gain, **kwargs)
    generated_candidates = None
    if feature_cost is None:
        scanit_kwargs = dict(scanit_kwargs or {})
        scanit_kwargs.setdefault("technology", technology)
        if max(len(slice_a), len(slice_b)) <= 800:
            costs = {
                "simple_pca": compute_simple_pca_cost(slice_a, slice_b),
                "scanit_dgi": compute_scanit_cost(slice_a, slice_b, **scanit_kwargs),
            }
            alignment, decision, generated_candidates = _run_small_cost_candidates(
                slice_a, slice_b, costs, technology, stability_runs=stability_runs,
                scanit_min_gain=scanit_min_gain,
            )
        else:
            feature_cost = compute_scanit_cost(slice_a, slice_b, **scanit_kwargs)
            alignment, decision = (
                _run_merfish_automated_alignment(slice_a, slice_b, feature_cost, **kwargs)
                if technology == "merfish" else
                _run_dlpfc_automated_alignment(slice_a, slice_b, feature_cost, **kwargs)
            )
            decision["feature_cost_source"] = "scanit_dgi"
            decision["feature_cost_selection_uses_ground_truth"] = False
    elif max(len(slice_a), len(slice_b)) <= 800:
        alignment, decision = _run_small_window_alignment(slice_a, slice_b, feature_cost, technology)
        decision["feature_cost_source"] = "provided"
    elif technology == "merfish":
        alignment, decision = _run_merfish_automated_alignment(slice_a, slice_b, feature_cost, **kwargs)
        decision["feature_cost_source"] = "provided"
    elif technology == "dlpfc":
        alignment, decision = _run_dlpfc_automated_alignment(slice_a, slice_b, feature_cost, **kwargs)
        decision["feature_cost_source"] = "provided"
    else:
        raise ValueError("technology must be 'dlpfc' or 'merfish'")

    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "alignment_matrix.npy", alignment)
        (folder / "cutoff_decision.json").write_text(json.dumps(decision, indent=2))
        if generated_candidates is not None:
            for name, item in generated_candidates.items():
                np.save(folder / f"alignment_matrix_{name}.npy", item["alignment"])
    return alignment, decision



def _prepare_spatial_units(slice_a, slice_b, technology, pixels_per_100um, data_technology=None):
    """Copy inputs and convert XY; never mutate caller-owned coordinates."""
    actual = (data_technology or _infer_technology(slice_a, slice_b)) if technology == "test" else technology
    if actual not in {"dlpfc", "merfish"}:
        raise ValueError("data_technology must be 'dlpfc' or 'merfish'")
    default = 138.0 if actual == "dlpfc" else 100.0
    calibration = np.asarray(default if pixels_per_100um is None else pixels_per_100um, dtype=float)
    if calibration.ndim == 0:
        calibration = np.repeat(calibration, 2)
    if calibration.shape != (2,) or not np.isfinite(calibration).all() or (calibration <= 0).any():
        raise ValueError("pixels_per_100um must be a positive finite scalar or (A, B) pair")
    # Experimental rules were calibrated in native DLPFC pixels / MERFISH um.
    # Convert through um into that canonical frame, preserving existing test thresholds.
    internal_per_um = default / 100.0 if technology == "test" else 1.0
    result = []
    for obj, pixels in zip((slice_a, slice_b), calibration):
        xy = np.asarray(obj.obsm["spatial"], dtype=float)
        if xy.ndim != 2 or xy.shape != (len(obj), 2) or not np.isfinite(xy).all():
            raise ValueError("obsm['spatial'] must contain finite XY coordinates, shape (n, 2)")
        copied = obj.copy()
        copied.obsm["spatial"] = xy * (100.0 * internal_per_um / pixels)
        result.append(copied)
    info = dict(pixels_per_100um_A=float(calibration[0]), pixels_per_100um_B=float(calibration[1]),
                default_calibration_used=pixels_per_100um is None,
                micrometers_per_input_unit_A=float(100/calibration[0]),
                micrometers_per_input_unit_B=float(100/calibration[1]),
                internal_coordinate_unit="um" if internal_per_um == 1 else "calibrated_dlpfc_pixel",
                internal_units_per_um=internal_per_um, input_coordinates_modified=False)
    return result[0], result[1], info


def run_automated_alignment(slice_a, slice_b, feature_cost=None, *, technology="dlpfc",
                            pixels_per_100um=None, output_dir=None, scanit_kwargs=None,
                            stability_runs=1, scanit_min_gain=0.50, **kwargs):
    """Align slices with explicit spatial calibration.

    pixels_per_100um : float or (float, float), optional
        Number of input coordinate units per 100 micrometers, shared or (A, B).
        Defaults: 138 for DLPFC, 100 for MERFISH. For coordinates already in um,
        pass 100 even for DLPFC. Assumes isotropic XY scaling within each slice.
        Input AnnData objects are copied and remain unchanged.

    DLPFC/MERFISH selectors, downsampling and recovery use um internally.
    DLPFC similar cutoffs are 400/200 um; far cutoffs are exactly 150/150 um.
    distance_step, if supplied, is in um for the DLPFC workflow (default 25 um).
    Historical *_spatial_px template fields are retained as legacy metadata only.
    technology='test' keeps experimental thresholds in their calibrated frame;
    its calibration default is determined by data_technology or assay inference.
    Supplied feature costs are reused without modification. Missing costs are
    computed after coordinate conversion. Evaluation on original input coordinates
    still requires separately specifying the intended metric distance units.
    """
    technology = (technology or "dlpfc").lower()
    if technology not in {"dlpfc", "merfish", "test"}:
        raise ValueError("technology must be 'dlpfc', 'merfish', or 'test'")
    a, b, spatial_units = _prepare_spatial_units(
        slice_a, slice_b, technology, pixels_per_100um, kwargs.get("data_technology"))
    alignment, decision = _run_automated_alignment_coordinates(
        a, b, feature_cost, technology=technology, output_dir=output_dir,
        scanit_kwargs=scanit_kwargs, stability_runs=stability_runs,
        scanit_min_gain=scanit_min_gain, **kwargs)
    decision["spatial_units"] = spatial_units
    if output_dir is not None:
        (Path(output_dir) / "cutoff_decision.json").write_text(json.dumps(decision, indent=2))
    return alignment, decision

from ._manual_alignment import run_manual_alignment

__all__ = [
    "CutoffTemplate", "TEMPLATES", "MERFISH_TEMPLATES",
    "choose_tar_distance", "compute_simple_pca_cost", "compute_scanit_cost",
    "run_automated_alignment", "run_manual_alignment",
]
