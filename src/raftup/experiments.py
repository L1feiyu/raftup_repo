"""Small, explicit helpers for revision tutorials; no automatic ground-truth fitting."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


def load_pair(manifest):
    """Load two prepared h5ad files and an ordered .npy feature cost.

    Relative paths are resolved against the JSON manifest's directory.
    Identifiers are mandatory: matching shape alone cannot verify cell ordering.
    """
    import anndata as ad
    path = Path(manifest).expanduser().resolve()
    cfg = json.loads(path.read_text())
    def resolve(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else path.parent / p
    a, b = (ad.read_h5ad(resolve(cfg[k])) for k in ('slice_a', 'slice_b'))
    cost = np.load(resolve(cfg['feature_cost']), allow_pickle=False)
    for obj, key in [(a, 'cell_ids_a'), (b, 'cell_ids_b')]:
        ids = json.loads(resolve(cfg[key]).read_text())
        if list(obj.obs_names.astype(str)) != ids or not obj.obs_names.is_unique:
            raise ValueError(f'Cell order mismatch: {key}')
        xy = np.asarray(obj.obsm['spatial'])
        if xy.shape != (len(obj), 2) or not np.isfinite(xy).all():
            raise ValueError('spatial must be finite n-by-2 coordinates')
    if cost.shape != (len(a), len(b)) or not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError('Invalid feature-cost matrix')
    return a, b, cost, cfg


def save_alignment(P, decision, output):
    """Save final support and recorded decisions; refuse overwriting an existing run."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'alignment_matrix.npz').exists():
        raise FileExistsError('Choose a fresh output directory')
    sparse.save_npz(output / 'alignment_matrix.npz', sparse.csr_matrix(P))
    (output / 'decision.json').write_text(json.dumps(decision, indent=2))


def coverage(P):
    p = sparse.csr_matrix(P)
    return {'coverage_A': float(np.asarray((p > 0).sum(axis=1)).ravel().astype(bool).mean()),
            'coverage_B': float(np.asarray((p > 0).sum(axis=0)).ravel().astype(bool).mean())}


def rigid_from_alignment(P, x, y):
    """Weighted proper rigid B-to-A transform, row-vector convention; no scaling.

    Uses all positive alignment weights. Does not use labels or reference coordinates.
    """
    p = sparse.coo_matrix(P)
    keep = p.data > 0
    i, j, w = p.row[keep], p.col[keep], p.data[keep]
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(i) < 2:
        raise ValueError('At least two positive correspondences are required')
    w = w / w.sum()
    cx, cy = (x[i] * w[:, None]).sum(0), (y[j] * w[:, None]).sum(0)
    u, _, vt = np.linalg.svd((y[j]-cy).T @ ((x[i]-cx)*w[:, None]))
    d = np.eye(2); d[-1, -1] = np.linalg.det(u @ vt)
    r = u @ d @ vt
    return r, cx-cy @ r


def purity_concordance(x, y_aligned, purity_a, purity_b, max_distance=100.):
    """Spearman and MAE on spatial mutual 1-NN pairs within a distance cutoff.

    Coordinates and max_distance must share units (micrometers for HT206B1).
    Purity arrays must follow the coordinate row order. Reject missing values.
    """
    x, y = np.asarray(x, float), np.asarray(y_aligned, float)
    a, b = np.asarray(purity_a, float), np.asarray(purity_b, float)
    if len(x) != len(a) or len(y) != len(b) or not all(np.isfinite(z).all() for z in [x,y,a,b]):
        raise ValueError('Coordinates and purity must be finite and correctly ordered')
    dist, j = cKDTree(y).query(x)
    back = cKDTree(x).query(y)[1]
    i = np.flatnonzero((back[j] == np.arange(len(x))) & (dist <= max_distance))
    jj = j[i]
    return {'n_pairs': len(i),
            'purity_spearman': float(spearmanr(a[i], b[jj]).statistic) if len(i)>1 else float('nan'),
            'purity_MAE': float(np.mean(abs(a[i]-b[jj]))) if len(i) else float('nan')}, i, jj


def tissue_nrmse(predicted, reference, tissue_diagonal):
    """XY RMSE / fixed reference tissue diagonal; arrays already share IDs and frame."""
    p, q = np.asarray(predicted, float), np.asarray(reference, float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 2 or tissue_diagonal <= 0:
        raise ValueError('Expected matched n-by-2 XY arrays and positive diagonal')
    return float(np.sqrt(np.mean(np.sum((p-q)**2, axis=1))) / tissue_diagonal)


def spatial_distortion(P, x, y, cutoff, block_size=256):
    """Exact unweighted audit over unordered distinct positive-support edge pairs.

    Includes edges sharing a source or target. Compares original pairwise
    distances, not fitted coordinates or recovery's anchor constraints.
    Complexity is quadratic in number of support edges, with bounded blocks.
    """
    p = sparse.csr_matrix(P); p.sum_duplicates(); p.eliminate_zeros()
    i, j = (p > 0).nonzero(); n = len(i)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if p.shape != (len(x),len(y)) or cutoff < 0 or block_size < 1:
        raise ValueError('Invalid shapes, cutoff, or block size')
    bins = np.array([0.,10.,25.,50.,100.,np.inf]); hist = np.zeros(5, dtype=np.int64)
    for start in range(0,n,block_size):
        end = min(start+block_size,n)
        for other in range(start,n,block_size):
            stop = min(other+block_size,n)
            excess = abs(cdist(x[i[start:end]],x[i[other:stop]])-
                         cdist(y[j[start:end]],y[j[other:stop]]))-cutoff
            if start == other: excess = excess[np.triu_indices(end-start,1)]
            else: excess = excess.ravel()
            hist += np.histogram(excess[excess>0],bins=bins)[0]
    total = n*(n-1)//2
    return {'n_edges':n, 'total_combinations':total, 'violating_combinations':int(hist.sum()),
            'ordinary_violation_rate':float(hist.sum()/total) if total else float('nan'),
            'excess_counts':hist.tolist(), 'excess_bin_edges':[0,10,25,50,100,'inf']}
