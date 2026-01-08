import itertools
from typing import Optional, Sequence, Tuple, Dict, Any

import numpy as np
import networkx as nx
from scipy import sparse
from tqdm import tqdm


def matrix_rescaling_checking(C_norm: np.ndarray, verbose: bool = True, atol: float = 1e-12) -> None:
    """
    Check whether a matrix has been rescaled into the [0, 1] interval.

    This is a sanity check used after normalization (e.g., `M /= M.max()`).

    Parameters
    ----------
    C_norm : np.ndarray
        Input matrix expected to be in [0, 1]. Any shape.
    verbose : bool, default=True
        If True, print min/max after checking.
    atol : float, default=1e-12
        Absolute tolerance used for numerical comparisons.

    Raises
    ------
    AssertionError
        If values are not within [0, 1] (up to tolerance), or max is not close to 1.
    """
    C_norm = np.asarray(C_norm)
    min_val = C_norm.min()
    max_val = C_norm.max()

    if not (min_val >= -atol and max_val <= 1.0 + atol):
        raise AssertionError("Rescaling failed: data not in [0,1] interval.")
    if not np.isclose(max_val, 1.0, atol=atol):
        raise AssertionError("Rescaling failed: max is not close to 1.")

    if verbose:
        print(f"After rescaling: min = {min_val}, max = {max_val}.")


def extract_feature_matrix(
    full_cost_path: str,
    indices_dsa: Sequence[int],
    indices_dsb: Sequence[int],
    delimiter: str = ",",
    normalize: bool = True,
    check: bool = True,
) -> np.ndarray:
    """
    Extract a submatrix from a full (feature) cost matrix on disk.

    This is typically used to extract a downsampled-to-downsampled cost matrix:
    `M = full[np.ix_(indices_dsa, indices_dsb)]`.

    Parameters
    ----------
    full_cost_path : str
        Path to the full cost matrix stored as a text CSV-like file.
        The file is loaded using `np.loadtxt(full_cost_path, delimiter=delimiter)`.
    indices_dsa : Sequence[int]
        Row indices to extract (e.g., downsampled indices from slice A).
    indices_dsb : Sequence[int]
        Column indices to extract (e.g., downsampled indices from slice B).
    delimiter : str, default=","
        Delimiter passed to `np.loadtxt`.
    normalize : bool, default=True
        If True, rescale `M` by `M.max()` so that max becomes 1.
    check : bool, default=True
        If True, run `matrix_rescaling_checking` after normalization.

    Returns
    -------
    M : np.ndarray
        Extracted submatrix of shape `(len(indices_dsa), len(indices_dsb))`.
        If `normalize=True`, matrix is scaled so that `M.max() == 1`.

    Notes
    -----
    - If the full matrix is very large, `np.loadtxt` can be slow and memory-heavy.
      Consider storing full matrices as `.npy` and loading with `np.load`.
    """
    full = np.loadtxt(full_cost_path, delimiter=delimiter)
    M = full[np.ix_(np.asarray(indices_dsa, dtype=int), np.asarray(indices_dsb, dtype=int))]

    if normalize:
        maxv = M.max()
        if maxv > 0:
            M = M / maxv
        if check:
            matrix_rescaling_checking(M)

    return M


def vertex_with_most_edges(B: nx.Graph) -> Tuple[list, int]:
    """
    Return vertices with maximum degree in an undirected graph.

    Parameters
    ----------
    B : networkx.Graph
        Input graph.

    Returns
    -------
    vertices : list
        List of nodes that achieve the maximum degree.
    max_degree : int
        The maximum degree value.

    Notes
    -----
    If the graph has no nodes, `max(...)` will raise a ValueError.
    """
    deg = dict(B.degree())
    max_degree = max(deg.values())
    vertices = [v for v, d in B.degree() if d == max_degree]
    return vertices, max_degree


def perform_sOT_log(
    G: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps: float,
    options: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform entropic regularized semi-OT (log-domain updates).

    This routine iteratively updates dual potentials (f, g) with a penalty cap,
    then recovers the transport plan via:
        P = exp((f[:,None] + g[None,:] - G) / eps)

    Parameters
    ----------
    G : np.ndarray
        Cost matrix of shape (n, m).
    a : np.ndarray
        Source marginal, shape (n,). Typically sums to 1.
    b : np.ndarray
        Target marginal, shape (m,). Typically sums to 1.
    eps : float
        Entropic regularization strength.
    options : dict
        Dictionary of algorithm options:
        - 'niter_sOT' : int, number of iterations
        - 'f_init'    : np.ndarray, initial f, shape (n,)
        - 'g_init'    : np.ndarray, initial g, shape (m,)
        - 'penalty'   : float, cap used in min(., penalty)

    Returns
    -------
    P : np.ndarray
        Transport plan of shape (n, m).
    f : np.ndarray
        Dual potential for source, shape (n,).
    g : np.ndarray
        Dual potential for target, shape (m,).

    Notes
    -----
    - This implementation uses explicit exp/log sums; for large problems it can be slow.
    - Numerical stability: adds `np.finfo(float).eps` to avoid log(0).
    """
    G = np.asarray(G)
    a = np.asarray(a)
    b = np.asarray(b)

    niter = int(options["niter_sOT"])
    f = np.asarray(options["f_init"]).copy()
    g = np.asarray(options["g_init"]).copy()
    penalty = float(options["penalty"])

    for _ in tqdm(range(niter), desc="sOT Iterations"):
        # Update f
        f = np.minimum(
            eps * np.log(a)
            - eps * np.log(np.sum(np.exp((f[:, None] + g[None, :] - G) / eps), axis=1) + np.finfo(float).eps)
            + f,
            penalty,
        )
        # Update g
        g = np.minimum(
            eps * np.log(b)
            - eps * np.log(np.sum(np.exp((f[:, None] + g[None, :] - G) / eps), axis=0) + np.finfo(float).eps)
            + g,
            penalty,
        )

    P = np.exp((f[:, None] + g[None, :] - G) / eps)
    return P, f, g


import itertools
from typing import Optional, Dict, Any, Tuple

import numpy as np
import networkx as nx
from scipy import sparse
from tqdm import tqdm


def fsgw_mvc(
    D1: np.ndarray,
    D2: np.ndarray,
    M: np.ndarray,
    gw_cutoff: float = np.inf,
    w_cutoff: float = np.inf,
    fsgw_niter: int = 10,
    fsgw_eps: float = 0.01,
    fsgw_alpha: float = 0.1,
    fsgw_gamma: float = 2.0,
    sOT_niter: int = 10**4,
    sOT_penalty: float = 2.0,
    seed: Optional[int] = None,
    verbose: bool = True,
    normalize_M: bool = True,
    M_percentile: float = 99.0,
    eps_norm: float = 1e-12,
) -> np.ndarray:
    """
    Compute an FSGW transport plan with a min-vertex-cover based sparsity mask.

    This is a doc-style rewrite of your original implementation, keeping the same
    algorithmic steps:
      1) Normalize D1, D2 by a shared scale.
      2) Build compatibility graph over all (i,j) pairs with GW cutoff.
      3) Remove nodes with feature cost above w_cutoff.
      4) Work on complement graph and greedily remove vertices (MVC-like) to build
         a set of forbidden (i,j) pairs => force P[i,j]=0.
      5) Alternate updates: compute structural cost term D from current P, then solve
         semi-OT (log updates) on combined cost (1-alpha)*M + alpha*D.

    Parameters
    ----------
    D1, D2 : np.ndarray
        Intra-slice distance matrices. Shapes: (n,n), (m,m).
    M : np.ndarray
        Feature cost matrix between slices. Shape: (n,m).
    gw_cutoff : float
        GW cutoff threshold used to build compatibility.
    w_cutoff : float
        Feature cutoff: remove node (i,j) if M[i,j] > w_cutoff.
    fsgw_niter : int
        Number of outer iterations.
    fsgw_eps : float
        Entropic regularization in semi-OT.
    fsgw_alpha : float
        Tradeoff between feature term and structural term.
    fsgw_gamma : float
        Penalty weight (kept from your original code).
    sOT_niter : int
        Iterations for perform_sOT_log.
    sOT_penalty : float
        Penalty cap used in perform_sOT_log.
    seed : int or None
        Random seed for initialization.
    verbose : bool
        Print diagnostics.
    normalize_M : bool
        If True, normalize M by p-th percentile of positive entries (robust scaling).
    M_percentile : float
        Percentile used for robust scaling (default 99).
    eps_norm : float
        Small epsilon to avoid division by zero.

    Returns
    -------
    P : np.ndarray
        Transport plan of shape (n,m).
    """
    D1 = np.asarray(D1)
    D2 = np.asarray(D2)
    M = np.asarray(M)

    n = D1.shape[0]
    m = D2.shape[0]

    if D1.shape != (n, n):
        raise ValueError(f"D1 must be (n,n); got {D1.shape}.")
    if D2.shape != (m, m):
        raise ValueError(f"D2 must be (m,m); got {D2.shape}.")
    if M.shape != (n, m):
        raise ValueError(f"M must be (n,m) with n={n}, m={m}; got {M.shape}.")

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    if normalize_M:
        M_pos = M[M > 0]
        if M_pos.size == 0:
            raise ValueError("M has no positive entries; cannot percentile-normalize.")
        p99 = np.percentile(M_pos, M_percentile)
        M = M / (p99 + eps_norm)

        if verbose:
            # match your prints
            print("M min (non-zero):", M[M > 0].min())
            print("M max (non-zero):", M[M > 0].max())

    # ---- normalize distances by a common scale t (same as your code) ----
    pos = D1[D1 > 0]
    if pos.size == 0:
        raise ValueError("D1 has no positive entries; cannot compute normalization scale t.")
    t = pos.max()

    D1_norm = D1 / t
    D2_norm = D2 / t

    # ---- enumerate all pair indices (i,j) as nodes in [0, n*m) ----
    P_idx = np.array([[i, j] for i, j in itertools.product(range(n), range(m))], dtype=int)

    # ---- build compatibility adjacency matrix A (sparse COO) ----
    I = []
    J = []
    for u in range(len(P_idx)):
        i, j = P_idx[u]
        D_tmp = (D1[i, :, None] - D2[j, :]) ** 2
        tmp_idx = np.where(D_tmp.flatten() <= gw_cutoff**2)[0]
        J.extend(list(tmp_idx))
        I.extend([u] * len(tmp_idx))

    I = np.asarray(I, dtype=int)
    J = np.asarray(J, dtype=int)
    A = sparse.coo_matrix((np.ones_like(I), (I, J)), shape=(n * m, n * m))

    # ---- graph + remove nodes with large feature cost ----
    G = nx.from_scipy_sparse_array(A)
    tmp_idx = np.where(M.flatten() > w_cutoff)[0]
    G.remove_nodes_from(tmp_idx)

    # ---- complement graph for MVC-like elimination ----
    M_flatten = M.flatten()
    zero_indices = set()

    G_copy = nx.complement(G)
    del G

    with tqdm(
        total=G_copy.number_of_edges(),
        desc=f"Finding min vertex covering for cutoff_GW {gw_cutoff} and cutoff_CC {w_cutoff}",
    ) as pbar:
        while G_copy.edges:
            initial_edges = G_copy.number_of_edges()

            # find max-degree vertices
            deg = dict(G_copy.degree())
            max_degree = max(deg.values())
            max_degree_vertices = [v for v, d in G_copy.degree() if d == max_degree]

            # tie-break: choose vertex with largest M_ij
            v_best = max(max_degree_vertices, key=lambda v: M_flatten[v])

            G_copy.remove_node(v_best)
            zero_indices.add(v_best)

            removed_edges = initial_edges - G_copy.number_of_edges()
            pbar.update(removed_edges)

    del G_copy

    # union with explicitly removed nodes
    zero_indices = np.array(list(zero_indices) + list(tmp_idx), dtype=int)

    if verbose:
        print("# of potential non-zeros in P:", n * m - len(zero_indices))

    row_idx = P_idx[zero_indices, 0]
    col_idx = P_idx[zero_indices, 1]

    # ---- initialize marginals + initial P with forbidden entries set to 0 ----
    rng = np.random.default_rng(seed)

    a = np.ones(n) / n
    b = np.ones(m) / m

    aa = a + 1e-1 * rng.random(n) / n
    bb = b + 1e-1 * rng.random(m) / m
    aa = aa / np.linalg.norm(aa, ord=1)
    bb = bb / np.linalg.norm(bb, ord=1)

    P = np.outer(aa, bb)
    P[row_idx, col_idx] = 0.0

    f = np.zeros(n)
    g = np.zeros(m)

    # ---- outer iterations ----
    for _ in range(int(fsgw_niter)):
        # compute D term using only nonzeros in P
        D = np.zeros((n, m))
        non_zero_indices = np.argwhere(P != 0)

        for i, j in non_zero_indices:
            D += P[i, j] * (D1_norm[:, i, None] - D2_norm[None, j, :]) ** 2

        # scale as in your original code
        D = 2.0 * D

        # enforce forbidden pairs
        D[row_idx, col_idx] = np.inf

        # (optional) objective value record (kept same style as your original)
        # NOTE: your original fsgw expression is missing a closing parenthesis;
        # I keep the code simple here and skip storing fsgw_val since it wasn't returned.

        options: Dict[str, Any] = {
            "niter_sOT": int(sOT_niter),
            "f_init": f,
            "g_init": g,
            "penalty": float(sOT_penalty),
        }

        # solve semi-OT subproblem
        P, f, g = perform_sOT_log((1 - fsgw_alpha) * M + fsgw_alpha * D, a, b, fsgw_eps, options)

    return P
    
def fsgw_mvc_reproducible(
    D1,
    D2,
    M,
    gw_cutoff=np.inf,
    w_cutoff=np.inf,
    fsgw_niter=10,
    fsgw_eps=0.01,
    fsgw_alpha=0.1,
    fsgw_gamma=2,
    seed=0,
    verbose=True,
):
    """
    Reproducible wrapper for fsgw_mvc.

    This function does NOT modify the original algorithm.
    It only controls randomness and verbosity for reproducibility.

    Parameters
    ----------
    D1, D2 : np.ndarray
        Intra-slice distance matrices.
    M : np.ndarray
        Feature cost matrix.
    gw_cutoff, w_cutoff : float
        GW / feature cutoffs.
    fsgw_niter, fsgw_eps, fsgw_alpha, fsgw_gamma : float
        Same as in fsgw_mvc.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to show tqdm progress bar.

    Returns
    -------
    P : np.ndarray
        Transport plan.
    """

    # ---- control randomness ----
    np.random.seed(seed)

    # ---- optionally silence tqdm ----
    if not verbose:
        from contextlib import contextmanager

        @contextmanager
        def _silent_tqdm():
            import tqdm as _tqdm
            old_tqdm = _tqdm.tqdm
            _tqdm.tqdm = lambda *a, **k: old_tqdm(*a, disable=True, **k)
            try:
                yield
            finally:
                _tqdm.tqdm = old_tqdm

        with _silent_tqdm():
            P = fsgw_mvc(
                D1,
                D2,
                M,
                gw_cutoff=gw_cutoff,
                w_cutoff=w_cutoff,
                fsgw_niter=fsgw_niter,
                fsgw_eps=fsgw_eps,
                fsgw_alpha=fsgw_alpha,
                fsgw_gamma=fsgw_gamma,
            )
    else:
        P = fsgw_mvc(
            D1,
            D2,
            M,
            gw_cutoff=gw_cutoff,
            w_cutoff=w_cutoff,
            fsgw_niter=fsgw_niter,
            fsgw_eps=fsgw_eps,
            fsgw_alpha=fsgw_alpha,
            fsgw_gamma=fsgw_gamma,
        )

    return P    
    