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


def fsgw_mvc_warm_start(
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
    
def fsgw_mvc_exp(
    D1,
    D2,
    M,
    gw_cutoff=np.inf,
    w_cutoff=np.inf,
    fsgw_niter=10,
    fsgw_eps=0.01,
    fsgw_alpha=0.1,
    fsgw_gamma=2,
    seed=42,                     # ✅ 默认 seed 改为 42
):
    # -----------------------------
    # reproducibility (only this)
    # -----------------------------
    if seed is not None:
        np.random.seed(seed)

    n = D1.shape[0]
    m = D2.shape[0]

    # ---- normalize M (you asked for this) ----
    M = M / M.max()

    t = D1[D1 > 0].max()
    D1_norm = D1 / t
    D2_norm = D2 / t

    P_idx = np.array([[i, j] for i, j in itertools.product(range(n), range(m))], dtype=int)

    I, J = [], []
    for u in range(len(P_idx)):
        i, j = P_idx[u]
        D_tmp = (D1[i, :, None] - D2[j, :]) ** 2
        tmp_idx = np.where(D_tmp.flatten() <= gw_cutoff**2)[0]
        J.extend(tmp_idx.tolist())
        I.extend([u] * len(tmp_idx))

    A = sparse.coo_matrix(
        (np.ones(len(I)), (np.array(I), np.array(J))),
        shape=(n * m, n * m),
    )

    G = nx.from_scipy_sparse_array(A)
    tmp_idx = np.where(M.flatten() > w_cutoff)[0]
    G.remove_nodes_from(tmp_idx)

    M_flatten = M.flatten()
    zero_indices = set()
    G_copy = nx.complement(G)
    del G

    with tqdm(total=G_copy.number_of_edges(),
              desc=f"Finding MVC for gw={gw_cutoff}, w={w_cutoff}") as pbar:
        while G_copy.edges:
            initial_edges = G_copy.number_of_edges()
            vertices, _ = vertex_with_most_edges(G_copy)
            v_best = max(vertices, key=lambda v: M_flatten[v])
            G_copy.remove_node(v_best)
            zero_indices.add(v_best)
            pbar.update(initial_edges - G_copy.number_of_edges())

    zero_indices = np.array(list(zero_indices) + list(tmp_idx), dtype=int)

    row_idx = P_idx[zero_indices, 0]
    col_idx = P_idx[zero_indices, 1]

    a = np.ones(n) / n
    b = np.ones(m) / m
    aa = a + 1e-1 * np.random.rand(n) / n
    bb = b + 1e-1 * np.random.rand(m) / m
    aa /= aa.sum()
    bb /= bb.sum()

    P = np.outer(aa, bb)
    P[row_idx, col_idx] = 0.0

    f = np.zeros(n)
    g = np.zeros(m)

    for _ in range(fsgw_niter):
        D = np.zeros((n, m))
        for i, j in np.argwhere(P != 0):
            D += P[i, j] * (D1_norm[:, i, None] - D2_norm[None, j, :]) ** 2

        D = 2 * D
        D[row_idx, col_idx] = np.inf

        options = {
            "niter_sOT": 10**4,
            "f_init": f,
            "g_init": g,
            "penalty": 2,
        }

        P, f, g = perform_sOT_log(
            (1 - fsgw_alpha) * M + fsgw_alpha * D,
            a, b, fsgw_eps, options
        )

    return P   
    
def fsgw_mvc_relative(
    D1,
    D2,
    M,
    gw_cutoff_ratio = 0.2,
    w_cutoff = np.inf,
    fsgw_niter = 10,
    fsgw_eps = 0.01,
    fsgw_alpha = 0.1,
    fsgw_gamma = 2,
    
):
    n = D1.shape[0]
    m = D2.shape[0]
 
    t = D1[D1 > 0].max()
    D1_norm = D1 / t
    D2_norm = D2 / t
    
    P_idx = np.array( [[i, j] for i, j in itertools.product(range(n), range(m))], dtype=int )
    
    I = []
    J = []
    Q = np.ones_like(D2)
    for i in range(len(P_idx)):
        D_tmp = np.abs(D1[P_idx[i][0],:,np.newaxis] - D2[P_idx[i][1],:])
        D_tmp_min = np.minimum(D1[P_idx[i][0],:,np.newaxis], D2[P_idx[i][1],:])
        tmp_idx = np.where(D_tmp.flatten() <= gw_cutoff_ratio * D_tmp_min.flatten())[0]
        J.extend(list(tmp_idx))
        I.extend([i for _ in range(len(tmp_idx))])
    I = np.array(I, int)
    J = np.array(J, int)
    D = np.ones_like(I)
    A = sparse.coo_matrix((D, (I, J)), shape=(n*m, n*m))
    
    
    G = nx.from_scipy_sparse_array(A)
    tmp_idx = np.where(M.flatten() >  w_cutoff)[0]
    G.remove_nodes_from(tmp_idx)

    
    M_flatten = M.flatten()
    zero_indices = set()
    G_copy = nx.complement(G)
    del G
    with tqdm(total=G_copy.number_of_edges(), desc=f"Finding min vertex covering for gw_cutoff_ratio {gw_cutoff_ratio} and cutoff_CC {w_cutoff}") as pbar:
        while G_copy.edges:
            initial_edges = G_copy.number_of_edges()
            max_degree_vertices, max_degree = vertex_with_most_edges(G_copy)
            max_Cij_value = -float('inf')
            vertex_with_max_Cij = None
            for vertex in max_degree_vertices:
                if M_flatten[vertex] > max_Cij_value:
                    max_Cij_value = M_flatten[vertex]
                    vertex_with_max_Cij = vertex
    
            G_copy.remove_node(vertex_with_max_Cij)
            zero_indices.add(vertex_with_max_Cij)
            removed_edges = initial_edges - G_copy.number_of_edges()
            pbar.update(removed_edges)
    del G_copy
    zero_indices = list(zero_indices) + list(tmp_idx)
    zero_indices = np.array(zero_indices)
    print("# of potential non-zeros in P:", n*m - len(zero_indices))

    
    row_idx = P_idx[zero_indices,0]
    col_idx = P_idx[zero_indices,1]

    a = np.ones(n) / n
    b = np.ones(m) / m
    aa = a + 1e-1 * np.random.rand(n) / n
    bb = b + 1e-1 * np.random.rand(m) / m

    aa = aa / np.linalg.norm(aa, ord=1)
    bb = bb / np.linalg.norm(bb, ord=1)

    P = np.outer(aa, bb)
    P[row_idx, col_idx] = 0
    f = np.zeros(n)
    g = np.zeros(m)

    fsgw_val = []

    for p in range(fsgw_niter):

        D = np.zeros((n, m)) 
        non_zero_indices = np.argwhere(P != 0)
        for i, j in non_zero_indices:
            # Compute the contribution for each non-zero entry of P
            D += P[i, j] * (D1_norm[:, i, None] - D2_norm[None, j, :])**2
        fsgw = (1 - fsgw_alpha) * np.sum(M * P) + fsgw_alpha*(np.sum(D * P)
            +fsgw_gamma*(np.sum(a)+np.sum(b)-2*np.sum(P))+fsgw_eps*np.sum(P * (np.log(P+10**(-20)*np.ones((n,m)))-np.ones((n,m)))))
        fsgw_val.append(fsgw)

        D = 2*D

        D[row_idx, col_idx] = np.inf

        options = {
            'niter_sOT': 10**4,
            'f_init': np.zeros(n),
            'g_init': np.zeros(m),
            'penalty': 2
        }

        P, f, g = perform_sOT_log( (1 - fsgw_alpha)* M + fsgw_alpha*D, a, b, fsgw_eps, options)

    return P    
     

import itertools
import numpy as np
from scipy import sparse
from tqdm import tqdm


def fsgw_mvc_exp_sparse(
    D1,
    D2,
    M,
    gw_cutoff=np.inf,
    w_cutoff=np.inf,
    fsgw_niter=10,
    fsgw_eps=0.01,
    fsgw_alpha=0.1,
    fsgw_gamma=2,
    seed=42,
    show_progress=True,
    return_log=False,
):
    """
    Memory-efficient version of fsgw_mvc_exp.

    Main replacement:
        Original:
            build compatibility graph G
            build explicit complement graph nx.complement(G)
            greedy MVC on complement graph

        New:
            build compatibility matrix A as COO
            convert A to CSR
            use active mask
            repeatedly remove min-degree vertex in current compatibility graph

    Equivalence:
        In the current active set S,

            deg_{complement(G[S])}(v) = |S| - 1 - deg_{G[S]}(v)

        Therefore, choosing the max-degree vertex in the complement graph
        is equivalent to choosing the min-degree vertex in the compatibility graph.

    Tie-break:
        Same as the original code:
            among equal-degree candidates, remove the one with largest M_flatten[v].
    """

    # -----------------------------
    # reproducibility
    # -----------------------------
    if seed is not None:
        np.random.seed(seed)

    n = D1.shape[0]
    m = D2.shape[0]
    N_pair = n * m

    # -----------------------------
    # normalize M
    # -----------------------------
    M = np.asarray(M, dtype=float)
    M_max = M.max()
    if M_max <= 0:
        raise ValueError("M.max() must be positive for normalization.")
    M = M / M_max

    M_flatten = M.flatten()

    # -----------------------------
    # normalize D1, D2
    # -----------------------------
    positive_D1 = D1[D1 > 0]
    if len(positive_D1) == 0:
        raise ValueError("D1 has no positive entries, cannot normalize by D1[D1 > 0].max().")

    t = positive_D1.max()
    D1_norm = D1 / t
    D2_norm = D2 / t

    # -----------------------------
    # flattened index mapping
    # u = i * m + j
    # P_idx[u] = [i, j]
    # -----------------------------
    P_idx = np.array(
        [[i, j] for i, j in itertools.product(range(n), range(m))],
        dtype=int
    )

    # ============================================================
    # 1. Build compatibility matrix A as COO
    # ============================================================
    I, J = [], []

    iterator = range(N_pair)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=N_pair,
            desc=f"Building COO compatibility A, gw={gw_cutoff}",
            leave=True
        )

    for u in iterator:
        i, j = P_idx[u]

        # D_tmp[i', j'] = (D1[i, i'] - D2[j, j'])^2
        D_tmp = (D1[i, :, None] - D2[j, :]) ** 2

        # compatible vertices v = (i', j')
        tmp_compatible_idx = np.where(D_tmp.flatten() <= gw_cutoff**2)[0]

        J.extend(tmp_compatible_idx.tolist())
        I.extend([u] * len(tmp_compatible_idx))

    A_coo = sparse.coo_matrix(
        (np.ones(len(I), dtype=bool), (np.array(I), np.array(J))),
        shape=(N_pair, N_pair),
        dtype=bool
    )

    # ============================================================
    # 2. Convert COO -> CSR
    # ============================================================
    A = A_coo.tocsr().astype(bool)

    # Important:
    # NetworkX complement graph does not use self-loops.
    # Your COO construction includes A[u, u] = 1, so remove diagonal
    # to match the original complement-graph degree behavior.
    A.setdiag(False)
    A.eliminate_zeros()

    # ============================================================
    # 3. Feature cutoff: M > w_cutoff
    # ============================================================
    tmp_idx = np.where(M_flatten > w_cutoff)[0]

    active = np.ones(N_pair, dtype=bool)
    active[tmp_idx] = False

    zero_indices = set()

    # ============================================================
    # 4. Implicit complement greedy MVC
    # ============================================================
    initial_active = int(active.sum())

    if show_progress:
        pbar = tqdm(
            total=initial_active,
            desc=f"Finding MVC implicitly, gw={gw_cutoff}, w={w_cutoff}",
            leave=True
        )
    else:
        pbar = None

    while True:
        active_idx = np.flatnonzero(active)
        s = len(active_idx)

        if s <= 1:
            break

        # degree in current active induced compatibility graph A[S, S]
        A_sub = A[active_idx, :][:, active_idx]
        deg_compat = np.asarray(A_sub.sum(axis=1)).ravel()

        min_deg = deg_compat.min()

        # Stop iff current active compatibility graph is a clique.
        #
        # In a clique with s vertices, every vertex has degree s - 1.
        # This is equivalent to complement graph having no edges.
        if min_deg == s - 1:
            break

        # Equivalent to max-degree vertices in complement graph.
        candidates = active_idx[np.flatnonzero(deg_compat == min_deg)]

        # Same tie-break as original:
        # among equal-degree candidates, remove larger M cost.
        v_best = max(candidates, key=lambda v: M_flatten[v])

        active[v_best] = False
        zero_indices.add(int(v_best))

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                active=s - 1,
                min_deg=int(min_deg),
                deleted=len(zero_indices)
            )

    if pbar is not None:
        pbar.close()

    # Same final construction as original:
    # zero_indices from MVC + feature cutoff tmp_idx.
    zero_indices = np.array(list(zero_indices) + list(tmp_idx), dtype=int)

    row_idx = P_idx[zero_indices, 0]
    col_idx = P_idx[zero_indices, 1]

    # ============================================================
    # 5. Initialize P
    # ============================================================
    a = np.ones(n) / n
    b = np.ones(m) / m

    aa = a + 1e-1 * np.random.rand(n) / n
    bb = b + 1e-1 * np.random.rand(m) / m

    aa /= aa.sum()
    bb /= bb.sum()

    P = np.outer(aa, bb)
    P[row_idx, col_idx] = 0.0

    # ============================================================
    # 6. FSGW iterations
    # ============================================================
    f = np.zeros(n)
    g = np.zeros(m)

    for _ in tqdm(
        range(fsgw_niter),
        desc="FSGW iterations",
        leave=True,
        disable=not show_progress
    ):
        D = np.zeros((n, m))

        for i, j in np.argwhere(P != 0):
            D += P[i, j] * (D1_norm[:, i, None] - D2_norm[None, j, :]) ** 2

        D = 2 * D
        D[row_idx, col_idx] = np.inf

        options = {
            "niter_sOT": 10**4,
            "f_init": f,
            "g_init": g,
            "penalty": 2,
        }

        P, f, g = perform_sOT_log(
            (1 - fsgw_alpha) * M + fsgw_alpha * D,
            a,
            b,
            fsgw_eps,
            options
        )

    if return_log:
        log = {
            "zero_indices": zero_indices,
            "row_idx": row_idx,
            "col_idx": col_idx,
            "num_zero_indices": len(set(zero_indices.tolist())),
            "num_feature_cutoff": len(tmp_idx),
            "num_mvc_deleted": len(set(zero_indices.tolist())) - len(set(tmp_idx.tolist())),
            "num_final_active_after_mvc": int(active.sum()),
            "A_nnz": int(A.nnz),
            "N_pair": int(N_pair),
            "n": int(n),
            "m": int(m),
            "gw_cutoff": gw_cutoff,
            "w_cutoff": w_cutoff,
        }
        return P, log

    return P


import itertools
import time
import numpy as np
from scipy import sparse
from tqdm import tqdm


def fsgw_mvc_exp_sparse_incremental(
    D1,
    D2,
    M,
    gw_cutoff=np.inf,
    w_cutoff=np.inf,
    fsgw_niter=10,
    fsgw_eps=0.01,
    fsgw_alpha=0.1,
    fsgw_gamma=2,
    seed=42,
    show_progress=True,
    return_log=False,
):
    """
    Memory-efficient + faster incremental-degree version of fsgw_mvc_exp_sparse.

    Main idea
    ---------
    Original method:
        Build compatibility graph G.
        Build complement graph G_copy.
        Greedy MVC on G_copy by repeatedly removing max-degree vertex.

    New method:
        Build sparse compatibility matrix A.
        Do not build complement graph.
        On current active set S:

            deg_complement(v) = |S| - 1 - deg_compat(v)

        Therefore:
            max-degree in complement graph
            =
            min-degree in compatibility graph.

    Incremental degree update
    -------------------------
    Instead of recomputing

        A[active_idx, :][:, active_idx].sum(axis=1)

    at every iteration, we compute active degrees once, and when vertex v is
    removed, we decrement the degree of its active neighbors by 1.

    Tie-break
    ---------
    Same as original code:

        v_best = max(candidates, key=lambda v: M_flatten[v])

    i.e. among equal-degree candidates, remove the one with largest feature cost.
    """

    stage_times = {}
    t_total_start = time.perf_counter()

    # -----------------------------
    # reproducibility
    # -----------------------------
    if seed is not None:
        np.random.seed(seed)

    n = D1.shape[0]
    m = D2.shape[0]
    N_pair = n * m

    # ============================================================
    # 0. Normalize M, D1, D2
    # ============================================================
    t0 = time.perf_counter()

    M = np.asarray(M, dtype=float)
    M_max = M.max()
    if M_max <= 0:
        raise ValueError("M.max() must be positive for normalization.")
    M = M / M_max
    M_flatten = M.flatten()

    positive_D1 = D1[D1 > 0]
    if len(positive_D1) == 0:
        raise ValueError("D1 has no positive entries, cannot normalize by D1[D1 > 0].max().")

    t = positive_D1.max()
    D1_norm = D1 / t
    D2_norm = D2 / t

    stage_times["normalize_M_D"] = time.perf_counter() - t0

    # -----------------------------
    # flattened index mapping
    # u = i * m + j
    # P_idx[u] = [i, j]
    # -----------------------------
    t0 = time.perf_counter()

    P_idx = np.array(
        [[i, j] for i, j in itertools.product(range(n), range(m))],
        dtype=int
    )

    stage_times["build_P_idx"] = time.perf_counter() - t0

    # ============================================================
    # 1. Build compatibility matrix A as COO
    # ============================================================
    t0 = time.perf_counter()

    I, J = [], []

    iterator = range(N_pair)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=N_pair,
            desc=f"Building COO compatibility A, gw={gw_cutoff}",
            leave=True
        )

    for u in iterator:
        i, j = P_idx[u]

        # D_tmp[i', j'] = (D1[i, i'] - D2[j, j'])^2
        D_tmp = (D1[i, :, None] - D2[j, :]) ** 2

        # compatible vertices v = (i', j')
        tmp_compatible_idx = np.where(D_tmp.flatten() <= gw_cutoff**2)[0]

        J.extend(tmp_compatible_idx.tolist())
        I.extend([u] * len(tmp_compatible_idx))

    A_coo = sparse.coo_matrix(
        (np.ones(len(I), dtype=bool), (np.array(I), np.array(J))),
        shape=(N_pair, N_pair),
        dtype=bool
    )

    stage_times["build_A_coo"] = time.perf_counter() - t0

    # ============================================================
    # 2. Convert COO -> CSR
    # ============================================================
    t0 = time.perf_counter()

    A = A_coo.tocsr().astype(bool)

    # Important:
    # NetworkX complement graph does not use self-loops.
    # Your COO construction includes A[u, u] = 1, so remove diagonal.
    A.setdiag(False)
    A.eliminate_zeros()

    stage_times["coo_to_csr_remove_diag"] = time.perf_counter() - t0

    # ============================================================
    # 3. Feature cutoff: M > w_cutoff
    # ============================================================
    t0 = time.perf_counter()

    tmp_idx = np.where(M_flatten > w_cutoff)[0]

    active = np.ones(N_pair, dtype=bool)
    active[tmp_idx] = False

    zero_indices = set()

    initial_active = int(active.sum())

    stage_times["feature_cutoff_active_init"] = time.perf_counter() - t0

    # ============================================================
    # 4. Initial degree in active compatibility graph
    # ============================================================
    t0 = time.perf_counter()

    active_idx = np.flatnonzero(active)

    deg_compat = np.zeros(N_pair, dtype=np.int64)

    if len(active_idx) > 0:
        # Initial degree in A[S, S].
        deg_compat[active_idx] = np.asarray(
            A[active_idx, :][:, active_idx].sum(axis=1)
        ).ravel()

    stage_times["initial_active_degree"] = time.perf_counter() - t0

    # ============================================================
    # 5. Implicit complement greedy MVC with incremental degree update
    # ============================================================
    t0 = time.perf_counter()

    if show_progress:
        pbar = tqdm(
            total=initial_active,
            desc=f"Finding MVC incrementally, gw={gw_cutoff}, w={w_cutoff}",
            leave=True
        )
    else:
        pbar = None

    s = initial_active

    while True:
        if s <= 1:
            break

        # Only active vertices participate.
        active_idx = np.flatnonzero(active)

        # Minimum compatibility degree among active vertices.
        min_deg = deg_compat[active_idx].min()

        # Stop iff active compatibility graph is a clique.
        #
        # In a clique of size s, every active vertex has degree s - 1.
        # Equivalent to complement graph having no edges.
        if min_deg == s - 1:
            break

        candidates = active_idx[deg_compat[active_idx] == min_deg]

        # Same tie-break as original:
        # among equal-degree candidates, remove larger M cost.
        v_best = max(candidates, key=lambda v: M_flatten[v])

        # Get active neighbors of v_best BEFORE deleting it.
        row_start = A.indptr[v_best]
        row_end = A.indptr[v_best + 1]
        neighbors = A.indices[row_start:row_end]

        active_neighbors = neighbors[active[neighbors]]

        # Delete v_best.
        active[v_best] = False
        zero_indices.add(int(v_best))

        # Incremental degree update:
        # Removing v_best decreases the compatibility degree of its
        # active neighbors by 1.
        deg_compat[active_neighbors] -= 1

        # v_best no longer participates.
        deg_compat[v_best] = -1

        s -= 1

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                active=s,
                min_deg=int(min_deg),
                deleted=len(zero_indices)
            )

    if pbar is not None:
        pbar.close()

    stage_times["mvc_incremental"] = time.perf_counter() - t0

    # Same final construction as original:
    # zero_indices from MVC + feature cutoff tmp_idx.
    t0 = time.perf_counter()

    zero_indices = np.array(list(zero_indices) + list(tmp_idx), dtype=int)

    row_idx = P_idx[zero_indices, 0]
    col_idx = P_idx[zero_indices, 1]

    stage_times["zero_indices_to_row_col"] = time.perf_counter() - t0

    # ============================================================
    # 6. Initialize P
    # ============================================================
    t0 = time.perf_counter()

    a = np.ones(n) / n
    b = np.ones(m) / m

    aa = a + 1e-1 * np.random.rand(n) / n
    bb = b + 1e-1 * np.random.rand(m) / m

    aa /= aa.sum()
    bb /= bb.sum()

    P = np.outer(aa, bb)
    P[row_idx, col_idx] = 0.0

    stage_times["initialize_P"] = time.perf_counter() - t0

    # ============================================================
    # 7. FSGW iterations
    # ============================================================
    # NOTE:
    # This part is intentionally unchanged from your provided function.
    # ============================================================
    t0 = time.perf_counter()

    f = np.zeros(n)
    g = np.zeros(m)

    for _ in tqdm(
        range(fsgw_niter),
        desc="FSGW iterations",
        leave=True,
        disable=not show_progress
    ):
        D = np.zeros((n, m))

        for i, j in np.argwhere(P != 0):
            D += P[i, j] * (D1_norm[:, i, None] - D2_norm[None, j, :]) ** 2

        D = 2 * D
        D[row_idx, col_idx] = np.inf

        options = {
            "niter_sOT": 10**4,
            "f_init": f,
            "g_init": g,
            "penalty": 2,
        }

        P, f, g = perform_sOT_log(
            (1 - fsgw_alpha) * M + fsgw_alpha * D,
            a,
            b,
            fsgw_eps,
            options
        )

    stage_times["fsgw_iterations"] = time.perf_counter() - t0

    stage_times["total"] = time.perf_counter() - t_total_start

    if return_log:
        tmp_idx_set = set(tmp_idx.tolist())
        zero_set = set(zero_indices.tolist())

        log = {
            "zero_indices": zero_indices,
            "row_idx": row_idx,
            "col_idx": col_idx,
            "num_zero_indices": len(zero_set),
            "num_feature_cutoff": len(tmp_idx),
            "num_mvc_deleted": len(zero_set) - len(tmp_idx_set),
            "num_final_active_after_mvc": int(active.sum()),
            "A_nnz": int(A.nnz),
            "N_pair": int(N_pair),
            "n": int(n),
            "m": int(m),
            "gw_cutoff": gw_cutoff,
            "w_cutoff": w_cutoff,
            "stage_times": stage_times,
        }
        return P, log

    return P


import itertools
import time
import heapq
import numpy as np
from scipy import sparse
from tqdm import tqdm


def fsgw_mvc_exp_sparse_heap(
    D1,
    D2,
    M,
    gw_cutoff=np.inf,
    w_cutoff=np.inf,
    fsgw_niter=10,
    fsgw_eps=0.01,
    fsgw_alpha=0.1,
    fsgw_gamma=2,
    seed=42,
    show_progress=True,
    return_log=False,
):
    """
    Sparse + heap optimized version of fsgw_mvc_exp.

    Optimizations
    -------------
    1. Apply M cutoff before building A:
        vertices with M_flatten[v] > w_cutoff are invalid from the start.
        We do not store edges incident to those vertices.

    2. Use heap-based incremental min-degree peeling:
        Original complement-graph greedy MVC chooses

            argmax_v deg_{complement(G[S])}(v)

        Since

            deg_{complement(G[S])}(v) = |S| - 1 - deg_{G[S]}(v),

        this is equivalent to choosing

            argmin_v deg_{G[S]}(v)

        in the compatibility graph.

        Tie-break is preserved:
            among equal-degree vertices, remove the one with largest M_flatten[v].

    Important
    ---------
    The FSGW iteration part is kept unchanged from your original function.
    """

    stage_times = {}
    t_total_start = time.perf_counter()

    # -----------------------------
    # reproducibility
    # -----------------------------
    if seed is not None:
        np.random.seed(seed)

    n = D1.shape[0]
    m = D2.shape[0]
    N_pair = n * m

    # ============================================================
    # 0. Normalize M, D1, D2
    # ============================================================
    t0 = time.perf_counter()

    M = np.asarray(M, dtype=float)
    M_max = M.max()
    if M_max <= 0:
        raise ValueError("M.max() must be positive for normalization.")

    M = M / M_max
    M_flatten = M.flatten()

    positive_D1 = D1[D1 > 0]
    if len(positive_D1) == 0:
        raise ValueError(
            "D1 has no positive entries, cannot normalize by D1[D1 > 0].max()."
        )

    t = positive_D1.max()
    D1_norm = D1 / t
    D2_norm = D2 / t

    stage_times["normalize_M_D"] = time.perf_counter() - t0

    # ============================================================
    # 1. M cutoff before graph construction
    # ============================================================
    t0 = time.perf_counter()

    valid_by_M = M_flatten <= w_cutoff
    tmp_idx = np.where(~valid_by_M)[0]

    num_valid_by_M = int(valid_by_M.sum())
    num_blocked_by_M = int(len(tmp_idx))

    stage_times["precompute_M_cutoff"] = time.perf_counter() - t0

    if show_progress:
        print("Total pair vertices:", N_pair)
        print("Valid by M cutoff:", num_valid_by_M)
        print("Blocked by M cutoff:", num_blocked_by_M)
        print("Valid fraction:", num_valid_by_M / N_pair)

    # ============================================================
    # 2. Build compatibility matrix A as COO
    #    Shape remains (N_pair, N_pair), but edges involving invalid
    #    M-cutoff vertices are not stored.
    # ============================================================
    t0 = time.perf_counter()

    I, J = [], []

    iterator = range(N_pair)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=N_pair,
            desc=f"Building COO A with M cutoff, gw={gw_cutoff}, w={w_cutoff}",
            leave=True
        )

    for u in iterator:
        # If this vertex is already blocked by M cutoff,
        # it should not participate in the graph.
        if not valid_by_M[u]:
            continue

        i = u // m
        j = u % m

        # D_tmp[i', j'] = (D1[i, i'] - D2[j, j'])^2
        D_tmp = (D1[i, :, None] - D2[j, :]) ** 2

        # compatible vertices v = (i', j')
        tmp_compatible_idx = np.where(D_tmp.ravel() <= gw_cutoff**2)[0]

        # Keep only vertices that also pass M cutoff.
        tmp_compatible_idx = tmp_compatible_idx[valid_by_M[tmp_compatible_idx]]

        J.extend(tmp_compatible_idx.tolist())
        I.extend([u] * len(tmp_compatible_idx))

    A_coo = sparse.coo_matrix(
        (np.ones(len(I), dtype=bool), (np.asarray(I), np.asarray(J))),
        shape=(N_pair, N_pair),
        dtype=bool
    )

    stage_times["build_A_coo_with_M_cutoff"] = time.perf_counter() - t0

    # ============================================================
    # 3. Convert COO -> CSR and remove self-loops
    # ============================================================
    t0 = time.perf_counter()

    A = A_coo.tocsr().astype(bool)

    # NetworkX complement does not use self-loops.
    # The COO construction includes A[u, u] = 1 for valid vertices,
    # so remove diagonal to match original behavior.
    A.setdiag(False)
    A.eliminate_zeros()

    stage_times["coo_to_csr_remove_diag"] = time.perf_counter() - t0

    # ============================================================
    # 4. Initialize active mask and initial active degree
    # ============================================================
    t0 = time.perf_counter()

    active = valid_by_M.copy()
    zero_indices = set()

    active_idx = np.flatnonzero(active)
    initial_active = int(len(active_idx))

    deg_compat = np.zeros(N_pair, dtype=np.int64)

    if initial_active > 0:
        deg_compat[active_idx] = np.asarray(
            A[active_idx, :][:, active_idx].sum(axis=1)
        ).ravel()

    stage_times["initial_active_degree"] = time.perf_counter() - t0

    # ============================================================
    # 5. Heap-based implicit complement greedy MVC
    # ============================================================
    t0 = time.perf_counter()

    heap = []

    # Heap key:
    #   1. smaller compatibility degree first
    #   2. larger M cost first via -M_flatten[v]
    #   3. smaller vertex index first if both tie
    for v in active_idx:
        heapq.heappush(
            heap,
            (int(deg_compat[v]), -float(M_flatten[v]), int(v))
        )

    s = initial_active

    if show_progress:
        pbar = tqdm(
            total=initial_active,
            desc=f"Finding MVC by heap, gw={gw_cutoff}, w={w_cutoff}",
            leave=True
        )
    else:
        pbar = None

    while True:
        if s <= 1:
            break

        # Pop valid, non-stale heap entry.
        #
        # Stale entries appear because when a neighbor degree decreases,
        # we push a new heap entry instead of deleting the old one.
        while heap:
            deg_v, neg_m_v, v_best = heapq.heappop(heap)

            if active[v_best] and deg_v == deg_compat[v_best]:
                break
        else:
            # Heap exhausted.
            break

        # Stop iff active compatibility graph is a clique.
        #
        # In a clique with s vertices, every active vertex has degree s - 1.
        # Since heap gives the minimum active degree, deg_v == s - 1 means
        # all active vertices have degree s - 1.
        if deg_v == s - 1:
            break

        # Get active neighbors before deleting v_best.
        row_start = A.indptr[v_best]
        row_end = A.indptr[v_best + 1]
        neighbors = A.indices[row_start:row_end]

        active_neighbors = neighbors[active[neighbors]]

        # Delete v_best.
        active[v_best] = False
        zero_indices.add(int(v_best))

        # Incrementally update degrees of active neighbors.
        deg_compat[active_neighbors] -= 1

        # v_best no longer participates.
        deg_compat[v_best] = -1

        s -= 1

        # Push updated active neighbors into heap.
        # Old heap entries for these neighbors become stale.
        for u in active_neighbors:
            heapq.heappush(
                heap,
                (int(deg_compat[u]), -float(M_flatten[u]), int(u))
            )

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                active=s,
                min_deg=int(deg_v),
                deleted=len(zero_indices),
                heap=len(heap)
            )

    if pbar is not None:
        pbar.close()

    stage_times["mvc_heap"] = time.perf_counter() - t0

    # ============================================================
    # 6. Convert zero indices to row/column indices
    # ============================================================
    t0 = time.perf_counter()

    # Same final construction as original:
    # zero_indices from MVC + feature cutoff tmp_idx.
    zero_indices = np.array(list(zero_indices) + list(tmp_idx), dtype=int)

    # Equivalent to:
    #   row_idx = P_idx[zero_indices, 0]
    #   col_idx = P_idx[zero_indices, 1]
    # but avoids storing P_idx.
    row_idx = zero_indices // m
    col_idx = zero_indices % m

    stage_times["zero_indices_to_row_col"] = time.perf_counter() - t0

    # ============================================================
    # 7. Initialize P
    # ============================================================
    t0 = time.perf_counter()

    a = np.ones(n) / n
    b = np.ones(m) / m

    aa = a + 1e-1 * np.random.rand(n) / n
    bb = b + 1e-1 * np.random.rand(m) / m

    aa /= aa.sum()
    bb /= bb.sum()

    P = np.outer(aa, bb)
    P[row_idx, col_idx] = 0.0

    stage_times["initialize_P"] = time.perf_counter() - t0

    # ============================================================
    # 8. FSGW iterations
    #    This part is unchanged from your original function.
    # ============================================================
    t0 = time.perf_counter()

    f = np.zeros(n)
    g = np.zeros(m)

    for _ in tqdm(
        range(fsgw_niter),
        desc="FSGW iterations",
        leave=True,
        disable=not show_progress
    ):
        D = np.zeros((n, m))

        for i, j in np.argwhere(P != 0):
            D += P[i, j] * (D1_norm[:, i, None] - D2_norm[None, j, :]) ** 2

        D = 2 * D
        D[row_idx, col_idx] = np.inf

        options = {
            "niter_sOT": 10**4,
            "f_init": f,
            "g_init": g,
            "penalty": 2,
        }

        P, f, g = perform_sOT_log(
            (1 - fsgw_alpha) * M + fsgw_alpha * D,
            a,
            b,
            fsgw_eps,
            options
        )

    stage_times["fsgw_iterations"] = time.perf_counter() - t0

    stage_times["total"] = time.perf_counter() - t_total_start

    # ============================================================
    # 9. Return
    # ============================================================
    if return_log:
        zero_set = set(zero_indices.tolist())
        tmp_idx_set = set(tmp_idx.tolist())

        log = {
            "zero_indices": zero_indices,
            "row_idx": row_idx,
            "col_idx": col_idx,
            "num_zero_indices": len(zero_set),
            "num_feature_cutoff": len(tmp_idx),
            "num_mvc_deleted": len(zero_set) - len(tmp_idx_set),
            "num_final_active_after_mvc": int(active.sum()),
            "A_nnz": int(A.nnz),
            "N_pair": int(N_pair),
            "n": int(n),
            "m": int(m),
            "gw_cutoff": gw_cutoff,
            "w_cutoff": w_cutoff,
            "num_valid_by_M": num_valid_by_M,
            "num_blocked_by_M": num_blocked_by_M,
            "valid_fraction_by_M": num_valid_by_M / N_pair,
            "stage_times": stage_times,
        }

        return P, log

    return P