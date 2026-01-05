#recover full mapping

import numpy as np
import pandas as pd
from scipy import sparse

def matrix_rescaling_checking(C_norm, verbose=True, atol=1e-12):
    min_val = C_norm.min()
    max_val = C_norm.max()
    if not (min_val >= -atol and max_val <= 1.0 + atol):
        raise AssertionError("Rescaling failed: data not in [0,1] interval.")

    if not np.isclose(max_val, 1.0, atol=atol):
        raise AssertionError("Rescaling failed: max is not close to 1.")

    if verbose:
        print(f"After rescaling: min = {min_val}, max = {max_val}.")

def generate_binary_matching(sliceA, downsampled_sliceA, sliceB, downsampled_sliceB, P, cutoff_GW, cutoff_CC, output_dir):
    """
    Generate a binary matching matrix L_sig and derive L_full for alignment.
    
    Parameters:
    - sliceA: AnnData, original slice A object.
    - downsampled_sliceA: AnnData, downsampled version of slice A.
    - sliceB: AnnData, original slice B object.
    - downsampled_sliceB: AnnData, downsampled version of slice B.
    - P: np.ndarray, probabilistic matching matrix.
    - cutoff_GW: float, cutoff value for fused supervised Gromov-Wasserstein alignment.
    - cutoff_CC: float, cutoff value for other matching criteria.
    - output_dir: str, directory to save intermediate and final outputs.
    
    Returns:
    - L_sig: np.ndarray, the downsampled binary matching matrix.
    - L_full: np.ndarray, the full binary matching matrix.
    - indices_A_matching_part
    - indices_B_matching_part
    """
    # Step 1: Generate binary matching matrix L_sig
    K = np.where(P > 1e-6, P, 0)
    L = np.zeros_like(K)
    for i in range(K.shape[0]):
        if np.any(K[i, :] > 0):
            max_j = np.argmax(K[i, :])
            L[i, max_j] = 1

    idx_1 = np.where(L.sum(axis=1) > 0)[0]
    idx_2 = np.where(L.sum(axis=0) > 0)[0]
    L_sig = L[idx_1, :][:, idx_2]

    # Step 2: Extract aligned downsampled slices
    downsampled_sliceA_fsgw_matching_part = downsampled_sliceA[idx_1].copy()
    downsampled_sliceB_fsgw_matching_part = downsampled_sliceB[idx_2].copy()

    # # Step 3: Save spatial coordinates
    # pd.DataFrame(sliceA.obsm['spatial']).to_csv(f'{output_dir}/sliceA_spatial.csv', index=False, header=False)
    # pd.DataFrame(sliceB.obsm['spatial']).to_csv(f'{output_dir}/sliceB_spatial.csv', index=False, header=False)

    # Step 4: Generate and save indices for matching parts
    indices_A_matching_part = [
        np.where(sliceA.obs_names == downsampled_sliceA_fsgw_matching_part.obs_names[i])[0][0]
        for i in range(downsampled_sliceA_fsgw_matching_part.shape[0])
    ]
    indices_B_matching_part = [
        np.where(sliceB.obs_names == downsampled_sliceB_fsgw_matching_part.obs_names[i])[0][0]
        for i in range(downsampled_sliceB_fsgw_matching_part.shape[0])
    ]

    # pd.DataFrame(indices_A_matching_part, columns=['Indices']).to_csv(
    #     f'{output_dir}/A_matching_part_{cutoff_GW}_{cutoff_CC}.csv', index=False)
    # pd.DataFrame(indices_B_matching_part, columns=['Indices']).to_csv(
    #     f'{output_dir}/B_matching_part_{cutoff_GW}_{cutoff_CC}.csv', index=False)

    # # Step 5: Save L_sig
    # pd.DataFrame(L_sig).to_csv(
    #     f'{output_dir}/L_sig_{cutoff_GW}_{cutoff_CC}.csv', index=False, header=False)

    # Step 6: Derive L_full
    s, t = sliceA.shape[0], sliceB.shape[0]
    L_full = np.zeros((s, t))
    for i in range(len(indices_A_matching_part)):
        for j in range(len(indices_B_matching_part)):
            L_full[indices_A_matching_part[i], indices_B_matching_part[j]] = L_sig[i, j]

    return L_sig, L_full, indices_A_matching_part, indices_B_matching_part

import numpy as np
from scipy.spatial import distance_matrix

# GW cost in recovering stage
def compute_gw_recover(X1, X2, P, idx1, idx2, delta=0.5, p_cost=2):
    D1 = distance_matrix(X1, X1[idx1, :])
    D2 = distance_matrix(X2, X2[idx2, :])
    
    tmp_idx1 = np.where(P.sum(axis=1) >= delta / D1.shape[1])[0]
    tmp_idx2 = np.where(P.sum(axis=0) >= delta / D2.shape[1])[0]
    
    P_sig = P[tmp_idx1, :][:, tmp_idx2]
    P_sig_1_to_2 = P_sig / P_sig.sum(axis=1, keepdims=True)
    P_sig_2_to_1 = (P_sig / P_sig.sum(axis=0, keepdims=True)).T
    
    D1_to_D2 = np.matmul(D1[:, tmp_idx1], P_sig_1_to_2)
    D2_to_D1 = np.matmul(D2[:, tmp_idx2], P_sig_2_to_1)
    
    C1 = distance_matrix(D1[:, tmp_idx1], D2_to_D1, p=p_cost)
    C2 = distance_matrix(D1_to_D2, D2[:, tmp_idx2], p=p_cost)
    
    C = 0.5 * (C1 + C2)
    
    return C 

#################################################
    #old version
#################################################

# # sot
# def sot_sinkhorn_l1_sparse(a,b,C,eps,m,nitermax=10000,stopthr=1e-8,verbose=False):
#     """ Solve the unnormalized optimal transport with l1 penalty in sparse matrix format.

#     Parameters
#     ----------
#     a : (ns,) numpy.ndarray
#         Source distribution. The summation should be less than or equal to 1.
#     b : (nt,) numpy.ndarray
#         Target distribution. The summation should be less than or equal to 1.
#     C : (ns,nt) scipy.sparse.coo_matrix
#         The cost matrix in coo sparse format. The entries exceeds the cost cutoff are omitted. The naturally zero entries should be explicitely included.
#     eps : float
#         The coefficient for entropy regularization.
#     m : float
#         The coefficient for penalizing unmatched mass.
#     nitermax : int, optional
#         The max number of iterations. Defaults to 10000.
#     stopthr : float, optional
#         The threshold for terminating the iteration. Defaults to 1e-8.

#     Returns
#     -------
#     (ns,nt) scipy.sparse.coo_matrix
#         The optimal transport matrix. The locations of entries should agree with C and there might by explicit zero entries.
#     """
#     tmp_K = C.copy()
#     f = np.zeros_like(a)
#     g = np.zeros_like(b)
#     r = np.zeros_like(a)
#     s = np.zeros_like(b)
#     niter = 0
#     err = 100
#     while niter <= nitermax and err > stopthr:
#         fprev = f
#         gprev = g
#         # Iteration
#         tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
#         f = eps * np.log(a) \
#             - eps * np.log( np.sum( tmp_K, axis=1 ).A.reshape(-1) \
#             + np.exp( ( -m + f ) / eps ) ) + f
#         tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
#         g = eps * np.log(b) \
#             - eps * np.log( np.sum( tmp_K, axis=0 ).A.reshape(-1) \
#             + np.exp( ( -m + g ) / eps ) ) + g
#         # Check relative error
#         if niter % 10 == 0:
#             err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
#             err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
#             err = 0.5 * (err_f + err_g)
#         niter = niter + 1

#     if verbose:
#         print('Number of iterations in unot:', niter)
#     tmp_K.data = np.exp( ( -C.data + f[C.row] + g[C.col] ) / eps )
#     return tmp_K


# # sot-based sgw
# def recover_full_mapping(
#     M,
#     X1, 
#     X2, 
#     P, 
#     idx1, 
#     idx2,
#     pruned_P = False, 
#     delta = 0.5,
#     eps: float = 5e-2,
#     thresh_CGW: float = 0.04,
#     thresh_CCC: float = 0.65,
#     weight: float = 0.1,
#     nitermax: int = 1e5,
#     stopthr: float = 1e-8,
#     p_cost = 2
# ):
#     D1 = distance_matrix(X1, X1[idx1,:])
#     D2 = distance_matrix(X2, X2[idx2,:])
#     n1, n1_sub = D1.shape
#     n2, n2_sub = D2.shape

#     a = np.ones(n1) / n1
#     b = np.ones(n2) / n2
#     if pruned_P:
#         P_sig = P
#         D1_sig = D1
#         D2_sig = D2
#     else:
#         tmp_idx1 = np.where(P.sum(axis=1) >= delta / n1_sub)[0]
#         tmp_idx2 = np.where(P.sum(axis=0) >= delta / n2_sub)[0]
#         P_sig = P[tmp_idx1,:][:,tmp_idx2]
#         D1_sig = D1[:,tmp_idx1]
#         D2_sig = D2[:,tmp_idx2]
#     P_sig_1_to_2 = P_sig.copy() / P_sig.sum(axis=1).reshape(-1,1)
#     P_sig_2_to_1 = ( P_sig.copy() / P_sig.sum(axis=0).reshape(1,-1) ).T
#     D1_to_D2 = np.matmul(D1_sig, P_sig_1_to_2)
#     D2_to_D1 = np.matmul(D2_sig, P_sig_2_to_1)

#     #############################
#     C1 = distance_matrix(D1_sig, D2_to_D1, p=p_cost)
#     C2 = distance_matrix(D1_to_D2, D2_sig, p=p_cost)
    
#     diff_in_C1 = D1_sig[:, None, :] - D2_to_D1[None, :, :]
#     diff_in_C2 = D1_to_D2[:, None, :] - D2_sig[None, :, :]
    
#     mask = (np.any(np.abs(diff_in_C1) > thresh_CGW, axis=2) | 
#             np.any(np.abs(diff_in_C2) > thresh_CGW, axis=2))
    
#     zero_indices_C = np.where(mask)
    
#     C = 0.5 * (C1 + C2)
#     C_percentile_99 = np.percentile(C, 99)
#     C_capped = np.minimum(C, C_percentile_99)
#     C_norm = C_capped / C_percentile_99
#     matrix_rescaling_checking(C_norm)
#     C_norm += 1e-10
    
#     M_percentile_99 = np.percentile(M, 99)
#     M_capped = np.minimum(M, M_percentile_99)
#     M_norm = M_capped / M_percentile_99
#     matrix_rescaling_checking(M_norm)
#     M_norm += 1e-10
    
#     CM = weight * C_norm + (1 - weight) * M_norm
#     CM[zero_indices_C] = 0
#     CM[M > thresh_CCC] = 0

#     CM_sparse = sparse.coo_matrix(CM)
#     CM_sparse.data -= 1e-10
#     CM_sparse = CM_sparse / CM_sparse.max()
#     ###############################
    
#     m = 2

#     P_full = sot_sinkhorn_l1_sparse(a, b, CM_sparse, eps, m, nitermax=nitermax, stopthr=stopthr)

#     return P_full

#########################################################

from tqdm import tqdm

def sot_sinkhorn_l1_sparse(a, b, C, eps, m, nitermax=20000, stopthr=1e-8, verbose=False):
    """ Solve the unnormalized optimal transport with l1 penalty in sparse matrix format. """

    tmp_K = C.copy()
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    r = np.zeros_like(a)
    s = np.zeros_like(b)
    niter = 0
    err = 100

    pbar = tqdm(total=nitermax, desc=" Sinkhorn-L1", leave=False, disable=not verbose)

    while niter <= nitermax and err > stopthr:
        fprev = f.copy()
        gprev = g.copy()

        # Iteration
        tmp_K.data = np.exp((-C.data + f[C.row] + g[C.col]) / eps)
        f = eps * np.log(a) \
            - eps * np.log(np.sum(tmp_K, axis=1).A.reshape(-1)
            + np.exp((-m + f) / eps)) + f

        tmp_K.data = np.exp((-C.data + f[C.row] + g[C.col]) / eps)
        g = eps * np.log(b) \
            - eps * np.log(np.sum(tmp_K, axis=0).A.reshape(-1)
            + np.exp((-m + g) / eps)) + g

        if niter % 10 == 0:
            err_f = abs(f - fprev).max() / max(abs(f).max(), abs(fprev).max(), 1.)
            err_g = abs(g - gprev).max() / max(abs(g).max(), abs(gprev).max(), 1.)
            err = 0.5 * (err_f + err_g)

        niter += 1
        if niter % 500 == 0 or niter == nitermax:
            pbar.update(500)

    pbar.close()
    if verbose:
        print(f'Number of iterations in unot: {niter}, final error: {err:.2e}')

    tmp_K.data = np.exp((-C.data + f[C.row] + g[C.col]) / eps)
    return tmp_K


#1014 updated
import numpy as np
from scipy.spatial import distance_matrix
from scipy import sparse
from tqdm import tqdm

def _safe_row_normalize(mat, axis=1, eps=1e-12):
    m = mat.copy()
    if axis == 1:
        s = m.sum(axis=1, keepdims=True)
    else:
        s = m.sum(axis=0, keepdims=True)
    s = np.maximum(s, eps)
    return m / s

def _topk_indices_per_row(D, k):
    # 返回一个 list: 第 i 个元素是 D[i] 的 k 个最小值索引（不含自身锚点的特殊处理这里不做）
    k = min(k, D.shape[1])
    # argpartition 比 argsort 快
    idx = np.argpartition(D, kth=k-1, axis=1)[:, :k]
    # 在这 k 个里按实际距离排序
    row_order = np.take_along_axis(D, idx, axis=1).argsort(axis=1)
    return [idx[i, row_order[i]] for i in range(D.shape[0])]

def recover_full_mapping_knn(
    M,
    X1, 
    X2, 
    P,          # subslice transport plan (n1_sub x n2_sub)
    idx1,       # indices (in X1) of subslice-1 anchors, len n1_sub
    idx2,       # indices (in X2) of subslice-2 anchors, len n2_sub
    k1=32,      # kNN size in slice-1 anchor space (for C1)
    k2=32,      # kNN size in slice-2 anchor space (for C2)
    pruned_P = False, 
    delta = 0.5,
    eps: float = 5e-3,
    thresh_CGW: float = 345,
    thresh_CCC: float = 0.65,
    weight: float = 0.1,
    nitermax: int = int(2e4),
    stopthr: float = 1e-8,
    p_cost = 2,
    matrix_rescaling_checking=lambda x: None
):
    """
    与原函数一致的接口 + k1/k2。
    需要你把 sot_solver 传入（例如之前的 sot_sinkhorn_l1_sparse）。
    """
    steps = 8
    pbar = tqdm(total=steps, desc="recover_full_mapping_knn", leave=False)

    # 1) 全量到“锚点”的距离
    # D1: N1 x n1_sub  ; D2: N2 x n2_sub
    D1 = distance_matrix(X1, X1[idx1, :])
    D2 = distance_matrix(X2, X2[idx2, :])
    pbar.update(1)

    N1, n1_sub = D1.shape
    N2, n2_sub = D2.shape

    a = np.ones(N1) / N1
    b = np.ones(N2) / N2

    # 2) 根据 P 的质量筛锚点列
    if pruned_P:
        P_sig = P
        D1_sig = D1
        D2_sig = D2
    else:
        tmp_idx1 = np.where(P.sum(axis=1) >= delta / n1_sub)[0]
        tmp_idx2 = np.where(P.sum(axis=0) >= delta / n2_sub)[0]
        P_sig = P[np.ix_(tmp_idx1, tmp_idx2)]
        D1_sig = D1[:, tmp_idx1]    # N1 x n1_sig
        D2_sig = D2[:, tmp_idx2]    # N2 x n2_sig
    n1_sig = D1_sig.shape[1]
    n2_sig = D2_sig.shape[1]
    pbar.update(1)

    # 3) 
    #   在 n1_sig 坐标系里：用 row-normalized P 将 D2_sig -> D2_to_D1 (N2 x n1_sig)
    #   在 n2_sig 坐标系里：用 col-normalized P 将 D1_sig -> D1_to_D2 (N1 x n2_sig)
    P_row = _safe_row_normalize(P_sig, axis=1)            # n1_sig x n2_sig
    P_colT = _safe_row_normalize(P_sig, axis=0).T         # n2_sig x n1_sig
    D1_to_D2 = D1_sig @ P_row                              # N1 x n2_sig
    D2_to_D1 = D2_sig @ P_colT                             # N2 x n1_sig
    pbar.update(1)

    # 4) 为每个 i/j 选各自 kNN 的锚点索引
    K1_list = _topk_indices_per_row(D1_sig, k1)   # list of arrays, each in [0, n1_sig)
    K2_list = _topk_indices_per_row(D2_sig, k2)   # list of arrays, each in [0, n2_sig)

    # 5) 基于逐点 kNN 的 C1, C2；同时构造 CGW 屏蔽用的 diff
    C1 = np.empty((N1, N2), dtype=float)
    C2 = np.empty((N1, N2), dtype=float)

    
    for j in range(N2):
        Kj1 = None  # 用于 C1 的列集由"i"决定，不是 j；这里先置空
        Kj2 = K2_list[j]  # 用于 C2 的列集由 j 决定（在 n2_sig 坐标系）
        # 预取 D2_sig[j, Kj2] 与 D1_to_D2[:, Kj2]
        D2_sig_j = D2_sig[j, Kj2]                # (k2,)
        D1_to_D2_Kj2 = D1_to_D2[:, Kj2]          # (N1, k2)

        # 局部差值（用于 CGW 屏蔽 & cost）
        diff_C2 = D1_to_D2_Kj2 - D2_sig_j[None, :]  # (N1, k2)
        
        C2[:, j] = np.sum(np.abs(diff_C2) ** p_cost, axis=1) ** (1.0 / p_cost)

        for i in range(N1):
            Ki1 = K1_list[i]                          # 在 n1_sig 坐标系
            # 预取 D1_sig[i, Ki1] 与 D2_to_D1[j, Ki1]
            d1_i = D1_sig[i, Ki1]                     # (k1,)
            d2proj_j = D2_to_D1[j, Ki1]               # (k1,)
            diff_C1 = d1_i - d2proj_j                 # (k1,)
            C1[i, j] = (np.sum(np.abs(diff_C1) ** p_cost)) ** (1.0 / p_cost)

    pbar.update(1)

    # 6) CGW and CCC
    #   为了与上面的 C1/C2 一致，我们再做一次“是否越界”的检查（逐点子坐标）
    mask = np.zeros((N1, N2), dtype=bool)
    for j in range(N2):
        Kj2 = K2_list[j]
        diff_C2 = D1_to_D2[:, Kj2] - D2_sig[j, Kj2][None, :]
        over2 = np.any(np.abs(diff_C2) > thresh_CGW, axis=1)   # (N1,)
        mask[:, j] |= over2
        for i in range(N1):
            Ki1 = K1_list[i]
            diff_C1 = D1_sig[i, Ki1] - D2_to_D1[j, Ki1]
            if np.any(np.abs(diff_C1) > thresh_CGW):
                mask[i, j] = True

    zero_indices_C = np.where(mask)
    print("Number of blocked positions in recover_full sOT (kNN):", np.sum(mask))
    print("Number of possible aligned positions (kNN):", np.sum(~mask))
    pbar.update(1)

    # 7) 归一化 + 与 M 融合
    C = 0.5 * (C1 + C2)
    C_percentile_99 = np.percentile(C, 99)
    C_capped = np.minimum(C, C_percentile_99)
    C_norm = C_capped / max(C_percentile_99, 1e-12)
    matrix_rescaling_checking(C_norm)
    C_norm += 1e-10
    pbar.update(1)

    M_percentile_99 = np.percentile(M, 99)
    M_capped = np.minimum(M, M_percentile_99)
    M_norm = M_capped / max(M_percentile_99, 1e-12)
    matrix_rescaling_checking(M_norm)
    M_norm += 1e-10
    pbar.update(1)

    CM = weight * C_norm + (1 - weight) * M_norm
    CM[zero_indices_C] = 0
    CM[M_norm > thresh_CCC] = 0

    CM_sparse = sparse.coo_matrix(CM)
    CM_sparse.data -= 1e-10
    CM_sparse = CM_sparse / max(CM_sparse.max(), 1e-12)
    pbar.update(1)

    # 8) 运行 sOT
    m = 2
    P_full = sot_sinkhorn_l1_sparse(a, b, CM_sparse, eps, m, nitermax=nitermax, stopthr=stopthr, verbose=True)

    pbar.update(1)
    pbar.close()
    return P_full