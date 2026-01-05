#3 metrics

import numpy as np
import sklearn
import sklearn.metrics.pairwise
import pandas as pd

# fig7 DLPFC


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, distance_matrix

def compute_pointwise_neighbor_preservation(P, neighbors1, neighbors2, k=200):
    n1, n2 = P.shape
    P = np.asarray(P, dtype=np.float32)

    row_sums = P.sum(axis=1, keepdims=True)
    valid_mask = row_sums[:, 0] != 0
    row_sums[row_sums == 0] = 1  # avoid division by zero

    max_j_indices = np.argmax(P, axis=1)
    preservation_rates = np.zeros(n1)

    for i in range(n1):
        if not valid_mask[i]:
            continue  # skip zero-row entries

        j = max_j_indices[i]
        neighbors_j = neighbors2[j]
        k_indices = neighbors1[i]

        if len(k_indices) == 0 or len(neighbors_j) == 0:
            continue

        P_k_neighbors = P[k_indices][:, neighbors_j]
        
        ratio_k_list = P_k_neighbors.sum(axis=1) / row_sums[k_indices, 0]
        preservation_rates[i] = np.mean(ratio_k_list) if len(ratio_k_list) > 0 else 0

    num_valid = np.sum(valid_mask)
    print(f"{num_valid} out of {n1} points used ({num_valid / n1:.2%})")

    return preservation_rates

def knn_neighbors(X, k):
    tree = KDTree(X)
    idxs = tree.query(X, k=k+1)[1][:,1:]
    return idxs

def GPR_original(P, X1, X2, dis_cut=150.0, P_cut = 1e-100):
    idx_1 = np.where(P.sum(axis=1) >= P_cut)[0]
    X1 = X1[idx_1,:]
    P = P[idx_1,:]
    P[np.where(P<1e-100)] = 0.0
    D1 = distance_matrix(X1, X1)
    D2 = distance_matrix(X2, X2)

    # Neighbors    
    nb_idx_1 = {}
    for i in range(D1.shape[0]):
        nb_idx_1[i] = np.where(D1[i,:] <= dis_cut)[0]
    nb_idx_2 = {}
    for i in range(D2.shape[0]):
        nb_idx_2[i] = np.where(D2[i,:] <= dis_cut)[0]
    # preservation matrix ij
    pr_mat = np.zeros_like(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i,j] == 0:
                continue
            tmp = 0
            idx_1 = nb_idx_1[i]
            idx_2 = nb_idx_2[j]
            for k in idx_1:
                denom = P[k,:].sum()
                if denom == 0:
                    continue
                tmp += P[k,idx_2].sum() / denom
            pr_mat[i,j] = tmp / max(1, len(idx_1))

    # preservation score i
    pr_vec = np.zeros([P.shape[0]])
    for i in range(len(pr_vec)):
        denom = P[i,:].sum()
        if denom == 0:
            pr_vec[i] = 0.0
        else:
            pr_vec[i] = (pr_mat[i,:] * P[i,:]).sum() / denom
    
    return pr_vec.mean()



from scipy.spatial import distance_matrix
def GPR(P, X1, X2, dis_cut=150, P_cut = 1e-100):
    idx_1 = np.where(P.sum(axis=1) >= P_cut)[0]
    X1 = X1[idx_1,:]
    P = P[idx_1,:]
    P[np.where(P<1e-100)] = 0.0
    D1 = distance_matrix(X1, X1)
    # nonzero_D1 = D1[D1 > 0]
    # print(f"D1_min: {nonzero_D1.min()}")
    D2 = distance_matrix(X2, X2)
    # nonzero_D2 = D2[D2 > 0]
    # print(f"D2_min: {nonzero_D2.min()}")


    # Neighbors    
    nb_idx_1 = {}
    for i in range(D1.shape[0]):
        nb_idx_1[i] = np.where(D1[i,:] <= dis_cut)[0]
        #nb_idx_1[i] = np.argsort(D1[i,:])[:7]
    nb_idx_2 = {}
    for i in range(D2.shape[0]):
        nb_idx_2[i] = np.where(D2[i,:] <= dis_cut)[0]
        #nb_idx_2[i] = np.argsort(D2[i,:])[:7]
    # preservation matrix ij
    pr_mat = np.zeros_like(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
        # for j in [np.argmax(P[i,:])]:
            if P[i,j] == 0:
                continue
            tmp = 0
            idx_1 = nb_idx_1[i]
            idx_2 = nb_idx_2[j]
            for k in idx_1:
                tmp += P[k,idx_2].sum() / P[k,:].sum()
            pr_mat[i,j] = tmp / len(idx_1)

    # preservation score i
    pr_vec = np.zeros([P.shape[0]])
    for i in range(len(pr_vec)):
        # pr_vec[i] = pr_mat[i,np.argmax(P[i,:])]
        pr_vec[i] = (pr_mat[i,:] * P[i,:]).sum() / P[i,:].sum()
    
    return pr_vec.mean()


# # Neighbors for CPP
# neighbors1 = knn_neighbors(XA, k=200)
# neighbors2 = knn_neighbors(XB, k=200)

# # Run both metrics
# cpp = compute_pointwise_neighbor_preservation(P, neighbors1, neighbors2, k=200)
# gpr_mean, gpr_vec = GPR_original(P, XA, XB, dis_cut=0.07, P_cut=1e-12)


########

import numpy as np

def cal_layer_based_alignment_result_single(alignment, labels):
    res = []
    #l_dict = {"Layer1": 0, "Layer2": 1, "Layer3": 2, "Layer4": 3, "Layer5": 4, "Layer6": 5, "WM": 6, np.nan: -1}
    l_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
    cnt0 = 0

    for i, elem in enumerate(alignment):
        if labels[i] == '-1' or labels[elem.argmax() + alignment.shape[0]] == '-1':
            continue 
        if l_dict[labels[i]] == l_dict[labels[elem.argmax() + alignment.shape[0]]]:
            cnt0 += 1

    print(alignment.shape[0])
    print(cnt0/alignment.shape[0])
    res = cnt0/alignment.shape[0]
    return res

import numpy as np

def cal_layer_based_alignment_result_full_skip_all_zero(alignment, labels):
    res = []
    l_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
    
    cnt = np.zeros(7, dtype=int)
    valid_rows = 0

    for i, elem in enumerate(alignment):
        # 跳过全零行
        if elem.max() == 0:
            continue  

        j = elem.argmax()
        if labels[i] == '-1' or labels[alignment.shape[0] + j] == '-1':
            continue

        diff = abs(l_dict[str(labels[i])] - l_dict[str(labels[alignment.shape[0] + j])])
        if 0 <= diff <= 6:
            cnt[diff] += 1
        valid_rows += 1

    # 分母只用有效行数
    denom = max(valid_rows, 1)
    res = (cnt / denom).tolist()
    
    print(*res)
    return res

def cal_layer_based_alignment_result_merfish_skill_all_zero(alignment, labels):
    """
    计算 MERFISH 的对齐准确率：
    - 跳过 alignment 中全零的行
    - 跳过任意一端 label 为 -1 或未在 l_dict 中的行
    - 只用有效行数 valid_rows 作为分母
    """

    # 区域 → 数字的映射
    l_dict = {
        "BST": 0,
        "MPA": 1,
        "MPN": 2,
        "PV": 3,
        "PVH": 4,
        "PVT": 5,
        "V3": 6,
        "fx": 7,
        # np.nan: -1   # 这里用不到也可以不写
    }

    cnt0 = 0          # label 一致的计数
    valid_rows = 0    # 有效行计数（非全零 & label 有效）

    n = alignment.shape[0]

    for i, elem in enumerate(alignment):
        # 1) 跳过全零行
        if elem.max() == 0:
            continue

        # 2) 找到当前行的 argmax 匹配 j
        j = elem.argmax()

        label_i = labels[i]
        label_j = labels[n + j]

        # 3) 跳过无效 label（-1 或不在字典里）
        if label_i == '-1' or label_j == '-1':
            continue
        if (label_i not in l_dict) or (label_j not in l_dict):
            continue

        # 4) 统计 label 是否一致
        if l_dict[label_i] == l_dict[label_j]:
            cnt0 += 1

        valid_rows += 1

    # 分母只用有效行数，避免除 0
    denom = max(valid_rows, 1)
    acc = cnt0 / denom

    #print("valid_rows:", valid_rows)
    #print("accuracy:", acc)
    return acc


def cal_layer_based_alignment_result_full(alignment, labels):
    res = []
    #l_dict = {"Layer1": 0, "Layer2": 1, "Layer3": 2, "Layer4": 3, "Layer5": 4, "Layer6": 5, "WM": 6, np.nan: -1}
    l_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
    cnt0 = 0
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    for i, elem in enumerate(alignment):
        if labels[i] == '-1' or labels[elem.argmax() + alignment.shape[0]] == '-1':
            continue 
        if l_dict[labels[i]] == l_dict[labels[elem.argmax() + alignment.shape[0]]]:
            cnt0 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 1:
            cnt1 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 2:
            cnt2 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 3:
            cnt3 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 4:
            cnt4 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 5:
            cnt5 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 6:
            cnt6 += 1
    #print(alignment.shape[0])
    print(cnt0/alignment.shape[0], cnt1/alignment.shape[0], cnt2/alignment.shape[0], cnt3/alignment.shape[0], cnt4/alignment.shape[0], cnt5/alignment.shape[0], cnt6/alignment.shape[0])
    res.extend([cnt0/alignment.shape[0], cnt1/alignment.shape[0], cnt2/alignment.shape[0], cnt3/alignment.shape[0], cnt4/alignment.shape[0], cnt5/alignment.shape[0], cnt6/alignment.shape[0]])
    return res

# fig 7 merfish

import numpy as np
import sklearn
import sklearn.metrics.pairwise
import pandas as pd


def cal_layer_based_alignment_result_merfish(alignment, labels):
    res = []
    l_dict = {"BST": 0, "MPA": 1, "MPN": 2, "PV": 3, "PVH": 4, "PVT": 5, "V3": 6, "fx": 7, np.nan: -1}
    #l_dict = {"Layer1": 0, "Layer2": 1, "Layer3": 2, "Layer4": 3, "Layer5": 4, "Layer6": 5, "WM": 6, np.nan: -1}
    #l_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
    cnt0 = 0

    for i, elem in enumerate(alignment):
        if labels[i] == '-1' or labels[elem.argmax() + alignment.shape[0]] == '-1':
            continue 
        if l_dict[labels[i]] == l_dict[labels[elem.argmax() + alignment.shape[0]]]:
            cnt0 += 1

    print(alignment.shape[0])
    print(cnt0/alignment.shape[0])
    res = cnt0/alignment.shape[0]
    return res




# fig8
import numpy as np
import sklearn
import sklearn.metrics.pairwise
import pandas as pd

def cal_alignment_acc(alignment, gt):
    gt = gt.to_numpy()

    # alignment[np.where(alignment!=np.max(alignment))] = 0
    # alignment[np.where(alignment==np.max(alignment))] = 1

    # Find the maximum value in each row
    # max_values = np.max(alignment, axis=1)
    # print(gt)
    # Create a new array with zeros and set the maximum values in each row
    result = np.zeros_like(alignment)
    for i in range(alignment.shape[0]):
        result[i, np.argmax(alignment[i])] = 1
    #     print(np.argmax(gt[i]))
    #     print(np.argmax(alignment[i]))
    
    # print(result)
    
    s = (result * gt).sum()

    acc = s / alignment.shape[1]
    return acc

def get_gt_result(alignment, gt):
    gt = gt.to_numpy()

    # alignment[np.where(alignment!=np.max(alignment))] = 0
    # alignment[np.where(alignment==np.max(alignment))] = 1

    # Find the maximum value in each row
    # max_values = np.max(alignment, axis=1)
    # print(gt)
    # Create a new array with zeros and set the maximum values in each row
    result = np.zeros_like(alignment)
    for i in range(alignment.shape[0]):
        result[i, np.argmax(alignment[i])] = 1
        # print(np.argmax(gt[i]))
        # print(np.argmax(alignment[i]))
    return gt, result

# fig 9
def plot_aligned_misaligned(alignment, labels, adata1, adata2, data='DLPFC', sec='151507_151508', tool='STAligner', save_dir="./"):

    if data == 'DLPFC':
        spot_s = 75

    matched_idx_list = []
    ad1_match_label = []
    ad2_match_label = [2] * alignment.shape[1]


    for i, elem in enumerate(alignment):
        matched_idx_list.append(elem.argmax())
        if labels[i] == labels[elem.argmax() + alignment.shape[0]]:
            ad1_match_label.append(1)
            ad2_match_label[elem.argmax()] = 1
        else:
            ad1_match_label.append(0)
            ad2_match_label[elem.argmax()] = 0

    adata1.obs['matching_spots'] = ad1_match_label
    adata2.obs['matching_spots'] = ad2_match_label

    adata1.obs['matching_spots'] = adata1.obs['matching_spots'].astype('category')
    adata1.obs['matching_spots'] = adata1.obs['matching_spots'].map({1: 'aligned', 0: 'mis-aligned'})

    adata2.obs['matching_spots'] = adata2.obs['matching_spots'].astype('category')
    adata2.obs['matching_spots'] = adata2.obs['matching_spots'].map({1: 'aligned', 0: 'mis-aligned', 2: 'unaligned'})

    fig, ax = plt.subplots(2,1, figsize=(6,18), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.2})
    sc.pl.spatial(adata1, title=tool, color="matching_spots", spot_size=spot_s, ax=ax[0], show=False)
    sc.pl.spatial(adata2, title=tool, color="matching_spots", spot_size=spot_s, ax=ax[1], show=False)
    
    # Ensure the aspect ratio of the second subplot matches the first
    for axis in ax:
        axis.legend().remove()
    
    plt.tight_layout(pad=3.0)
    fig.text(0.5, 0.03, "Ratio=" + str("{:.2f}".format(alignment.shape[0]/len(set(matched_idx_list)))), 
             fontsize=52, 
             verticalalignment='bottom', 
             horizontalalignment='center')
    
    plt.savefig(os.path.join(save_dir, "SAM" + tool + sec + "viz.pdf"), bbox_inches="tight")
    plt.show()

def get_ratio(alignment, labels):
    matched_idx_list = []
    ad1_match_label = []
    ad2_match_label = [2] * alignment.shape[1]


    for i, elem in enumerate(alignment):
        # print(i, elem)
        # print(elem.argmax(), alignment.shape[0])
        matched_idx_list.append(elem.argmax())
        if labels[i] == labels[elem.argmax() + alignment.shape[0]]:
            ad1_match_label.append(1)
            ad2_match_label[elem.argmax()] = 1
        else:
            ad1_match_label.append(0)
            ad2_match_label[elem.argmax()] = 0
    
    return alignment.shape[0]/len(set(matched_idx_list))