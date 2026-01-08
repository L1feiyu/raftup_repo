import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import squidpy as sq
import pandas as pd
import numpy as np

import numpy as np

def downsample_slice_by_window(
    sliceA,
    rect_bounds,                 # (xmin, xmax, ymin, ymax)
    max_points=None,             # e.g., 200；None 表示不限制
    strict=True,                 # True: 超过 max_points 抛错；False: 自动截断到 max_points
    coord_key="spatial",         # 如果坐标不在 'spatial'，可改
):
    """
    按矩形小窗口从 sliceA 中挑点（downsample），返回子集 AnnData 和其在原始 sliceA 中的索引。
    - 与你现有 subslice 代码一致：downsampled_sliceA = sliceA[indices_dsa].copy()
    - indices_dsa 是相对于原始 sliceA 的索引（保持原始顺序）

    Parameters
    ----------
    sliceA : AnnData
    rect_bounds : tuple (xmin, xmax, ymin, ymax)
    max_points : int or None
        若指定，限制窗口内最多点数。strict=True 时超过直接报错。
    strict : bool
        True：超过 max_points 抛 ValueError；False：自动保留前 max_points（按原始顺序）。
    coord_key : str
        坐标所在的 obsm 键。

    Returns
    -------
    downsampled_sliceA : AnnData
    indices_dsa : np.ndarray (int)
    """
    if coord_key not in sliceA.obsm_keys():
        raise KeyError(f"sliceA.obsm['{coord_key}'] 不存在，可用键：{list(sliceA.obsm_keys())}")

    coords = np.asarray(sliceA.obsm[coord_key])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"sliceA.obsm['{coord_key}'] 形状应为 (n,2)+，当前 {coords.shape}")

    xmin, xmax, ymin, ymax = rect_bounds

    # 按窗口筛选（注意是闭区间）
    mask = (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax) & \
           (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)

    # 将 True 的位置作为原始索引取出；保持原始顺序
    indices_dsa = np.nonzero(mask)[0]

    n_sub = len(indices_dsa)
    print(f"[downsample_slice_by_window] Sub-slice contains {n_sub} points in window {rect_bounds}")

    # 数量约束
    if max_points is not None and n_sub > max_points:
        if strict:
            raise ValueError(
                f"Window contains {n_sub} points (> {max_points}). "
                f"Please shrink the rectangle or increase max_points."
            )
        else:
            # 非严格：按原始顺序截断
            indices_dsa = indices_dsa[:max_points]
            n_sub = max_points
            print(f"→ Truncated to {max_points} points (strict=False).")

    # 生成子集（若为空则返回 None 和空索引）
    if n_sub > 0:
        downsampled_sliceA = sliceA[indices_dsa].copy()
    else:
        downsampled_sliceA = None

    return downsampled_sliceA, indices_dsa

def downsample_slice(sliceA, tar_distance):
 
    points_A = sliceA.obsm['spatial'].tolist()
    spatial_coords_A = np.array(points_A)

    def create_hexagonal_grid(spatial_coords, spacing):
        max_x, max_y = np.max(spatial_coords, axis=0)
        side_length = spacing / (2 * np.sqrt(3))
        vertical_spacing = 1.5 * side_length
        nx = int(np.ceil(max_x / spacing))
        ny = int(np.ceil(max_y / vertical_spacing))
        
        grid_points = []
        for x in range(nx):
            for y in range(ny):
                offset = 0 if y % 2 == 0 else spacing / 2
                grid_points.append([x * spacing + offset, y * vertical_spacing])
        return np.array(grid_points)

    def downsample_spots(spatial_coords, target_distance):
        grid_points = create_hexagonal_grid(spatial_coords, target_distance)
        downsampled_indices = []
        for point in spatial_coords:
            distances = np.linalg.norm(grid_points - point, axis=1)
            closest_point_idx = np.argmin(distances)
            if closest_point_idx not in downsampled_indices:
                downsampled_indices.append(closest_point_idx)
        downsampled_coords = grid_points[downsampled_indices]
        return downsampled_coords, downsampled_indices

    downsampled_coords_A, _ = downsample_spots(spatial_coords_A, tar_distance)

    def find_closest_original_coords_return_coords(downsampled_coords, original_coords):
        closest_coords = []
        for down_coord in downsampled_coords:
            distances = np.linalg.norm(original_coords - down_coord, axis=1)
            closest_idx = np.argmin(distances)
            closest_coords.append(original_coords[closest_idx])
        return closest_coords

    closest_original_coords_A = find_closest_original_coords_return_coords(downsampled_coords_A, spatial_coords_A)

    indices_dsa = []
    for point in closest_original_coords_A:
        index = np.where((sliceA.obsm['spatial'] == point).all(axis=1))[0]
        if len(index) > 0: 
            indices_dsa.append(index[0])

    if indices_dsa:
        downsampled_sliceA = sliceA[indices_dsa].copy()
    else:
        downsampled_sliceA = None

    return downsampled_sliceA, indices_dsa

def visualize_downsampled_points(sliceA, downsampled_sliceA):
    """
    Visualize the original and downsampled spatial coordinates from sliceA and downsampled_sliceA.

    Parameters:
    - sliceA: AnnData object, contains the original spatial coordinates in `sliceA.obsm['spatial']`.
    - downsampled_sliceA: AnnData object, contains the downsampled spatial coordinates in `downsampled_sliceA.obsm['spatial']`.

    Returns:
    - None, displays a matplotlib plot of the original and downsampled points.
    """
    # Extract coordinates from downsampled slice
    points_new_A = downsampled_sliceA.obsm['spatial'].tolist()
    x_coords_new_A, y_coords_new_A = zip(*points_new_A)

    # Extract coordinates from original slice
    points_A = sliceA.obsm['spatial'].tolist()
    x_coords_A, y_coords_A = zip(*points_A)

    # Calculate point counts
    total_points = len(points_A)
    downsampled_points = len(points_new_A)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords_A, y_coords_A, c='red', s=5, label=f'Original Points ({total_points})')  # Original points
    plt.scatter(x_coords_new_A, y_coords_new_A, c='blue', label=f'Downsampled Points ({downsampled_points})')  # Downsampled points
    
    plt.title('2D Points Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def visualize_subslice(
    sliceA,
    rect_bounds,                      # (xmin, xmax, ymin, ymax)
    max_points=200,
    color_key: str = "original_clusters",
    palette=None,                     # 可传 dict {layer: color} 或序列；None 用 tab20
    figsize=(8, 6),
    s=None,                           # 点大小，None 时随样本量自适应
    alpha=0.85,
    invert_y=True,                    # 常见像素坐标系建议 True
    edgecolors="none",
    show_legend=True,
    title="Sub-slice Visualization",
    save_path=None, 
    **kwargs
):
    # --- 取坐标与分层 ---
    if "spatial" not in sliceA.obsm_keys():
        raise KeyError("sliceA.obsm['spatial'] 不存在。")
    if color_key not in sliceA.obs.columns:
        raise KeyError(f"sliceA.obs['{color_key}'] 不存在。")

    points_A = np.asarray(sliceA.obsm["spatial"])
    if points_A.ndim != 2 or points_A.shape[1] < 2:
        raise ValueError(f"sliceA.obsm['spatial'] 形状需为 (n,2)+，当前 {points_A.shape}")

    x_coords_A, y_coords_A = points_A[:, 0], points_A[:, 1]
    labels = sliceA.obs[color_key].astype("category")
    cats = list(labels.cat.categories)

    n = len(sliceA)
    if s is None:
        s = max(2.0, min(18.0, 120000.0 / max(n, 1)))

    # --- 颜色映射 ---
    if palette is None:
        base = get_cmap("tab20")
        base_colors = [base(i % base.N) for i in range(len(cats))]
        color_map = {cat: base_colors[i] for i, cat in enumerate(cats)}
    elif isinstance(palette, dict):
        missing = [c for c in cats if c not in palette]
        if missing:
            raise ValueError(f"palette 缺少颜色：{missing}")
        color_map = palette
    else:
        seq = list(palette)
        if not seq:
            raise ValueError("palette 为空。")
        color_map = {cat: seq[i % len(seq)] for i, cat in enumerate(cats)}

    # --- 子区域计数与校验 ---
    xmin, xmax, ymin, ymax = rect_bounds
    mask = (x_coords_A >= xmin) & (x_coords_A <= xmax) & \
           (y_coords_A >= ymin) & (y_coords_A <= ymax)
    n_sub = int(mask.sum())
    print(f"Sub-slice contains {n_sub} points")
    if n_sub > max_points:
        raise ValueError(
            f"Selected rectangle contains {n_sub} points (> {max_points}). "
            f"Please shrink the rectangle or lower max_points."
        )

    # --- 作图（按 layer 上色；子区域仅画矩形框，不额外上色）---
    fig, ax = plt.subplots(figsize=figsize)

    # 按类逐层绘制
    for cat in cats:
        m = (labels.values == cat)
        ax.scatter(x_coords_A[m], y_coords_A[m], s=s, c=[color_map[cat]],
                   alpha=alpha, edgecolors=edgecolors, label=f"{cat} ({m.sum()})")

    # 等距坐标
    ax.set_aspect("equal", adjustable="datalim")
    if invert_y:
        ax.invert_yaxis()

    # 矩形框标注子区域
    rect_x = [xmin, xmax, xmax, xmin, xmin]
    rect_y = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(rect_x, rect_y, c='black', linestyle='--', linewidth=1.5, label='Rectangle Bounds')

    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    ax.set_axis_off()

    # 标题加上子区域点数
    ax.set_title(f"{title} : {n_sub} points")
    if show_legend:
        ax.legend(loc="best", frameon=True, ncol=1)  # 强制单列
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 子区域布尔索引 mask 已经计算好
    if save_path is not None:
        subslice = sliceA[mask].copy()
        subslice.write_h5ad(save_path)
        print(f"✅ Sub-slice saved to {save_path}")

    return color_map


import numpy as np
import matplotlib.pyplot as plt


def maxmin_downsample_slice_merfish(sliceA, num_landmarks: int):
    """
    Downsample a MERFISH slice using max-min (farthest point) sampling
    on spatial coordinates.

    This function is intended for MERFISH-style dense point clouds and
    is mainly used for experimental comparisons / ablations.

    Parameters
    ----------
    sliceA : AnnData
        Input slice with spatial coordinates in `sliceA.obsm['spatial']`.
    num_landmarks : int
        Number of landmarks to select.

    Returns
    -------
    downsampled_sliceA : AnnData or None
        Downsampled slice.
    indices_dsa : list[int]
        Indices of selected spots in the original slice.
    """
    if "spatial" not in sliceA.obsm_keys():
        raise KeyError("sliceA.obsm['spatial'] does not exist.")

    points = np.asarray(sliceA.obsm["spatial"])
    n = points.shape[0]

    if num_landmarks > n:
        raise ValueError(
            f"num_landmarks ({num_landmarks}) exceeds number of points ({n})."
        )

    # ---- max-min (farthest point) sampling ----
    landmarks = [np.random.randint(n)]
    distances = np.full(n, np.inf)

    for _ in range(num_landmarks - 1):
        last = points[landmarks[-1]]
        new_dist = np.linalg.norm(points - last, axis=1)
        distances = np.minimum(distances, new_dist)
        next_landmark = np.argmax(distances)
        landmarks.append(next_landmark)

    indices_dsa = list(map(int, landmarks))

    if len(indices_dsa) > 0:
        downsampled_sliceA = sliceA[indices_dsa].copy()
    else:
        downsampled_sliceA = None

    return downsampled_sliceA, indices_dsa


def visualize_merfish_downsampled_points(sliceA, downsampled_sliceA):
    """
    Visualize original vs downsampled spatial points for MERFISH data.

    Parameters
    ----------
    sliceA : AnnData
        Original slice with `obsm['spatial']`.
    downsampled_sliceA : AnnData
        Downsampled slice obtained from maxmin_downsample_slice_merfish.

    Returns
    -------
    None
    """
    if downsampled_sliceA is None:
        raise ValueError("downsampled_sliceA is None; nothing to visualize.")

    A = np.asarray(sliceA.obsm["spatial"])
    A_ds = np.asarray(downsampled_sliceA.obsm["spatial"])

    plt.figure(figsize=(8, 6))
    plt.scatter(
        A[:, 0], A[:, 1],
        s=4, c="lightgray",
        label=f"Original ({len(A)})"
    )
    plt.scatter(
        A_ds[:, 0], A_ds[:, 1],
        s=20, c="red",
        label=f"Downsampled ({len(A_ds)})"
    )

    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.title("MERFISH max-min downsampling")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.show()

