import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import squidpy as sq
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.neighbors import LocalOutlierFactor

def maxmin_sampling(points, num_landmarks, random_state=23):
    """
    Downsample using deterministic max-min sampling.

    Parameters:
    - points: np.ndarray, spatial coordinates.
    - num_landmarks: int, number of selected landmarks.
    - random_state: int or None, seed for reproducibility.

    Returns:
    - downsampled_coords: np.ndarray, selected landmark coordinates.
    - landmark_indices: list, indices of selected points.
    """
    num_points = points.shape[0]
    if num_landmarks > num_points:
        raise ValueError("num_landmarks cannot exceed the number of available points.")
    
    if random_state is not None:
        np.random.seed(random_state)

    landmarks = [np.random.randint(num_points)]
    distances = np.full(num_points, np.inf)

    for _ in range(num_landmarks - 1):
        last_landmark = points[landmarks[-1]]
        new_distances = np.linalg.norm(points - last_landmark, axis=1)
        distances = np.minimum(distances, new_distances)
        new_landmark = np.argmax(distances)
        landmarks.append(new_landmark)

    return points[np.array(landmarks)], landmarks



def robust_maxmin_downsample_slice(sliceA, num_landmarks, outlier_fraction=0.05, n_neighbors=20):
    """
    Downsample the spatial coordinates of a slice using max-min sampling,
    excluding detected outlier points.

    Parameters:
    - sliceA: AnnData object, containing spatial data.
    - num_landmarks: int, number of landmarks to select.
    - outlier_fraction: float, expected fraction of outliers (default 5%).
    - n_neighbors: int, number of neighbors used in LOF (default 20).

    Returns:
    - downsampled_sliceA: AnnData object, downsampled version of sliceA.
    - indices_dsa: list, indices of selected spots in the original data.
    - cleaned_indices: list, indices of points after outlier removal.
    """
    spatial_coords_A = np.array(sliceA.obsm['spatial'])
    
    # Step 1: Detect and remove outliers using Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=outlier_fraction)
    is_inlier = lof.fit_predict(spatial_coords_A) > 0  # Inliers are labeled as 1

    cleaned_coords = spatial_coords_A[is_inlier]
    cleaned_indices = np.where(is_inlier)[0]

    if cleaned_coords.shape[0] < num_landmarks:
        raise ValueError(f"Not enough points after removing outliers ({cleaned_coords.shape[0]} left, but {num_landmarks} requested).")

    # Step 2: Max-min sampling on cleaned data
    downsampled_coords_A, local_indices_dsa = maxmin_sampling(cleaned_coords, num_landmarks, random_state=23)
    
    # Step 3: Map local indices back to original sliceA indices
    indices_dsa = cleaned_indices[local_indices_dsa]

    # Create downsampled AnnData
    downsampled_sliceA = sliceA[indices_dsa].copy()

    return downsampled_sliceA, indices_dsa, cleaned_indices


def downsample_slice_by_window(
    sliceA,
    rect_bounds,                 # (xmin, xmax, ymin, ymax)
    max_points=None,             # e.g., 200; None means no limit
    strict=True,                 # True: raise if max_points is exceeded; False: truncate to max_points
    coord_key="spatial",         # Change this if coordinates are not stored in 'spatial'
):
    """
    Select observations from a rectangular window in sliceA and return the AnnData subset and its indices in the original sliceA.
    - Uses the same subsetting operation: downsampled_sliceA = sliceA[indices_dsa].copy()
    - indices_dsa contains indices into the original sliceA, preserving their order

    Parameters
    ----------
    sliceA : AnnData
    rect_bounds : tuple (xmin, xmax, ymin, ymax)
    max_points : int or None
        If specified, limit the number of observations in the window. Raise an error if exceeded when strict=True.
    strict : bool
        True: raise ValueError if max_points is exceeded; False: retain the first max_points observations in their original order.
    coord_key : str
        Key in obsm containing coordinates.

    Returns
    -------
    downsampled_sliceA : AnnData
    indices_dsa : np.ndarray (int)
    """
    if coord_key not in sliceA.obsm_keys():
        raise KeyError(f"sliceA.obsm['{coord_key}'] is missing. Available keys: {list(sliceA.obsm_keys())}")

    coords = np.asarray(sliceA.obsm[coord_key])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"sliceA.obsm['{coord_key}'] must have shape (n, 2) or more columns; current shape: {coords.shape}")

    xmin, xmax, ymin, ymax = rect_bounds

    # Filter by the window using inclusive boundaries
    mask = (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax) & \
           (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)

    # Extract True positions as original indices, preserving order
    indices_dsa = np.nonzero(mask)[0]

    n_sub = len(indices_dsa)
    print(f"[downsample_slice_by_window] Sub-slice contains {n_sub} points in window {rect_bounds}")

    # Observation-count constraint
    if max_points is not None and n_sub > max_points:
        if strict:
            raise ValueError(
                f"Window contains {n_sub} points (> {max_points}). "
                f"Please shrink the rectangle or increase max_points."
            )
        else:
            # Non-strict mode: truncate in the original order
            indices_dsa = indices_dsa[:max_points]
            n_sub = max_points
            print(f"→ Truncated to {max_points} points (strict=False).")

    # Create the subset (return None and empty indices if no observations remain)
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
    palette=None,                     # Accepts a dict {layer: color} or a sequence; None uses tab20
    figsize=(8, 6),
    s=None,                           # Marker size; None adjusts it to the sample size
    alpha=0.85,
    invert_y=True,                    # True is recommended for common pixel coordinate systems
    edgecolors="none",
    show_legend=True,
    title="Sub-slice Visualization",
    save_path=None, 
    **kwargs
):
    # --- Read coordinates and layer annotations ---
    if "spatial" not in sliceA.obsm_keys():
        raise KeyError("sliceA.obsm['spatial'] is missing.")
    if color_key not in sliceA.obs.columns:
        raise KeyError(f"sliceA.obs['{color_key}'] is missing.")

    points_A = np.asarray(sliceA.obsm["spatial"])
    if points_A.ndim != 2 or points_A.shape[1] < 2:
        raise ValueError(f"sliceA.obsm['spatial'] must have shape (n, 2) or more columns; current shape: {points_A.shape}")

    x_coords_A, y_coords_A = points_A[:, 0], points_A[:, 1]
    labels = sliceA.obs[color_key].astype("category")
    cats = list(labels.cat.categories)

    n = len(sliceA)
    if s is None:
        s = max(2.0, min(18.0, 120000.0 / max(n, 1)))

    # --- Color mapping ---
    if palette is None:
        base = get_cmap("tab20")
        base_colors = [base(i % base.N) for i in range(len(cats))]
        color_map = {cat: base_colors[i] for i, cat in enumerate(cats)}
    elif isinstance(palette, dict):
        missing = [c for c in cats if c not in palette]
        if missing:
            raise ValueError(f"palette is missing colors: {missing}")
        color_map = palette
    else:
        seq = list(palette)
        if not seq:
            raise ValueError("palette is empty.")
        color_map = {cat: seq[i % len(seq)] for i, cat in enumerate(cats)}

    # --- Count and validate observations in subregions ---
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

    # --- Plot by layer; outline subregions with rectangles without additional coloring---
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each layer separately
    for cat in cats:
        m = (labels.values == cat)
        ax.scatter(x_coords_A[m], y_coords_A[m], s=s, c=[color_map[cat]],
                   alpha=alpha, edgecolors=edgecolors, label=f"{cat} ({m.sum()})")

    # Equal axis aspect ratio
    ax.set_aspect("equal", adjustable="datalim")
    if invert_y:
        ax.invert_yaxis()

    # Outline subregions with rectangles
    rect_x = [xmin, xmax, xmax, xmin, xmin]
    rect_y = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(rect_x, rect_y, c='black', linestyle='--', linewidth=1.5, label='Rectangle Bounds')

    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    ax.set_axis_off()

    # Include subregion observation counts in the title
    ax.set_title(f"{title} : {n_sub} points")
    if show_legend:
        ax.legend(loc="best", frameon=True, ncol=1)  # Use a single column
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # The boolean subregion mask has already been computed
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

