# 2d and 3d plots
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#2d
def plot_3d(X1, X2, P, idx1, idx2, linewidth=1, thresh=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1[:,0], X1[:,1], np.zeros(X1.shape[0]), c='red', s=5, alpha=0.6)
    ax.scatter(X2[:,0], X2[:,1], np.ones(X2.shape[0]), c='blue', s=5, alpha=0.6)
    n1, n2 = P.shape
    for i in range(P.shape[0]):
        if np.sum(P[i,:]) > thresh / n1:
            j = np.argmax(P[i,:])
            ax.plot([X1[idx1[i],0], X2[idx2[j],0]],[X1[idx1[i],1], X2[idx2[j],1]],[0,1], c='r', linewidth=linewidth)
    for j in range(P.shape[1]):
        if np.sum(P[:,j]) > thresh / n2:
            i = np.argmax(P[:,j])
            ax.plot([X1[idx1[i],0], X2[idx2[j],0]],[X1[idx1[i],1], X2[idx2[j],1]],[0,1], c='b', linewidth=linewidth)    


#3d
import plotly.graph_objs as go
import numpy as np
import seaborn as sns

def plot_slices_overlap_FSGW_color(
    slices, P, acc, cutoff_CGW, output_dir, layer_to_color_map=None, plotabv=1e-4
):
    """
    Plot the overlap of two slices in 3D using fused supervised Gromov-Wasserstein (FSGW) alignment results.
    
    Parameters:
    - slices: list of AnnData objects, the two slices to plot.
    - P: np.ndarray, the probabilistic matching matrix.
    - acc: float, ARI score or alignment accuracy.
    - cutoff_CGW: float, cutoff value for fused supervised Gromov-Wasserstein alignment.
    - output_dir: str, directory to save the plot as an interactive HTML file.
    - layer_to_color_map: dict, mapping of layer names to colors (optional, default is None).
    - plotabv: float, threshold for plotting connection weights (default is 1e-4).

    Returns:
    - None, saves the interactive plot as an HTML file.
    """
    if layer_to_color_map is None:
        layer_to_color_map = {'Layer{0}'.format(i + 1): sns.color_palette()[i] for i in range(6)}
        layer_to_color_map['WM'] = sns.color_palette()[6]

    fig = go.Figure()

    x_offset = 0  # X-axis offset for the second slice
    y_offset = 0
    z_offset = 200  # Z-axis offset for better 3D visualization

    x1 = slices[0].obsm['spatial']
    x2 = slices[1].obsm['spatial']

    # Plot the slices first, then the connections
    for i, adata in enumerate(slices):
        ground_truth_values = adata.obs['ground_truth'].astype('str')
        colors = [layer_to_color_map.get(layer, '#808080') for layer in ground_truth_values]

        fig.add_trace(go.Scatter3d(
            x=adata.obsm['spatial'][:, 0] + i * x_offset,
            y=adata.obsm['spatial'][:, 1] + i * y_offset,
            z=[i * z_offset] * len(adata.obsm['spatial']),
            mode='markers',
            marker=dict(size=6, color=colors),
            hoverinfo='skip',
            showlegend=False
        ))

    # Plot connections with weights and hover effects
    for i in range(P.shape[0]):
        j = np.argmax(P[i, :])
        if P[i, j] > plotabv:
            fig.add_trace(go.Scatter3d(
                x=[x1[i, 0], x2[j, 0] + x_offset],
                y=[x1[i, 1], x2[j, 1] + y_offset],
                z=[0, z_offset],
                mode='lines',
                line=dict(color='grey', width=2),
                hoverinfo='text',
                hovertext=f'Connection weight: {P[i, j]:.4f}<br>Point 1: ({i})<br>Point 2: ({j})',
                showlegend=False
            ))

    # Add manual legend
    for layer, color in layer_to_color_map.items():
        fig.add_trace(go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=layer
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f's={P.sum():.4f} ARI={acc}',
            x=0.5,
            xanchor='center',
            yanchor='top',
            y=0.95
        ),
        showlegend=True,
        legend=dict(title='Cortex layer'),
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        width=1000,
        height=500
    )

    # Save as interactive HTML file
    output_path = f"{output_dir}/colorplot_{cutoff_CGW}.html"
    fig.write_html(output_path)

    print(f"Plot saved to {output_path}")

##########
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_slice_by_layer(
    adata,
    coord_key: str = "spatial",
    color_key: str = "original_clusters",
    palette=None,                 # 可传入 dict: {layer_name: "#RRGGBB"} 或 list/tuple 颜色序列
    figsize=(8, 7),
    s=None,                       # 点大小（默认随样本量自适应）
    alpha=0.85,
    equal_aspect=True,
    invert_y=True,                # 许多空间转录组坐标 Y 轴向下增大；若图上下颠倒可改为 False
    edgecolors="none",
    title: str = None,
):
    """
    根据 AnnData 的真实 2D 坐标绘制样本点，并按层 (obs[color_key]) 上色。

    Parameters
    ----------
    adata : AnnData
        含有空间坐标与分层信息的对象
    coord_key : str
        2D 坐标所在的 .obsm 键名（默认 'spatial'）
    color_key : str
        分层信息所在的 .obs 列名（默认 'original_clusters'）
    palette : dict or list/tuple or None
        若为 dict，键是层名、值是颜色；若为序列，将按顺序分配给各层；None 则自动使用 'tab20'
    figsize : tuple
        图尺寸
    s : float or None
        散点大小，None 时按样本量自适应
    alpha : float
        散点透明度
    equal_aspect : bool
        是否强制等比例坐标轴
    invert_y : bool
        是否翻转 y 轴方向（默认 True，适配常见空间切片坐标）
    edgecolors : str
        散点边缘色
    title : str or None
        自定义标题；None 时自动生成
    """

    # --- 基本检查 ---
    if coord_key not in adata.obsm_keys():
        raise KeyError(f"adata.obsm['{coord_key}'] 不存在。可用键：{list(adata.obsm_keys())}")
    if color_key not in adata.obs.columns:
        raise KeyError(f"adata.obs['{color_key}'] 不存在。可用列部分示例：{list(adata.obs.columns[:10])}")

    coords = np.asarray(adata.obsm[coord_key])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{coord_key}'] 必须是 (n_obs, 2) 或以上，当前形状：{coords.shape}")

    x, y = coords[:, 0], coords[:, 1]
    labels = adata.obs[color_key].astype("category")
    cats = list(labels.cat.categories)

    n = len(adata)
    k = len(cats)

    # --- 点大小自适应 ---
    if s is None:
        # 简单随规模缩放：大样本更小
        s = max(2.0, min(18.0, 120000.0 / max(n, 1)))

    # --- 构建调色板 ---
    color_map = {}
    if palette is None:
        # 使用 tab20 循环
        base = get_cmap("tab20")
        base_colors = [base(i % base.N) for i in range(k)]
        color_map = {cat: base_colors[i] for i, cat in enumerate(cats)}
    elif isinstance(palette, dict):
        # 用户显式提供 dict
        missing = [c for c in cats if c not in palette]
        if missing:
            raise ValueError(f"palette 缺少以下层的颜色：{missing}")
        color_map = palette
    else:
        # 序列：按顺序分配（不够则循环）
        palette_seq = list(palette)
        if len(palette_seq) == 0:
            raise ValueError("palette 为空。")
        color_map = {cat: palette_seq[i % len(palette_seq)] for i, cat in enumerate(cats)}

    # --- 作图 ---
    fig, ax = plt.subplots(figsize=figsize)

    # 背景淡灰（可开关：把下面这行注释掉即可不画背景）
    ax.scatter(x, y, s=max(1.0, s*0.35), c="lightgray", alpha=0.6, edgecolors="none", label="_bg")

    # 分层逐类绘制
    handles = []
    for cat in cats:
        mask = (labels.values == cat)
        xi, yi = x[mask], y[mask]
        h = ax.scatter(xi, yi, s=s, c=[color_map[cat]], alpha=alpha, edgecolors=edgecolors, label=f"{cat} ({mask.sum()})")
        handles.append(h)

    # 轴与标题
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title is None:
        ax.set_title(f"Slice by '{color_key}' • n={n}, layers={k}")
    else:
        ax.set_title(title)

    # 图例（按两列排，避免过长）
    if len(handles) > 0:
        ax.legend(loc="best", frameon=True, markerscale=1.0, ncol=1)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 返回颜色映射，便于复用/保存
    return color_map