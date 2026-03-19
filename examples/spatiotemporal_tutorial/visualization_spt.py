import numpy as np
from collections import defaultdict
from itertools import product
from mayavi import mlab
from tvtk.api import tvtk
from mayavi.sources.vtk_data_source import VTKDataSource

def plot_st_triplet_mayavi_fast(
    A_xy, B_xy, C_xy,
    W12, W23,
    labels_A, labels_B, labels_C,
    *,
    z_levels=(0.0, 1.0, 2.0),
    type_colors=None,                # dict {"A":(r,g,b),...} -> point colors
    pair_colors=None,                # dict {("A","B"):(r,g,b),...} -> edge colors
    use_tubes=True,                  # True: tubes (pretty), False: lines (fastest)
    tube_radius_intra=0.018,         # for same-type edges (A→A, etc.)
    tube_radius_inter=0.012,         # for cross-type edges (A→B, etc.)
    tube_sides=8,
    line_width=1.5,                  # used if use_tubes=False
    point_scale=0.65,
    show_planes=False,
    plane_alpha=0.08,
    bg=(1,1,1)
):
    """Batch-render 3 slices with typed nodes and pair-colored edges."""
    # ---- Inputs & basics
    A_xy = np.asarray(A_xy, float); B_xy = np.asarray(B_xy, float); C_xy = np.asarray(C_xy, float)
    W12 = np.asarray(W12); W23 = np.asarray(W23)
    labels_A = np.asarray(labels_A); labels_B = np.asarray(labels_B); labels_C = np.asarray(labels_C)
    n1, n2, n3 = len(A_xy), len(B_xy), len(C_xy)
    assert W12.shape == (n1, n2); assert W23.shape == (n2, n3)
    assert len(labels_A)==n1 and len(labels_B)==n2 and len(labels_C)==n3
    zA, zB, zC = z_levels

    # Types and colors
    all_types = tuple(sorted(set(labels_A) | set(labels_B) | set(labels_C)))
    if type_colors is None:
        base = [(0.12,0.47,0.71),(0.17,0.63,0.17),(0.84,0.15,0.16),(0.58,0.40,0.74)]
        type_colors = {t: base[i % len(base)] for i,t in enumerate(all_types)}
    if pair_colors is None:
        # Same-type edges use the node color; cross-type get a contrasting palette
        palette = [
            (0.10,0.63,0.79),(0.85,0.37,0.01),(0.55,0.34,0.29),
            (0.89,0.47,0.76),(0.40,0.76,0.65),(0.80,0.68,0.22),
        ]
        pair_colors = {}
        idx = 0
        for s,t in product(all_types, repeat=2):
            pair_colors[(s,t)] = type_colors[s] if s==t else palette[idx % len(palette)]; idx+=1

    # Extents for optional planes
    all_xy = np.vstack([A_xy, B_xy, C_xy])
    xmin,ymin = all_xy.min(axis=0); xmax,ymax = all_xy.max(axis=0)
    pad_x = 0.05 * (xmax - xmin if xmax>xmin else 1.0)
    pad_y = 0.05 * (ymax - ymin if ymax>ymin else 1.0)
    xmin,xmax = xmin-pad_x, xmax+pad_x; ymin,ymax = ymin-pad_y, ymax+pad_y

    mlab.figure(size=(1000,820), bgcolor=bg)

    # ---- Points per slice, grouped by type (<= 3 actors per slice)
    def draw_slice(xy, z, labels, name):
        groups = defaultdict(list)
        for i,t in enumerate(labels): groups[t].append(i)
        for t,idxs in groups.items():
            idxs = np.asarray(idxs, int)
            mlab.points3d(
                xy[idxs,0], xy[idxs,1], np.full(len(idxs), z),
                scale_factor=point_scale, color=type_colors[t],
                resolution=18, opacity=1.0, name=f"{name}_{t}"
            )
        if show_planes:
            px,py = np.meshgrid(np.linspace(xmin,xmax,2), np.linspace(ymin,ymax,2))
            pz = np.full_like(px, z, float)
            mlab.mesh(px,py,pz, color=(0.96,0.96,0.96), opacity=plane_alpha)

    draw_slice(A_xy, zA, labels_A, "A")
    draw_slice(B_xy, zB, labels_B, "B")
    draw_slice(C_xy, zC, labels_C, "C")

    # ---- Helper: add a whole batch of 2-point line cells as one actor
    def add_edge_batch(P0, P1, color, intra: bool):
        """
        P0, P1: arrays (E,3) start/end points for E edges.
        Creates PolyData with E line cells, then renders as one actor (tube/line).
        """
        if len(P0)==0: return
        E = len(P0)
        pts = np.empty((2*E, 3), dtype=float)
        pts[0::2] = P0; pts[1::2] = P1
        # connectivity: [2, 0,1, 2,2,3, 2,4,5, ...]
        conn = np.empty((E,3), dtype=np.int64)
        conn[:,0] = 2
        conn[:,1] = 2*np.arange(E, dtype=np.int64)
        conn[:,2] = conn[:,1] + 1
        conn = conn.ravel()

        poly = tvtk.PolyData(points=pts)
        cells = tvtk.CellArray()
        cells.set_cells(E, conn)
        poly.lines = cells

        src = VTKDataSource(data=poly); mlab.pipeline.add_dataset(src)
        if use_tubes:
            r = tube_radius_intra if intra else tube_radius_inter
            tube = mlab.pipeline.tube(src, tube_radius=r, number_of_sides=tube_sides)
            mlab.pipeline.surface(tube, color=color)
        else:
            surf = mlab.pipeline.surface(src, color=color)
            surf.actor.property.line_width = float(line_width)

    # ---- Build and batch edges by (type_from, type_to)
    def batch_edges(xy_from, z_from, labels_from,
                    xy_to,   z_to,   labels_to,
                    W, title):
        I,J = np.where(W > 0)
        if len(I)==0: return
        src_types = labels_from[I]
        dst_types = labels_to[J]
        for s,t in product(all_types, repeat=2):
            mask = (src_types==s) & (dst_types==t)
            if not np.any(mask): continue
            i_sel = I[mask]; j_sel = J[mask]
            P0 = np.column_stack((xy_from[i_sel,0], xy_from[i_sel,1], np.full(len(i_sel), z_from)))
            P1 = np.column_stack((xy_to  [j_sel,0], xy_to  [j_sel,1], np.full(len(j_sel), z_to)))
            color = pair_colors[(s,t)]
            intra = (s==t)
            add_edge_batch(P0, P1, color, intra)

    batch_edges(A_xy, zA, labels_A, B_xy, zB, labels_B, W12, "AtoB")
    batch_edges(B_xy, zB, labels_B, C_xy, zC, labels_C, W23, "BtoC")

    mlab.orientation_axes()
    mlab.view(azimuth=45, elevation=70, distance='auto',
              focalpoint=((xmin+xmax)/2, (ymin+ymax)/2, (zA+zC)/2))
    return mlab.gcf()