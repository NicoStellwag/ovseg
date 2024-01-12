import open3d as o3d
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import random
import numpy as np


def generate_random_color():
    r = random.random()
    g = random.random()
    b = random.random()
    return (r, g, b)


def visualize_segments(
    coords, unique_segments, seg_connectivity, segment_ids, filename
):
    """
    Visualize geometric oversegmentation of scene.
    coords: np array (n_points, 3)
    unique_segments: np array (n_segments_unique,)
    seg_connectivity: np array (n_edges, 2)
    segment_ids: np array (n_points,)
    filename: where to save html
    """
    random.seed(123)
    segment_colors = {s_id.item(): generate_random_color() for s_id in unique_segments}
    segment_color_map = np.array([segment_colors[s_id.item()] for s_id in segment_ids])
    seg_midpoints = []
    for s_id in unique_segments:
        mask = segment_ids == s_id
        points = coords[mask]
        midpoint = points.mean(axis=0)
        seg_midpoints.append(midpoint)
    x_lines, y_lines, z_lines = [], [], []
    for edge in seg_connectivity:
        s_id_u, s_id_v = edge[0], edge[1]
        mid_u, mid_v = seg_midpoints[s_id_u], seg_midpoints[s_id_v]
        x_lines.append(mid_u[0])
        x_lines.append(mid_v[0])
        y_lines.append(mid_u[1])
        y_lines.append(mid_v[1])
        z_lines.append(mid_u[2])
        z_lines.append(mid_v[2])
        x_lines.append(None)  # hack to draw single lines
        y_lines.append(None)
        z_lines.append(None)
    line_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        name="lines",
        line={"color": "black"},
    )

    visualize_3d_feats(
        coords, segment_color_map, filename, additional_traces=[line_trace]
    )


def visualize_instances(coords, pointwise_instances, filename):
    """
    Visualize instance segmentation of scene.
    coords: np array (n_points, 3)
    pointwise_instances: np array (n_points, n_instances) - one hots
    filename: where to save html
    """
    random.seed(123)
    n_instances = pointwise_instances.shape[1]
    instance_colors = [generate_random_color() for _ in range(n_instances)]
    instance_color_map = np.matmul(pointwise_instances, np.array(instance_colors))
    visualize_3d_feats(coords, instance_color_map, filename)


def visualize_3d_feats(
    coords, feats, save_path=None, hovertext=None, additional_traces=None
):
    """
    Visualize features.
    If the feature vector has more than 3 dims, PCA is applied to map it down to 3.
    When using ssh + vscode, specify a save path to a html file,
    go into the hydra save dir and start a web server with:
    python -m http.server 8000
    VSCode will forward the port automatically, so you can view it
    in your local browser.
    """
    # if necessary map features to 3 dims using pca to visualize them as colors
    if feats.shape[1] != 3:
        pca = PCA(n_components=3)
        feats_reduced = pca.fit_transform(feats)
        minv = feats_reduced.min()
        maxv = feats_reduced.max()
        colors = (feats_reduced - minv) / (maxv - minv)
    else:
        colors = feats

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if hovertext:
        scatter = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(size=2, color=colors, opacity=0.8),
            hoverinfo="text",
            hovertext=hovertext,
        )
    else:
        scatter = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(size=2, color=colors, opacity=0.8),
        )

    data = [scatter]
    if additional_traces:
        data += additional_traces
    fig = go.Figure(data)

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

    return fig
