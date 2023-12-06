import open3d as o3d
from sklearn.decomposition import PCA
import plotly.graph_objects as go


def visualize_3d_feats(coords, feats, save_path=None):
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

    scatter = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(size=2, color=colors, opacity=0.8),
    )

    fig = go.Figure(data=[scatter])

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

    return fig
