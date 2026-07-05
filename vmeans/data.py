"""
Data Module - Desktop Version
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons, make_blobs, make_circles


def generate_structured_points(structure='spiral', n_points=500, seed=42):
    """
    Generates various 2D point distributions.
    """
    np.random.seed(seed)
    
    if structure == 'moons':
        X, _ = make_moons(n_samples=n_points, noise=0.05, random_state=seed)
    elif structure == 'blobs':
        X, _ = make_blobs(n_samples=n_points, centers=3, random_state=seed, cluster_std=0.8)
    elif structure == 'spiral':
        theta = np.sqrt(np.random.rand(n_points)) * 3 * np.pi
        r = 1.0 * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        X = np.vstack((x, y)).T
        X += np.random.randn(n_points, 2) * 0.1
    elif structure == 'ring':
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        r = np.random.normal(1.5, 0.1, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        X = np.vstack((x, y)).T
    elif structure == 'flower':
        theta = np.linspace(0, 2 * np.pi, n_points)
        r = 1.0 + 0.4 * np.sin(6 * theta) + np.random.normal(0, 0.05, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        X = np.vstack((x, y)).T
    elif structure == 'cross':
        n_half = n_points // 2
        x1 = np.random.uniform(-3, 3, n_half)
        y1 = np.random.normal(0, 0.15, n_half)
        x2 = np.random.normal(0, 0.15, n_points - n_half)
        y2 = np.random.uniform(-3, 3, n_points - n_half)
        X = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2])
        ])
    elif structure == 'quadrants':
        quarter = n_points // 4
        rem = n_points - 3 * quarter
        # Q1
        x1 = np.random.uniform(1, 4, quarter)
        y1 = np.random.uniform(1, 4, quarter)
        # Q2
        x2 = np.random.uniform(-4, -1, quarter)
        y2 = np.random.uniform(1, 4, quarter)
        # Q3
        x3 = np.random.uniform(-4, -1, quarter)
        y3 = np.random.uniform(-4, -1, quarter)
        # Q4
        x4 = np.random.uniform(1, 4, rem)
        y4 = np.random.uniform(-4, -1, rem)

        X = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2]),
            np.column_stack([x3, y3]),
            np.column_stack([x4, y4])
        ])
    elif structure == 'concentric_circles':
        X, _ = make_circles(n_samples=n_points, factor=0.5, noise=0.05, random_state=seed)
    elif structure == 'anisotropic_blobs':
        X, _ = make_blobs(n_samples=n_points, random_state=170, centers=3)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    elif structure == 'varied_blobs':
        X, _ = make_blobs(n_samples=n_points, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    elif structure == 'aggregation':
        # Aggregation dataset (Gionis et al. 2007) - 788 points, 7 clusters
        # Source: http://cs.joensuu.fi/sipu/datasets/Aggregation.txt
        # Loads real coordinates from Aggregation.npz (place in same directory as this script)
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        npz_path = os.path.join(script_dir, 'Aggregation.npz')
        txt_path = os.path.join(script_dir, 'Aggregation.txt')

        if os.path.exists(npz_path):
            X = np.load(npz_path)['data']
        elif os.path.exists(txt_path):
            # Tab-separated: x, y, label (label is ignored)
            raw = np.loadtxt(txt_path)
            X = raw[:, :2]  # keep only x, y columns
        else:
            raise FileNotFoundError(
                f"Aggregation dataset not found.\n"
                f"Please place 'Aggregation.npz' or 'Aggregation.txt' in:\n"
                f"  {script_dir}\n"
                f"Download from: http://cs.joensuu.fi/sipu/datasets/Aggregation.txt"
            )
    elif structure == 'zahn_compound':
        # Zahn's Compound dataset (Zahn 1971) - 399 points, 6 clusters
        # Source: http://cs.joensuu.fi/sipu/datasets/
        import os, urllib.request

        script_dir = os.path.dirname(os.path.abspath(__file__))
        npz_path = os.path.join(script_dir, 'Compound.npz')
        txt_path = os.path.join(script_dir, 'Compound.txt')
        url = 'http://cs.joensuu.fi/sipu/datasets/Compound.txt'

        if os.path.exists(npz_path):
            X = np.load(npz_path)['data']
        elif os.path.exists(txt_path):
            X = np.loadtxt(txt_path)[:, :2]
        else:
            print(f"Downloading Compound dataset from {url} ...")
            urllib.request.urlretrieve(url, txt_path)
            raw = np.loadtxt(txt_path)
            X = raw[:, :2]
            np.savez(npz_path, data=X)  # 缓存为 npz，下次加载更快
            print(f"Saved to {npz_path}")
    else:
        raise ValueError(f"Unsupported structure type: {structure}")

    return X


def compute_centroid(points):
    """Compute centroid of points"""
    return points.mean(axis=0)


def convert_to_polar(points, centroid):
    """Convert cartesian points to polar coordinates relative to centroid"""
    shifted = points - centroid
    r = np.linalg.norm(shifted, axis=1)
    theta = np.arctan2(shifted[:, 1], shifted[:, 0])
    theta = np.mod(theta, 2 * np.pi)
    return r, theta


def load_and_project_data(raw_data, n_components=2):
    """
    Simplified data loading - no normalization
    """
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(raw_data)
    return projected
