import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import time

def find_normal(pts):
    W, V = np.linalg.eig([np.cov(pts.T)])
    return V[0][np.argsort(W[0])[0]]

def cosine_similarity(norm1, norm2):
    # return (norm1 @ norm2.T) / (np.sqrt(np.sum(norm1**2)) * np.sqrt(np.sum(norm2**2)))
    return (norm1 @ norm2.T)

def compute_normal_estimates(pts, neighbor_radius, distance_treshold):

    KDtree = KDTree(pts)
    NN = KDtree.query_radius(pts, r=neighbor_radius)
    
    n_pts = len(pts)
    counts = np.ones(n_pts, dtype=int)
    norm_array = np.zeros((n_pts, 3), dtype=float)

    for neighbors_idx in NN:
        n_neighbors = len(neighbors_idx)
        neighbors_pts = pts[neighbors_idx]
        if n_neighbors < 3: continue

        num_iterations = int(np.log(0.01)/np.log(0.5**3))+1
        pcd_within_radius = o3d.geometry.PointCloud()
        pcd_within_radius.points = o3d.utility.Vector3dVector(neighbors_pts)
        _, inliers = pcd_within_radius.segment_plane(distance_threshold=distance_treshold,
                                                                        ransac_n=3,
                                                                        num_iterations=num_iterations)
        
        n_inliers = len(inliers)
        n_outliers = n_neighbors-n_inliers
        if n_outliers == 0 or n_inliers / n_outliers >= 1.0:
            normal = find_normal(neighbors_pts[inliers])
            inlier_idx = neighbors_idx[inliers]
            counts[inlier_idx] += 1
            normalized_normal = np.abs(normal)/np.linalg.norm(normal)
            norm_array[inlier_idx] += normalized_normal

    normals = norm_array / counts[:,None]

    return normals, NN