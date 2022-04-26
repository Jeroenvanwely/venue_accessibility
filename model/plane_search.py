import numpy as np
import open3d as o3d

from sklearn.neighbors import KDTree
from model.normal_estimation import find_normal, cosine_similarity
from scipy.spatial.distance import cdist

from model.connected_components import connected_components2

def find_plane(pts):
    """
    Function that takes a set of points a fits a plane to it, and returns
    its plane coordinates, plane length, plane normal.

    Parameters
    ----------
    pts : array
        Array of points.

    Output
    ------
    plane : array
        Array containing the first 4 coordinates of the plane that is fitted
        to the points in pts. i.e., plane vector.
    normal : array
        The plane normal vector.
    """

    num_iterations = int(np.log(0.01)/np.log(0.5**3))+1 * 3
    pcd_within_radius = o3d.geometry.PointCloud()
    pcd_within_radius.points = o3d.utility.Vector3dVector(np.array(pts))
    plane, inliers = pcd_within_radius.segment_plane(distance_threshold=0.12,
                                                                    ransac_n=3,
                                                                    num_iterations=num_iterations)

    normal = find_normal(pts[inliers])

    return plane, normal

def compute_planes(pts, pt_groups_idx):
    """
    Function that takes a set of points and an array containing arrays of 
    indices corresponding with points in pts that belong to the same group.
    It computes and returns the plane vector, plane length, plane normal 
    vector, and fourth plane coordinate d for each group of points.

    Parameters
    ----------
    pts : array
        Array of points.
    pt_groups_idx : array
        Array containing arrays of indices corresponding with points in pts 
        that belong to the same group.

    Output
    ------
    planes : array
        Array containing plane vectors.
    normals : array
        Array containing normal vectors of planes.
    """

    # For every point-group in 'pts_groups' compute its plane info.
    planes = []
    normals = []
    for group_idx in pt_groups_idx:

        pt_group = pts[group_idx]
        try:
            if len(group_idx) < 3:
                planes.append([0,0,0, None])
                normals.append([0,0,0])
                continue
            plane, normal = find_plane(pt_group)

            # Append plane info to corresponding lists.
            planes.append(plane)
            normals.append(normal)
        except:
            continue

    return planes, normals

def merge_connected_components(pts, pt_groups_idx):
    planes, normals = compute_planes(pts, pt_groups_idx)

    merged = []
    group_merged = np.zeros(len(planes))
    for i in range(len(planes)):
        if planes[i][3] == None: continue
        if group_merged[i] == 1: continue

        merged.append(pt_groups_idx[i])

        for j in range(i+1, len(planes)):

            if i == j or planes[j][3] == None: continue

            dist = (planes[i] @ planes[j].T) / (np.sqrt(np.sum(planes[i]**2)) * np.sqrt(np.sum(planes[j]**2)))
            
            means_1 = np.median(pts[pt_groups_idx[i]], axis=0)
            means_2 = np.median(pts[pt_groups_idx[j]], axis=0)

            pts2 = pts[pt_groups_idx[j]]
            dist1 = np.abs(planes[i][0]*pts2[:,0] + planes[i][1]*pts2[:,1] + planes[i][2]*pts2[:,2] + planes[i][3]) / (np.sqrt( planes[i][0]**2 + planes[i][1]**2 + planes[i][2]**2))
            average_dist = np.mean(dist1)

            max_i = np.max(pts[pt_groups_idx[i]][:, 0:2], axis=0)
            max_j = np.max(pts[pt_groups_idx[j]][:, 0:2], axis=0)
            min_i = np.min(pts[pt_groups_idx[i]][:, 0:2], axis=0)
            min_j = np.min(pts[pt_groups_idx[j]][:, 0:2], axis=0)

            bbox_i = [[max_i[0], max_i[1]], [max_i[0], min_i[1]], [min_i[0], max_i[1]], [min_i[0], min_i[1]]]
            bbox_j = [[max_j[0], max_j[1]], [max_j[0], min_j[1]], [min_j[0], max_j[1]], [min_j[0], min_j[1]]]
            
            distances = cdist(bbox_i, bbox_j)
            dist_betw_means_xy = np.min(distances, axis=None)

            # dist_betw_means_xy = np.sqrt(np.sum((means_1[:2] - means_2[:2])**2))
            if dist >= 0.3 and np.abs(means_2[2]-means_1[2]) < 0.15 and average_dist < 0.15 and dist_betw_means_xy < .8:
                merged[-1] = np.append(merged[-1], pt_groups_idx[j], axis=0)
                group_merged[j] = 1 

    return np.asarray(merged, dtype=object)

def merge_overlapping_groups(pts, pt_group_idx, new_pt_group_idx, overlap_ratio_t):
    """
    Function that merges groups of points that have much overlap with
    each other.

    Parameters
    ----------
    pt_group_idx : array
        Array of point indices corresponding with points that belong to the same
        group.
    new_pt_group_idx : array
        Array containing arrays of indices corresponding with points that belong 
        to the same group.
    overlap_ratio_t : float
        Float value defining the minimum overlapping point ratio two sets of points
        must have to want to merge them.

    Output
    ------
    new_pt_group_idx : array
        Array similar to the input 'new_pt_group_idx' except either point indices
        have been added to one of the groups, or an extra group has been added.
    """
    planes, normals = compute_planes(pts, new_pt_group_idx)
    plane = find_plane(pts[pt_group_idx])

    temp = np.sort(pts[pt_group_idx][2])
    temp_z_min = np.mean(temp[:10])
    temp_z_max = np.mean(temp[-10:])

    max_ratio = 0 # Initialize the maximum found ratio at 0.

    # Loop over every point-group in 'new_pt_group_idx' and compute the ratio
    # of overlapping points with 'plane_indices'.
    for j in range(len(new_pt_group_idx)):
        new = np.sort(pts[new_pt_group_idx[j]][2])
        new_z_min = np.mean(new[:10])
        new_z_max = np.mean(new[-10:])

        # diff = np.abs(new_z_min - temp_z_min) + (new_z_max - temp_z_max)**2)

        # overlapping_pts = np.array(set(pt_group_idx)&set(new_pt_group_idx[j]).tolist())
        overlap = len(set(pt_group_idx)&set(new_pt_group_idx[j]))
        overlap_ratio = overlap/len(pt_group_idx)

        # If the current 'max_ratio' is exceeded, i.e., we found a point-group
        # with more overlap, we set this point-group as the new most likely
        # candidate to merge with.
        # if overlap_ratio > max_ratio and diff <= 0.13:
        # if overlap < 35:
        #     for point in overlapping_pts:
        #         pts[point]

        if overlap > 35 and np.abs(new_z_min - temp_z_min) <= 0.15 and np.abs(new_z_max - temp_z_max) <= 0.15:
            max_ratio = overlap_ratio
            max_pt_group_idx = j

    # If the most likely merge candidate has an overlap ratio exceeding 
    # 'overlap_ratio_t', we merge 'new_pt_group_idx' with this candidate.
    if max_ratio > overlap_ratio_t:
        new_group = np.union1d(new_pt_group_idx[max_pt_group_idx], pt_group_idx)
        new_pt_group_idx[max_pt_group_idx] = new_group
    
    # Otherwise, there is not enough overlap with any other point-group, and thus
    # we add to 'new_pt_group_idx' without merging.
    else:
        new_pt_group_idx.append(pt_group_idx)
    
    return new_pt_group_idx

def plane_search(pts, pt_groups_idx, extra_pts_idx, params, point_normals):
    """
    Function that extends groups of points in pt_groups_idx with extra points.

    Parameters
    ----------
    pts : array
        Array of points.
    pt_groups_idx : array
        Array containing arrays of indices corresponding with points in pts 
        that belong to the same group.
    extra_pts_idx : array
        Array of points containing indices corresponding to points that could 
        potentially be added to point-groups in 'pt_groups_idx'.
    params : tuple
        Tuple of parameters containing:
            close_p_search_r : float 
                Search radius for points that are close to a group of points.
            dist_to_plane_t : float
                Threshold defining maximum distance a point can be to a plane to 
                still be considered as potentially within the plane.
            neighboor_search_r : float
                Search radius for near points.
            neighboors_t : float
                Threshold defining the minimum number of points that must be within 
                a radius 'neighboor_search_r' to not be considered an outlier.
            z_margin : float
                The margin the z-value of a point can be above the maximum or under 
                the minimum z-value of a group of points to still be considered as 
                also part of that group of points.
            overlap_ratio_t : float
                Threshold defining the minimum overlapping point ratio two sets of 
                points must have to be able to merge them.
            pt_group_search_r : float
                Search radius used for searching similar planes and to compute whether
                two group points are close to each other or not.
            dist_betw_plane_t : float
                Threshold value defining how far apart two planes are allowed to be, to
                still be considered similar.
    Output
    ------
    new_pt_groups : array
        Array similar to the input 'pt_groups_idx' except point indices might
        have been added to one of the groups or and entire groups have been removes
        or added.
    """

    # Parameters
    (close_p_search_r, dist_to_plane_t, z_margin, overlap_ratio_t) = params

    # Compute the plane info of each point-group.
    # planes, plane_lengths, norm_planes, d_coor = compute_planes(pts, pt_groups_idx)
    planes, normals = compute_planes(pts, pt_groups_idx)

    new_pt_groups = [] 
    for i, group_idx in enumerate(pt_groups_idx):
        
        # Do nothing if z-value of normal vector is lower then 0.35 i.e., not vertical.
        # if planes[i][3] == None or normals[i][2] >= 0.55: continue
        
        # Compute mean of 10 lowest and highest z-values from current point-group.
        group_pts = pts[group_idx]
        sorted_z = np.sort(group_pts[:,2])
        min_z = np.mean(sorted_z[:10])-z_margin
        max_z = np.mean(sorted_z[-10:])+z_margin

        if len(group_pts) < 3 or len(extra_pts_idx) < 3:
            new_pt_groups.append(group_idx)
            continue

        # Filter 1: Find points from 'extra_pts_idx' that are within 
        # 'close_p_search_radius' distance to any point in the current point-group.
        pts_tree = KDTree(group_pts)
        close_pt_count = pts_tree.query_radius(pts[extra_pts_idx], 
                                                r=close_p_search_r, 
                                                count_only=True)
        close_pts_idx = extra_pts_idx[np.asarray(close_pt_count > 0).nonzero()[0]]
        close_pts = pts[close_pts_idx]
        
        # Filter 2: Find points from 'close_pts_idx' that are within 
        # 'dist_to_plane_t' distance to the plane-vector of the current 
        # point-group and have z-values between max_z and min_z.
        pts2 = close_pts
        dist_to_plane = np.abs(planes[i][0]*pts2[:,0] + planes[i][1]*pts2[:,1] + planes[i][2]*pts2[:,2] + planes[i][3]) / (np.sqrt( planes[i][0]**2 + planes[i][1]**2 + planes[i][2]**2))

        z_coor = close_pts[:,2]
        conditions = ((dist_to_plane<dist_to_plane_t) & (z_coor<=max_z) & (z_coor>=min_z))
        pts_in_plane_idx = close_pts_idx[np.asarray(conditions).nonzero()[0]]
                    
        if len(pts_in_plane_idx) == 0:
            new_pt_groups.append(group_idx)
            continue

        _, plane_indices = connected_components2(pts, pts_in_plane_idx, np.array([group_idx]), point_normals, normals[i], dist_t=0.1)

        # # Merge 2: Merge newly found points with original point-group.
        # plane_indices = np.union1d(group_idx, pts_in_plane_idx)

        # Merge 3: Merge point-groups if overlap-ratio exceeds 'overlap_ratio_t'.
        new_pt_groups = merge_overlapping_groups(pts, plane_indices, new_pt_groups, 
                                                    overlap_ratio_t)

    return np.asarray(new_pt_groups, dtype=object)