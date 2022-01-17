import numpy as np
from numpy.linalg import svd
from sklearn.neighbors import KDTree

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
        Array containing the first 3 coordinates of the plane that is fitted
        to the points in pts. i.e., plane vector.
    plane_length : float
        Float value defining the length of the plane vector.
    norm_plane : array
        The plane normal vector.
    d : float
        Float value defining the fourth plane coordinate. Often referred to as
        d.
    """
    
    x__ = pts - pts.mean(axis=1)[:,np.newaxis]
    M = np.dot(x__, x__.T)
    
    plane = svd(M)[0][:,-1]
    plane_length = np.sqrt(np.sum(plane**2))
    norm_plane = plane / plane_length

    d = np.mean(np.dot(-1*pts.T, norm_plane)) # Fourth plane vector.

    return plane, plane_length, norm_plane, d

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
    plane_lengths : array
        Array containing plane lengths.
    norm_planes : array
        Array containing normal vectors of planes.
    d_coor : array
        Array containing the fourth plane coordinates d.
    """

    # For every point-group in 'pts_groups' compute its plane info.
    planes, plane_lengths, norm_planes, d_coor = [], [], [], []
    for group_idx in pt_groups_idx:

        # Compute plane info.
        pt_group = pts[group_idx].T
        plane, plane_length, norm_plane, d = find_plane(pt_group)

        # Append plane info to corresponding lists.
        planes.append(plane)
        plane_lengths.append(plane_length)
        norm_planes.append(norm_plane)
        d_coor.append(d)

    return planes, plane_lengths, norm_planes, d_coor

def merge_pts_of_similar_planes(pts, pt_groups_idx, pt_group_search_r, 
                                dist_betw_plane_t):
    """
    Function that merges groups of points that have similar planes and are 
    within 'pt_group_search_r' distance of each other.
    Parameters
    ----------
    pts : array
        Array of points.
    pt_groups_idx : array
        Array containing arrays of indices corresponding with points in pts 
        that belong to the same group.
    pt_group_search_r : float
        Search radius used for searching similar planes and to compute whether
        two group points are close to each other or not.
    dist_betw_plane_t : float
        Threshold value defining how far apart two planes are allowed to be, to
        still be considered similar.

    Output
    ------
    merged : array
        Array similar to input 'pt_groups_idx' except some groups have been merged.
    """

    planes, plane_lengths, norm_planes, d_coor = compute_planes(pts, pt_groups_idx)

    # Compute which plane vectors are similar.
    norm_planes = np.asarray(norm_planes)
    norm_tree = KDTree(norm_planes)
    similar_planes = norm_tree.query_radius(norm_planes, r=dist_betw_plane_t)

    merged = []
    for i in range(len(similar_planes)):

        # If 'n_similar_planes' is 0, this was set in a previous iteration. 
        # It means this plane has been merged already.
        n_similar_planes = len(similar_planes[i])
        if n_similar_planes == 0: continue
        else:
            merged.append(pt_groups_idx[i])

            # If 'n_similar_planes' is higher than 1 we know we found another
            # plane which is similar and thus should proceed.
            if n_similar_planes > 1:
                for j in range(0, n_similar_planes):
                    
                    sim_plane_idx = similar_planes[i][j]

                    # Continue if similar plane is the current plane itself.
                    if similar_planes[i][j] == i: continue

                    # Select similar plane point indices and similar plane points
                    sim_plane_pts_idx = pt_groups_idx[sim_plane_idx]
                    sim_plane_pts = pts[sim_plane_pts_idx]
                    
                    # Compute the average distance of the points in the similar plane
                    # to that of the current plane.
                    plane = planes[sim_plane_idx]
                    d = d_coor[sim_plane_idx]
                    plane_l = plane_lengths[sim_plane_idx]
                    dist_betw_plane = np.mean(np.abs(sim_plane_pts@plane+d)/plane_l)

                    # Compute how many points in similar plane are within 
                    # 'pt_group_search_r' distance of any point in current plane.
                    pts_group_tree = KDTree(pts[merged[-1]])
                    closes = pts_group_tree.query_radius(sim_plane_pts, 
                                                         r=pt_group_search_r, 
                                                         count_only=True)
                    near_pts = len(np.asarray(closes>0).nonzero()[0])
                    
                    # If distance between planes is less than 'dist_betw_plane_t'
                    # and there are points from each point-group near each other,
                    # we merge the two point-groups and set the similar planes of
                    # the similar plane to an empty list so it won't be merged again.
                    if dist_betw_plane<dist_betw_plane_t and near_pts>0:
                        merged[-1] = np.append(merged[-1], sim_plane_pts_idx, axis=0)
                        similar_planes[sim_plane_idx] = []

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

        diff = np.sqrt((new_z_min - temp_z_min)**2 + (new_z_max - temp_z_max)**2)

        overlap = len(set(pt_group_idx)&set(new_pt_group_idx[j]))
        overlap_ratio = overlap/len(pt_group_idx)

        # If the current 'max_ratio' is exceeded, i.e., we found a point-group
        # with more overlap, we set this point-group as the new most likely
        # candidate to merge with.
        if overlap_ratio > max_ratio and diff <= 0.03:
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

def plane_search(pts, pt_groups_idx, extra_pts_idx, params):
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
    (close_p_search_r, dist_to_plane_t, neighboors_t, neighboor_search_r, 
    z_margin, overlap_ratio_t, pt_group_search_r, dist_betw_plane_t) = params

    # Merge 1: Merge point-groups that lie within the same plane.
    pt_groups_idx = merge_pts_of_similar_planes(pts, pt_groups_idx,
                                                pt_group_search_r, dist_betw_plane_t)

    # Compute the plane info of each point-group.
    planes, plane_lengths, norm_planes, d_coor = compute_planes(pts, pt_groups_idx)

    new_pt_groups = [] 
    for i, group_idx in enumerate(pt_groups_idx):

        # Do nothing if z-value of normal vector is lower then 0.5 i.e., not vertical.
        if norm_planes[i][2] < 0.5:
            
            # Compute mean of 10 lowest and highest z-values from current point-group.
            group_pts = pts[group_idx]
            sorted_z = np.sort(group_pts[:,2])
            min_z = np.mean(sorted_z[:10])-z_margin
            max_z = np.mean(sorted_z[-10:])+z_margin

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
            dist_to_plane = np.abs(np.dot(close_pts, planes[i])+d_coor[i])/plane_lengths[i]
            z_coor = close_pts[:,2]
            conditions = ((dist_to_plane<dist_to_plane_t) & (z_coor<=max_z) & (z_coor>=min_z))
            pts_in_plane_idx = close_pts_idx[np.asarray(conditions).nonzero()[0]]
            pts_in_plane = pts[pts_in_plane_idx]
                        
            # Filter 3: Find points from 'pts_in_plane_idx' that aren't lonely
            # i.e. points that have at least 'neighboors_t' neighboors wihtin a
            # radius of 'neighboor_search_r'.
            if len(pts_in_plane_idx) == 0: continue
            plane_pts_tree = KDTree(pts_in_plane)
            n_neighboors = plane_pts_tree.query_radius(pts_in_plane, 
                                                       r=neighboor_search_r, 
                                                       count_only=True)
            temp_sub_idx = np.asarray((n_neighboors>=neighboors_t)).nonzero()[0]
            pts_in_plane_idx = pts_in_plane_idx[temp_sub_idx]
            if len(pts_in_plane_idx) == 0: continue

            # Merge 2: Merge newly found points with original point-group.
            plane_indices = np.union1d(group_idx, pts_in_plane_idx)

            # Merge 3: Merge point-groups if overlap-ratio exceeds 'overlap_ratio_t'.
            new_pt_groups = merge_overlapping_groups(pts, plane_indices, new_pt_groups, 
                                                     overlap_ratio_t)
            # new_pt_groups.append(plane_indices)

    return np.asarray(new_pt_groups, dtype=object)