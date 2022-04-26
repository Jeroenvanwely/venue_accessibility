import numpy as np
from sklearn.neighbors import KDTree

from model.normal_estimation import cosine_similarity

def connected_components(pts, poi_idx, normals, dist_t=0.1):
    """
    Connected components algorithm for finding points that are clustered 
    together by either being a direct or indirect neighboor to the other 
    points within the cluster.

    Parameters
    ----------
    pts : array
        Array of points of shape "number of points" X "dimension" 
        (can be any dimension).
    poi_idx : array
        Array containing indices of the points of interest. All other points
        will not be considered within this function.
    dist_t : float (default: 0.1)
        Distance threshold that defines the maximum distance two 
        points can be set apart to be still considered neighbors.

    Output
    ------
    connected_components : array
        Array containing arrays of indices corresponding with points belonging 
        to the same connected components area.
    """
    
    pts = pts[poi_idx] # Select the points of interest.
    number_of_pts = len(pts)
    
    # Compute neighbors of each point using KDTree.
    tree = KDTree(pts)
    neighbor_groups = tree.query_radius(pts, r=dist_t)

    # 'unlabeled' and 'labeled are arrays of indices corresponding to 
    # the points that are unlabeled and labeled, respectively.
    labeled, unlabeled = np.array([]), np.arange(0, number_of_pts, 1)

    # 'all_idx' is a list of indices of all points.
    all_idx = np.arange(0, number_of_pts, 1)

    # 'labels' is an array of the same length as the number of points
    # 1 means the corresponding point has been assigned to an area, 0
    # means it has not been assigned yet.
    labels = np.zeros(number_of_pts)
    
    connected_components = []
    while len(unlabeled) > 0:
        
        # Initialize new connected components area, i.e. select the
        # next in line unlabeled point and find its neighbors.
        pt_idx = unlabeled[0]
        neighbors_idx = list(neighbor_groups[pt_idx])
        labels[neighbors_idx] = 1 # Set points to labeled.
        
        prev_length = 0 # Set prev_length to zero.
        
        # Repeat until the length of 'neighbors_idx' remains unchanged,
        # i.e. until no new points are added to current connected components
        # are.
        while prev_length != len(neighbors_idx):
            
            # Save current number of points within 'neighbors_idx' list.
            cur_length = len(neighbors_idx)

            # Start looping over the new points added in previous iteration.
            for i in range(prev_length-1, len(neighbors_idx)):

                # For each newly added point, get its neighbors.
                group = neighbor_groups[neighbors_idx[i]]

                # For each neighbor of newly added point.
                for j in group:

                    # Filter out points that do not have similar normal
                    similarity = cosine_similarity(normals[neighbors_idx[i]], normals[j])
                    # If the neighbor is unlabeled.
                    if labels[j] == 0 and similarity > 0.99:

                        # Add neighbor index to 'neighbor_indices' and set too labeled.
                        neighbors_idx.append(j)
                        labels[j] = 1

            # Set 'prev_length' to 'cur_length' value which was obtained before 
            # adding the new points.
            prev_length = cur_length

        # Add 'neighbors_idx' as connected component area to list of connected
        # components.
        connected_components.append(poi_idx[neighbors_idx])

        # Update 'labeled' and 'unlabeled'.
        labeled = np.concatenate((labeled, neighbors_idx), axis=0)
        unlabeled = np.delete(all_idx, np.asarray(labeled, dtype=int))

    return np.asarray(connected_components, dtype=object)



def con_comps_helper(pts, new_pts_idx, group_pts_idx):
    # Compute neighbors of each point using KDTree.
    tree = KDTree(pts[group_pts_idx])
    if len(new_pts_idx) <= 3 or len(group_pts_idx) <= 3:
        return new_pts_idx, group_pts_idx
    new_pts_neighbor_count = tree.query_radius(pts[new_pts_idx], r=0.05, count_only=True)
    near_pts = np.asarray(new_pts_neighbor_count>0).nonzero()[0]
    new_pts_idx = np.asarray(new_pts_neighbor_count==0).nonzero()[0]
    group_pts_idx = np.union1d(group_pts_idx, near_pts)
    return new_pts_idx, group_pts_idx

def con_comp(pts, new_pts_idx, group_pts_idx):
    
    prev_group_pts_idx = group_pts_idx
    new_pts_idx, group_pts_idx = con_comps_helper(pts, new_pts_idx ,group_pts_idx)
    
    while len(prev_group_pts_idx) != len(group_pts_idx):
        new_pts_idx, group_pts_idx = con_comps_helper(pts, new_pts_idx, group_pts_idx)
    return new_pts_idx, group_pts_idx



def connected_components2(pts, poi_idx, group_pts_idx, normals, plane_normal, dist_t=0.1):
    """
    Connected components algorithm for finding points that are clustered 
    together by either being a direct or indirect neighboor to the other 
    points within the cluster.

    Parameters
    ----------
    pts : array
        Array of points of shape "number of points" X "dimension" 
        (can be any dimension).
    poi_idx : array
        Array containing indices of the points of interest. All other points
        will not be considered within this function.
    dist_t : float (default: 0.1)
        Distance threshold that defines the maximum distance two 
        points can be set apart to be still considered neighbors.

    Output
    ------
    connected_components : array
        Array containing arrays of indices corresponding with points belonging 
        to the same connected components area.
    """
    
    poi_idx = np.union1d(poi_idx, group_pts_idx) # merge the points.
    pts = pts[np.array(poi_idx, dtype=int)] # Select the points of interest.
    number_of_pts = len(pts)
    
    # Compute neighbors of each point using KDTree.
    tree = KDTree(pts)
    # neighbor_groups = tree.query_radius(pts, r=0.03)
    neighbor_groups = tree.query_radius(pts, r=dist_t)

    # 'unlabeled' and 'labeled are arrays of indices corresponding to 
    # the points that are unlabeled and labeled, respectively.
    labeled, unlabeled = np.array([]), np.arange(0, number_of_pts, 1)

    # 'all_idx' is a list of indices of all points.
    all_idx = np.arange(0, number_of_pts, 1)

    # 'labels' is an array of the same length as the number of points
    # 1 means the corresponding point has been assigned to an area, 0
    # means it has not been assigned yet.
    labels = np.zeros(number_of_pts)
    
    # connected_components = []
        
    # Initialize new connected components area, i.e. select the
    # next in line unlabeled point and find its neighbors.
    neighbors_idx = np.where((poi_idx in group_pts_idx))[0]
    neighbors_idx = [i for i, x in enumerate(poi_idx) if x in group_pts_idx]
    labels[neighbors_idx] = 1 # Set points to labeled.
    labeled = np.concatenate((labeled, neighbors_idx), axis=0)
    unlabeled = np.delete(all_idx, np.asarray(labeled, dtype=int))
    
    prev_length = 0 # Set prev_length to zero.
    
    # Repeat until the length of 'neighbors_idx' remains unchanged,
    # i.e. until no new points are added to current connected components
    # are.
    while prev_length != len(neighbors_idx):
        
        # Save current number of points within 'neighbors_idx' list.
        cur_length = len(neighbors_idx)

        # Start looping over the new points added in previous iteration.
        for i in range(prev_length-1, len(neighbors_idx)):

            # For each newly added point, get its neighbors.
            group = neighbor_groups[neighbors_idx[i]]

            # For each neighbor of newly added point.
            for j in group:

                similarity = cosine_similarity(plane_normal, normals[j])
                # If the neighbor is unlabeled.
                if labels[j] == 0 and similarity > 0.99:
                    # Add neighbor index to 'neighbor_indices' and set too labeled.
                    neighbors_idx.append(j)
                    labels[j] = 1

        # Set 'prev_length' to 'cur_length' value which was obtained before 
        # adding the new points.
        prev_length = cur_length

    # Add 'neighbors_idx' as connected component area to list of connected
    # components.
    # connected_components.append(poi_idx[neighbors_idx])

    # Update 'labeled' and 'unlabeled'.
    labeled = np.concatenate((labeled, neighbors_idx), axis=0)
    unlabeled = np.delete(all_idx, np.asarray(labeled, dtype=int))

    return poi_idx[unlabeled], poi_idx[neighbors_idx]


def connected_components3(pts, poi_idx, groups, dist_t=0.15):
    """
    Connected components algorithm for finding points that are clustered 
    together by either being a direct or indirect neighboor to the other 
    points within the cluster.

    Parameters
    ----------
    pts : array
        Array of points of shape "number of points" X "dimension" 
        (can be any dimension).
    poi_idx : array
        Array containing indices of the points of interest. All other points
        will not be considered within this function.
    dist_t : float (default: 0.1)
        Distance threshold that defines the maximum distance two 
        points can be set apart to be still considered neighbors.

    Output
    ------
    connected_components : array
        Array containing arrays of indices corresponding with points belonging 
        to the same connected components area.
    """
    prev_len = 0
    groups = groups.tolist()
    while len(poi_idx) > 0 and len(poi_idx) != prev_len:
        prev_len = len(poi_idx)
        for i, group_pts_idx in enumerate(groups):

            if i == 0:
                classes = np.zeros(len(group_pts_idx), dtype=int) + i
                all_idx = group_pts_idx
            else:
                new_classes = np.zeros(len(group_pts_idx), dtype=int) + i
                classes = np.concatenate((classes, new_classes), axis=None)
                all_idx = np.concatenate((all_idx, group_pts_idx), axis=None)

        poi_pts = pts[np.array(poi_idx, dtype=int)] # Select the points of interest.
        group_pts = pts[np.array(all_idx, dtype=int)]
        
        # Compute neighbors of each point using KDTree.
        tree = KDTree(group_pts)
        neighbor_groups = tree.query_radius(poi_pts, r=dist_t)

        removed_count = 0
        for i, neighbors in enumerate(neighbor_groups):
            if len(neighbors) == 0: continue
            groups[classes[neighbors[0]]] = np.array(np.append(groups[classes[neighbors[0]]], poi_idx[i-removed_count]),dtype=int)
            poi_idx = np.delete(poi_idx, i-removed_count)
            removed_count += 1


    return np.array(groups, dtype=object)