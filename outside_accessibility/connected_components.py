import numpy as np
from sklearn.neighbors import KDTree

def connected_components(pts, poi_idx, dist_t=0.1):
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
                    # If the neighbor is unlabeled.
                    if labels[j] == 0:
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