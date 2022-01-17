import numpy as np
from sklearn.neighbors import KDTree

def filter_scater_areas(pts):
    tree = KDTree(pts)
    close_pt_count = tree.query_radius(pts, r=0.1, count_only=True)
    ratio = len(np.where(close_pt_count>30)[0]) / len(pts)
    if ratio < 0.7:
        return False
    return True
    

def filter_pt_areas(plane_pt_groups_idx, pts, params):
    """
    Function to filter out small areas from given set of connected components.

    Parameters
    ----------
    connected_components : array
        Array containing arrays of indices corresponding with points belonging 
        to same connected components area.
    points : array
        Array of points.
    params : tuple
        Tuple of parameters containing:
            width_t : float
                width_t is a threshold value. If an area is less wide than 'width_t', we assume
                the area is not meant to be crossed by humans and therefore can not be a ramp,
                step, door, etc.
            height_t : float
                height_t is a threshold value. If the area-height of an area formed by a set
                of points is higher than 'height_t', we assume it is too high to be a step.
            n_pts_t : int
                Minimum number of points within an area. Areas with too few points are
                too small or too sparse to classify.

    Output
    ------
    connected_components_to_return : array
        Filtered array containing arrays of indices corresponding with points belonging 
        to connected components areas.
    """
    width_t, height_t, n_pts_t = params

    filt_plane_pt_groups_idx = []
    for pt_group in plane_pt_groups_idx:
        if not filter_scater_areas(pts[pt_group]): continue
        # Remove if area contains less than 'n_points' points.
        if len(pt_group) < n_pts_t: continue
        else:
            x_coor, y_coor, z_coor = pts[pt_group,0], pts[pt_group,1], pts[pt_group,2]
            x_min, x_max = np.min(x_coor), np.max(x_coor)
            y_min, y_max = np.min(y_coor), np.max(y_coor)
            z_sorted = np.sort(z_coor)
            z_min, z_max = np.mean(z_sorted[:25]), np.mean(z_sorted[-25:])

            # We add the point-group to the filtered plane point group indices list if the 
            # area-width exceeds 'width_t' and area-height exceeds 'height_t'.
            pt_group_width = np.sqrt((x_max-x_min)**2+(y_max-y_min)**2)
            pt_height_diff = z_max-z_min
            if pt_group_width > width_t and pt_height_diff < height_t:
                filt_plane_pt_groups_idx.append(pt_group)
    
    return np.asarray(filt_plane_pt_groups_idx, dtype=object)