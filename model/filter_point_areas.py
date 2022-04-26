import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d
from scipy.spatial import distance

def filter_scater_areas(pts, idx):
    tree = KDTree(pts[idx])
    close_pt_count = tree.query_radius(pts[idx], r=0.1, count_only=True)
    return idx[np.where(close_pt_count>5)]

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
    width_t, n_pts_t, max_height_diff = params

    filt_plane_pt_groups_idx, bbox_list = [], []
    for pt_group_idx in plane_pt_groups_idx:
        # Remove if area contains less than 'n_points' points.
        if len(pt_group_idx) < n_pts_t: continue
        else:
            pt_group = pts[pt_group_idx]
            
            z_coor = pt_group[:,2]
            z_sorted = np.sort(z_coor)
            z_min, z_max = np.median(z_sorted[:10]), np.median(z_sorted[-10:])
            height_diff = z_max-z_min

            bbox_class = o3d.geometry.OrientedBoundingBox()
            o3d_bbox = bbox_class.create_from_points(o3d.utility.Vector3dVector(pt_group))
            bbox = np.asarray(o3d_bbox.get_box_points())
            # bbox_xy = bbox[:,0:2]
            dists = distance.cdist(bbox, bbox)
            # pt_group_width = np.max(dists)
            lines = [[0, 1], [0,2], [0,3], [2,5], [5,3], [5,4], [2,7], [3,6], [6,4], [6,1], [1,7], [4,7]]
            distances = []
            for (p1, p2) in lines:
                distances.append(dists[p1][p2])
            distances = np.sort(np.array(distances))
            pt_group_width = distances[-1]
            pt_group_length = distances[-5]
            width_length_ratio = pt_group_width/pt_group_length


            # We add the point-group to the filtered plane point group indices list if the 
            # area-width exceeds 'width_t' and area-height exceeds 'height_t'.
            if pt_group_width > width_t and height_diff < max_height_diff and width_length_ratio>1.5:
                filt_plane_pt_groups_idx.append(np.array(pt_group_idx, dtype=int))
                bbox_list.append(bbox)
    
    # list_to_return = []
    # bbox_to_return = []
    # for i, group in enumerate(filt_plane_pt_groups_idx):
    #     new = np.array(filter_scater_areas(pts, group), dtype=int)
    #     if len(new) > 0:
    #         list_to_return.append(new)
    #         bbox_to_return.append(bbox_list[i])
    # return list_to_return, bbox_to_return

    return np.asarray(filt_plane_pt_groups_idx, dtype=object), bbox_list
