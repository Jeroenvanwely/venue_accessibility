import numpy as np
import copy

def get_conn_comp_stats(pts, V_planes):
    """
    Function that, given sets of vertical planes, returns its statistics

    Parameters
    ----------
    pts : array
        Array of points.
    V_planes : array
        An array containing arrays. Each sub-array is representative of a vertical 
        plane and contains the indices of the points that lie within its plane.

    Output
    ------
    stats : tuple
        Tuple containing:
        max_means: Array containing mean of highest 25 z-values for each vertical plane.
        min_means: Array containing mean of lowest 25 z-values for each vertical plane.
        x_mean: Array containing mean value of the x-coordinate for each vertical plane.
        y_mean: Array containing mean value of the y-coordinate for each vertical plane.
    """
    max_means, min_means, x_mean, y_mean, xy_median = [], [], [], [], []
    for v_plane in V_planes:
        v_plane = np.asarray(v_plane, dtype=int)
        sorted_z = sorted(pts[v_plane][:,2])
        max_means.append(np.mean(sorted_z[-25:]))
        min_means.append(np.mean(sorted_z[:25]))
        x_mean.append(np.mean(pts[v_plane][:,0]))
        y_mean.append(np.mean(pts[v_plane][:,1]))
        xy_median.append(np.mean(pts[v_plane][:,0:2], axis=0))
        

    stats = (np.asarray(max_means), np.asarray(min_means), 
            np.asarray(x_mean), np.asarray(y_mean), np.asarray(xy_median))
    print(np.asarray(xy_median))
    return stats

def find_normal(pts):
    W, V = np.linalg.eig([np.cov(pts.T)])
    return V[0][np.argsort(W[0])[0]]

def filter_steps_stairs(V_planes, close_steps, min_means, max_means, street_h):
    """
    A function that filters out the false steps from sets of potential steps and stairs

    Parameters
    ----------
    V_planes : array
        An array containing arrays. Each sub-array is representative of a vertical 
        plane and contains the indices of the points that lie within its plane.
    close_steps : array
        An array containing arrays of indices corresponding with vertical planes in V_plane.
        An array containing a single index implies the corresponding vertical plane has no
        other vertical planes near that could be considered as step and thus that we might 
        have a single step here. An array containing multiple indices implies the corresponding 
        vertical planes are close enough to each other te be considered consecutive parts of a stair.
    min_means : array
        Array containing mean of lowest 25 z-values for each vertical plane in V_planes.
    max_means : array
        Array containing mean of highest 25 z-values for each vertical plane in V_planes.
    street_h : float
        Height of the street/road/ground.
    Output
    ------
    stairs : array
        An array containing arrays of indices corresponding with planes in V_planes 
        that form either a step (single step) or a stair (multiple steps).
    step_heights : array
        An array containing step heights. Its shape is equal to that of the stairs array.
    """

    stairs, step_heights = [], []
    for steps in close_steps:
        stair, heights = [], []
        min_sorted = np.sort(min_means[steps])

        # The lowest step found may not be higher than the street level height added by 25 cm.
        # if min_sorted[0] < street_h+0.1 and min_sorted[0] >= street_h-0.1:
        if min_sorted[0] < street_h+0.1:
            max_sorted = np.sort(max_means[steps])

            for j in range(len(steps)):
                stair.append(list(V_planes[np.array(steps[j])]))
                heights.append(np.round((max_sorted[j]-min_sorted[j])*100, decimals=2))
            
            stairs.append(stair)
            step_heights.append(heights)

    return stairs, step_heights
from scipy.spatial.distance import cdist
def find_steps_and_stairs(pts, V_planes, params, street_h, bbox_list):
    """
    A function that finds steps and stairs given sets of points corresponding 
    to vertical planes.

    Parameters
    ----------
    pts : array
        Array of points.
    V_planes : array
        An array containing arrays. Each sub-array is representative of a vertical 
        plane and contains the indices of the points that lie within its plane.
    params : tuple
    Tuple of parameters containing:
        step_min_range : float
            Threshold value defining the maximum allowed difference between the maximum 
            height of the current step and the minimum height of the next step to still 
            consider the steps as consecutive parts of a stair.
        step_max_range : float
            Threshold value defining the maximum distance between the center points 
            of two steps in the x,y-plane to still consider the steps as a 
            consecutive parts of a stair.
    street_h : float
        Height of the street/road/ground.

    Output
    ------
    stairs : array
        An array containing arrays of indices corresponding with planes in V_planes 
        that form either a step (single step) or a stair (multiple steps).
    step_heights : array
        An array containing step heights. Its shape is equal to that of the stairs array.
    """
    step_min_range, step_max_range = params

    # Compute the statistics of the vertical components.
    stats = get_conn_comp_stats(pts, V_planes)
    max_means, min_means, x_mean, y_mean, xy_median = stats
    median_diff = cdist(xy_median, xy_median)

    close_steps = []
    n_potential_steps = len(V_planes)
    group_number_assignment = np.zeros((n_potential_steps), dtype=int)
    group_num = 1

    for i in range(n_potential_steps):
        
        # # Find vertical plane close enough to current vertical plane
        # # such that they could belong to the same stair.
        # conditions = ((min_means < max_means[i]+step_min_range) & 
        #              (min_means > max_means[i]-step_min_range) &
        #              (np.sqrt((x_mean[i]-x_mean)**2+(y_mean[i]-y_mean)**2)<step_max_range) &
        #              (x_mean[i]-x_mean != 0))
        # near_potential_step = np.asarray(conditions).nonzero()[0]

        difference = np.abs(bbox_list - bbox_list[i])
        x_diff = np.abs(difference[:,:,0])
        y_diff = np.abs(difference[:,:,1])
        height_diff = difference[:,:,2]
        max_x_diff = np.max(x_diff, axis=1)
        max_y_diff = np.max(y_diff, axis=1)
        max_height_diff = np.max(height_diff, axis=1)
        median_diff[i]

        condition = ((max_height_diff > 0) & (max_height_diff<1.) & (max_x_diff<1.2) & (max_y_diff<1.2))
        near_potential_step = np.squeeze(np.argwhere(condition), axis=1)

        if len(near_potential_step) == 0:
            group_number_assignment[i] = group_num
            group_num += 1
        else:
            for step_idx in near_potential_step:
                if group_number_assignment[step_idx] == 0 and group_number_assignment[i] == 0:
                    group_number_assignment[step_idx], group_number_assignment[i] = group_num, group_num
                    group_num += 1
                elif group_number_assignment[step_idx] != 0 and group_number_assignment[i] == 0:
                    group_number = group_number_assignment[step_idx]
                    group_number_assignment[i] = group_number
                elif group_number_assignment[step_idx] == 0 and group_number_assignment[i] != 0:
                    group_number = group_number_assignment[i]
                    group_number_assignment[step_idx] = group_number
                elif group_number_assignment[step_idx] != 0 and group_number_assignment[i] != 0 and group_number_assignment[step_idx] != group_number_assignment[i]:
                    group_number = group_number_assignment[step_idx]
                    group_number_assignment[np.squeeze(np.argwhere((group_number_assignment == group_number)), axis=1)] = group_number_assignment[i]
    
    groups = np.unique(group_number_assignment)
    close_steps = []
    for group_number in groups:
        close_steps.append(np.squeeze(np.argwhere((group_number_assignment == group_number)), axis=1))


    # Filter the potential steps and return the found steps and their heights.
    stairs, step_heights = filter_steps_stairs(V_planes, close_steps, min_means, 
                                               max_means, street_h)

    stairs2 = []
    step_heights2 = []
    for i, stair in enumerate(stairs):
        new_stair = []
        new_step_height = []
        if len(stair) == 0: continue
        if len(stair) == 1:
            stairs2.append([stair[0]])
            step_heights2.append([step_heights[i][0]])
            continue

        normals = []
        for j in range(len(stair)):
            normals.append(find_normal(pts[stairs[i][j]]))
        normals = np.array(normals, dtype=float)

        similarities = np.dot(normals, normals.T)
        lengths_vec = np.sqrt(np.sum(normals**2, axis=1))[np.newaxis]
        lengths_mat = np.dot(np.transpose(lengths_vec),  lengths_vec)
        sim = np.abs(similarities/lengths_mat)
        for j, row in enumerate(sim):
            if len(np.where((row<0.5))[0]) > np.floor(len(similarities)/2)+1:
                continue
            else:
                new_stair.append(stairs[i][j])
                new_step_height.append(step_heights[i][j])

        stairs2.append(new_stair)
        step_heights2.append(new_step_height)
    # return [], []
    return stairs2, step_heights2
    return [stairs2[0]], [step_heights2[0]]

    return stairs, step_heights