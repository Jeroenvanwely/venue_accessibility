import numpy as np


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
    max_means, min_means, x_mean, y_mean = [], [], [], []
    for v_plane in V_planes:
        v_plane = np.asarray(v_plane, dtype=int)
        sorted_z = sorted(pts[v_plane][:,2])
        max_means.append(np.mean(sorted_z[-25:]))
        min_means.append(np.mean(sorted_z[:25]))
        x_mean.append(np.mean(pts[v_plane][:,0]))
        y_mean.append(np.mean(pts[v_plane][:,1]))

    stats = (np.asarray(max_means), np.asarray(min_means), 
            np.asarray(x_mean), np.asarray(y_mean))
    return stats

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
    print(close_steps)

    stairs, step_heights = [], []
    for steps in close_steps:
        stair, heights = [], []
        min_sorted = np.sort(min_means[steps])

        # The lowest step found may not be higher than the street level height added by 25 cm.
        if min_sorted[0] < street_h+0.25:
            max_sorted = np.sort(max_means[steps])

            for j in range(len(steps)):
                stair += list(V_planes[np.array(steps[j])])
                heights.append(np.round((max_sorted[j]-min_sorted[j])*100, decimals=2))
            
            stairs.append(stair)
            step_heights.append(heights)

    return stairs, step_heights

def find_steps_and_stairs(pts, V_planes, params, street_h):
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
    print(len(V_planes))

    # Compute the statistics of the vertical components.
    stats = get_conn_comp_stats(pts, V_planes)
    max_means, min_means, x_mean, y_mean = stats

    close_steps = []
    n_potential_steps = len(V_planes)
    assigned = np.zeros(n_potential_steps , dtype=int)
    for i in range(n_potential_steps):
        
        # Find vertical plane close enough to current vertical plane
        # such that they could belong to the same stair.
        conditions = ((min_means < max_means[i]+step_min_range) & 
                     (min_means > max_means[i]-step_min_range) &
                     (np.sqrt((x_mean[i]-x_mean)**2+(y_mean[i]-y_mean)**2)<step_max_range) &
                     (x_mean[i]-x_mean != 0))
        near_potential_step = np.asarray(conditions).nonzero()[0]

        n_near_steps = len(near_potential_step)
        if n_near_steps > 0:

            near_potential_step = near_potential_step[0]
            placed = False # Set 'placed' switch to False.
            for j in range(len(close_steps)):
                
                if i in close_steps[j] and near_potential_step in close_steps[j]:
                    placed = True
                # If potential step 'i' has already been assigned to a stair,
                # 'near_potential_step' should also be assigned to this stair.
                elif i in close_steps[j]:
                    placed, assigned[near_potential_step] = True, 1
                    close_steps[j].append(near_potential_step)

                # If 'near_potential_step' has already been assigned to a stair,
                # potential step 'i' should also be assigned to this stair.
                elif near_potential_step in close_steps[j]:
                    placed, assigned[i] = True, 1
                    close_steps[j].append(i)
            
            # If 'placed' switch has not been set to True. i.e., If both potential 
            # step 'i' and 'near_potential_step' are not already assigned to a stair. 
            # Create new stair and assign both potential step 'i' and 
            # 'near_potential_step' to it.
            if placed == False:
                assigned[i], assigned[near_potential_step] = 1, 1
                close_steps.append([i, near_potential_step])

        # If no near step is found, we add new stair containing only
        # potential step 'i' as a step.
        elif assigned[i] == 0:
            assigned[i] = 1
            close_steps.append([i])

    # Filter the potential steps and return the found steps and their heights.
    stairs, step_heights = filter_steps_stairs(V_planes, close_steps, min_means, 
                                               max_means, street_h)

    return stairs, step_heights