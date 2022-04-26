import numpy as np
import open3d as o3d

import las_utils

from plane_search import plane_search
from raster_search import find_potential_step_points
from step_finder import find_steps_and_stairs
from connected_components import connected_components
from filter_point_areas import filter_pt_areas
from visualization import visualize



####################
# Tweede tile doet het minder goed door orientatie op de x en y as
# Mogelijk maakt roteren voor detectie verbeteringen.
####################

coor_list = [
            [119877.496,485266.533,119890.32,485254.402],
             [119871.711,485269.732,119883.878,485257.306],
             [119868.123,485272.814,119878.091,485258.847],
             [119859.817,485273.89,119872.515,485259.386],
             [119878.251,485259.371,119892.559,485248.155],
             [119878.761,485254.049,119894.988,485242.584],
             [119342.628,485155.657,119355.247,485141.759],
             [119302.238,485150.897,119315.189,485136.658],
             [119293.44, 485131.159, 119316.461, 485108.919]]
counter = 0
for  x_min, y_max, x_max, y_min in coor_list:
    
    print("\nVenue {}:".format(counter+1))
    
    if counter <= 5: tilecode = '2397_9705'
    else: tilecode = '2386_9702'
    counter += 1

    # Normal calculation params
    max_nn = 25
    small_radius = 0.08
    V_norm_t = 0.3
    # H_norm_t = 0.9

    # Connected components params
    V_dist_t = 0.1
    # H_dist_t = 0.08

    # Step finder params
    step_min_range = 0.3
    step_max_range = 0.8
    step_finder_params = (step_min_range, step_max_range)

    # Raster search params.
    steps = [0.02, 0.04, 0.06, 0.08]
    min_h_dif = 0.01
    max_h_dif = 0.3
    raster_search_params = (steps, min_h_dif, max_h_dif)

    # Filter point areas params.
    width_t = 0.4
    height_t = 0.35
    n_pts_t = 200
    filt_pt_areas_params = (width_t, height_t, n_pts_t)

    # plane search params.
    close_p_search_r = 1.5
    dist_to_plane_t = 0.06
    neighboors_t = 20
    neighboor_search_r = 0.3
    z_margin = 0.1
    overlap_ratio_t = 0.1
    pt_group_search_r = 0.2
    dist_betw_plane_t = 0.5
    plane_search_params = (close_p_search_r, dist_to_plane_t, neighboors_t, 
                        neighboor_search_r, z_margin, overlap_ratio_t, 
                        pt_group_search_r, dist_betw_plane_t)

    # Import laz file
    in_file = 'Data/Pointcloud/filtered_' + tilecode + '.laz'
    las = las_utils.read_las(in_file)

    # Get indices of points corresponding to first floor of venue
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    condition = ((x>=x_min) & (x<=x_max) & (y>=y_min) & (y<=y_max) & (z<3.0))
    venue_indices = np.asarray(condition).nonzero()[0]

    # Center venue points such that minimum value for each axis is zero
    x = x[venue_indices] - np.min(x[venue_indices])
    y = y[venue_indices] - np.min(y[venue_indices])
    z = z[venue_indices] - np.min(z[venue_indices])

    # Create Open3D venue pointcloud
    points = np.stack((x, y, z), axis=-1)
    venue_pcd = o3d.geometry.PointCloud()
    venue_pcd.points = o3d.utility.Vector3dVector(points)

    # Remove outliers from the venue pointcloud
    venue_pcd, ind = venue_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=0.5)

    # Update venue_indices and points/point coordinates
    venue_indices = venue_indices[ind]
    points, x, y, z = points[ind], x[ind], y[ind], z[ind]

    # Compute normal vectors of each point in cloud based on 'max_nn' nearest 
    # neighours found in small radius
    search_param = o3d.geometry.KDTreeSearchParamHybrid(small_radius, max_nn)
    venue_pcd.estimate_normals(search_param)
    V_norm_z = np.abs(np.asarray(venue_pcd.normals)[:,2])

    # Compute normal vectors of each point in cloud based on 'max_nn' nearest 
    # neighours found in large radius
    # search_param = o3d.geometry.KDTreeSearchParamHybrid(big_radius, max_nn=max_nn)
    # venue_pcd.estimate_normals(search_param)
    # H_norm_z = np.abs(np.asarray(venue_pcd.normals)[:,2])

    # Devide points into vertical, tilted and horizontal categories
    V_idx = np.asarray((V_norm_z<=V_norm_t)).nonzero()[0]
    # I_idx = np.asarray((V_norm_z>V_norm_t) & (V_norm_z<H_norm_t)).nonzero()[0]
    # H_idx = np.asarray((H_norm_z>=H_norm_t)).nonzero()[0]

    # Find points of interest within points classified as vertical
    V_step_poi = find_potential_step_points(points, V_idx, raster_search_params)

    # Find points of interest within all venue points
    step_poi = find_potential_step_points(points, None, raster_search_params)

    # Find connected components
    # H_con_comps = connected_components(points, H_dist_t, H_idx)

    V_con_comps = connected_components(points, V_step_poi, V_dist_t)
    V_con_comps = plane_search(points, V_con_comps, step_poi, plane_search_params)
    V_con_comps = filter_pt_areas(V_con_comps, points, filt_pt_areas_params)

    stairs, step_heights = None, None
    street_height = np.mean(np.partition(z,3000)[:3000])
    stairs, step_heights = find_steps_and_stairs(points, V_con_comps, 
                                                step_finder_params, street_height)
    # stairs, step_heights = None, None
    visualize(las, venue_indices,V_con_comps, venue_pcd, stairs, step_heights)