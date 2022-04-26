import numpy as np
import open3d as o3d
import copy
import csv
import ast

from model.plane_search import merge_connected_components, plane_search
from model.raster_search import find_potential_step_points
from model.step_finder import find_steps_and_stairs
from model.connected_components import connected_components
from model.filter_point_areas import filter_pt_areas
from model.visualization import visualize
from model.normal_estimation import compute_normal_estimates
from write_labels_func import label_data
from convert_to_las import write_to_las

from sklearn.neighbors import KDTree

def steps_and_stairs(dir):

    # Normal calculation params
    V_norm_t = 0.5

    # Connected components params
    V_dist_t = 0.15

    # Step finder params
    step_min_range = 0.13
    step_max_range = 2
    step_finder_params = (step_min_range, step_max_range)

    # Raster search params.
    # steps = [0.03, 0.05, 0.1, 0.15, 0.25, 0.3]
    steps = [0.1, 0.15, 0.25, 0.3]
    min_h_dif = 0.05
    max_h_dif = 0.3
    raster_search_params = (steps, min_h_dif, max_h_dif)

    # Filter point areas params.
    width_t = 0.4
    n_pts_t = 30
    max_height_diff = 0.45
    filt_pt_areas_params = (width_t, n_pts_t, max_height_diff)

    # plane search params.
    close_p_search_r = 2
    dist_to_plane_t = 0.1
    z_margin = 0.15
    overlap_ratio_t = 0.01
    plane_search_params = (close_p_search_r, dist_to_plane_t, z_margin, overlap_ratio_t)

    venue_pcd = o3d.io.read_point_cloud(dir.replace("'", "").replace("\n" ,""))

    # Remove outliers from the venue pointcloud
    venue_pcd, _ = venue_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=0.5)
    rgb = copy.deepcopy(np.asarray(venue_pcd.colors))
    points = np.array(venue_pcd.points)

    # Compute normal vectors of each point in cloud based on 'max_nn' nearest 
    # neighours found in small radius
    # neighbor_radius, distance_treshold = .08, .025
    neighbor_radius, distance_treshold = .08, .025
    normals, NN = compute_normal_estimates(points, neighbor_radius, distance_treshold)
    V_norm_x = np.abs(normals[:,0])
    V_norm_y = np.abs(normals[:,1])
    V_norm_z = np.abs(normals[:,2])

    # Devide points into vertical, tilted and horizontal categories
    V_idx = np.asarray((V_norm_z<=V_norm_t) & (((V_norm_x == None)&(V_norm_y == None)&(V_norm_z == None))==False)).nonzero()[0]
    not_H_idx = np.asarray((V_norm_z>V_norm_t) & (V_norm_z<0.65) & (((V_norm_x == None)&(V_norm_y == None)&(V_norm_z == None))==False)).nonzero()[0]
    
    # Find points of interest within points classified as vertical
    V_step_poi = find_potential_step_points(points, V_idx, raster_search_params)

    stairs, step_heights = None, None

    KDtree = KDTree(points[V_step_poi])
    NN = KDtree.query_radius(points[V_step_poi], r=0.05)

    temp = []
    for i in range(len(V_step_poi)):
        if len(NN[i]) >= 0: 
            temp.append(V_step_poi[i])

    V_step_poi = np.array(temp, dtype=int)

    V_con_comps = connected_components(points, V_step_poi, normals, V_dist_t)

    V_con_comps = merge_connected_components(points, V_con_comps)
    
    step_poi = find_potential_step_points(points, np.arange(0, len(points)), raster_search_params)
    step_poi = np.array(list((set(step_poi)&set(not_H_idx))))

    KDtree = KDTree(points[step_poi])
    NN = KDtree.query_radius(points[step_poi], r=0.06)
    temp = []
    for i in range(len(step_poi)):
        if len(NN[i]) >= 0: 
            temp.append(step_poi[i])
    step_poi = np.array(temp, dtype=int)

    V_con_comps = plane_search(points, V_con_comps, step_poi, plane_search_params, normals)
    V_con_comps = merge_connected_components(points, V_con_comps)
    V_con_comps, bbox_list = filter_pt_areas(V_con_comps, points, filt_pt_areas_params)
    if len(V_con_comps) > 0:
        ground_h = np.median(np.sort(points[:,2])[300])
        stairs, step_heights = find_steps_and_stairs(points, V_con_comps, 
                                                step_finder_params, ground_h, bbox_list)
                            
    else:
        stairs, step_heights = [], []
    labels = np.zeros(len(points))
    label_counter = 1
    for stair in stairs:
        for step in stair:
            labels[step] = label_counter
            label_counter += 1

    venue_pcd, bbox = visualize(venue_pcd, V_con_comps, stairs, step_heights)
    
    return venue_pcd, bbox, normals, labels, rgb


with open('last_venue_number.csv', 'r', encoding='UTF8', newline='') as num:
    reader = csv.reader(num)
    last_venue_number = int(next(reader)[0])

with open('adequite_venue_pcd.csv', 'r') as BGT_reader:
    for i, row in enumerate(BGT_reader):

        if i<last_venue_number: continue

        print("Detection results venue {}:".format(i))

        row = row.split(',')
        dir = row[1]

        pcd, bbox_list, normals, labels, rgb = steps_and_stairs(dir)
        xyz = np.array(pcd.points)
        rgb = rgb*(255**2)
        file_name = "las_files/" + dir.split("/")[1][0:-4] + "las"

        answer = input("Is the detection correct?\n")
        while answer != 'y' and answer != 'n':
            answer = input("Your answer should be 'y' or 'n'.\n")
        if answer == 'y':
            label_data(dir, bbox_list)
            with open('last_venue_number.csv', 'w', encoding='UTF8', newline='') as num:
                num_csv_writer = csv.writer(num)
                num_csv_writer.writerow([i+1])
            write_to_las(xyz, rgb, labels, normals, file_name)
        else:
            print("Change settings so label is correct.")



# total_steps = 0
# true_positives = 0
# false_positives = 0
# true_negative = 0
# false_negative = 0

# with open('test_data.csv', 'r') as BGT_reader:
#     for i, row in enumerate(BGT_reader):
#         if i < 15: continue
#         print("Detection results venue {}:".format(i))
#         row = row.split(',')
#         dir = row[1]
#         label = ast.literal_eval(row[2].replace(' ', ','))
#         stairs, step_heights = steps_and_stairs(dir)

#         total_steps += np.sum(label)

#         step_count = []
#         for stair in step_heights:
#             step_count.append(len(stair))

#         if len(label) == len(step_count):
#             label = np.sort(label)
#             step_count = np.sort(step_count)
#             for j in range(len(label)):
#                 true_step_count = label[j]
#                 estimate_step_count = step_count[j]
#                 if true_step_count == estimate_step_count:
#                     true_positives += true_step_count
#                 elif true_step_count > estimate_step_count:
#                     true_positives += estimate_step_count
#                     false_negative += true_step_count - estimate_step_count
#                 elif true_step_count < estimate_step_count:
#                     true_positives += true_step_count
#                     false_positives += estimate_step_count - true_step_count

#         elif len(label) > len(step_count):
#             label = np.sort(label)
#             step_count = np.sort(step_count)
#             j = 0
#             for j in range(len(step_count)):
#                 true_step_count = label[j]
#                 estimate_step_count = step_count[j]
#                 if true_step_count == estimate_step_count:
#                     true_positives += true_step_count
#                 elif true_step_count > estimate_step_count:
#                     true_positives += estimate_step_count
#                     false_negative += true_step_count - estimate_step_count
#                 elif true_step_count < estimate_step_count:
#                     true_positives += true_step_count
#                     false_positives += estimate_step_count - true_step_count
#             false_negative += np.sum(label[j:])

#         elif len(label) < len(step_count):
#             label = np.sort(label)
#             step_count = np.sort(step_count)
#             j = 0
#             for j in range(len(label)):
#                 true_step_count = label[j]
#                 estimate_step_count = step_count[j]
#                 if true_step_count == estimate_step_count:
#                     true_positives += true_step_count
#                 elif true_step_count > estimate_step_count:
#                     true_positives += estimate_step_count
#                     false_negative += true_step_count - estimate_step_count
#                 elif true_step_count < estimate_step_count:
#                     true_positives += true_step_count
#                     false_positives += estimate_step_count - true_step_count
#             false_positives += np.sum(step_count[j:])
#         print("Accuracy: ",true_positives/total_steps)
# print("Accuracy: ",true_positives/total_steps)
# print("false_positives: ", false_positives)
# print("true_positives: ", true_positives)
# print("false_negative: ", false_negative)
# print("true_negative: ", true_negative)
# print("total_steps: ", total_steps)