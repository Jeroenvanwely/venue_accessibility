import numpy as np
import copy
import time
import open3d as o3d
from tqdm import tqdm

from sklearn.neighbors import KDTree
from numpy.linalg import svd

import las_utils
np.seterr(divide='ignore', invalid='ignore')

def connected_comps(data, distance_threshold, V_H):
    data = data[V_H]
    number_of_points = len(data)
    tree = KDTree(data)
    neighbour_groups = tree.query_radius(data, r=distance_threshold)


    unlabeled = np.arange(0, number_of_points, 1)
    labels = np.zeros(number_of_points)
    connected_components = []

    while len(unlabeled) > 0:

        point_index = unlabeled[0]
        neighbours_indices = list(neighbour_groups[point_index])
        prev_length = 0

        while prev_length != len(neighbours_indices):

            cur_length = copy.deepcopy(len(neighbours_indices))
            for i in range(prev_length-1, len(neighbours_indices)):
                group = neighbour_groups[neighbours_indices[i]].tolist()
                for j in group:
                    if labels[j] == 0:
                        neighbours_indices.append(j)
                        labels[j] = 1
            prev_length = copy.deepcopy(cur_length)

        connected_components.append(neighbours_indices)
        unlabeled = np.where(labels == 0)[0]

    label_groups = connected_components
    connected_components = []
    for group in label_groups:
        connected_components.append(np.unique(V_H[group]))

    return np.asarray(connected_components, dtype=object)

def filter_small_areas(connected_components, x, y, z, t=0.45, con_com_n_points_t = 450):
    connected_components_to_return = []
    for i in range(len(connected_components)):
        if len(connected_components[i]) < con_com_n_points_t: continue # If area contains only 50 points do not add
        else:
            x_coor, y_coor, z_coor = x[connected_components[i]], y[connected_components[i]], z[connected_components[i]]
            x_min, x_max = np.min(x_coor), np.max(x_coor)
            y_min, y_max = np.min(y_coor), np.max(y_coor)
            z_min, z_max = np.min(z_coor), np.max(z_coor)
            if (x_max-x_min > t or y_max-y_min > t) and z_max-z_min<=0.5:
                connected_components_to_return.append(connected_components[i])
    return np.asarray(connected_components_to_return, dtype=object)

def gridify(points, V, step_size, min_h_dif=0.03, max_h_dif=0.35):
    x, y, z = points[V][:,0], points[V][:,1], points[V][:,2] # Split the coordinates for all the points
    x_min, y_min = 0, 0
    x_max, y_max = np.max(x), np.max(y)
    number_of_points = len(x)
    x_borders = np.arange(x_min, x_max+step_size, step_size)
    y_borders = np.arange(y_min, y_max+step_size, step_size)

    grid = []
    for i in range(len(x_borders)):
        row = []
        for j in range(len(y_borders)):
            row.append([])
        grid.append(row)

    for i in range(number_of_points):
        x_coor, y_coor = x[i], y[i]
        x_border = int((x_coor-(x_coor%step_size))/step_size)
        y_border = int((y_coor-(y_coor%step_size))/step_size)
        grid[x_border][y_border].append(i)

    indices_to_return = []
    for i in range(len(x_borders)):
        for j in range(len(y_borders)):
            point_indices = grid[i][j]
            if len(point_indices) > 15:
                z_values = z[point_indices]
                z_min, z_max = np.min(z_values), np.max(z_values)
                # print(z_max)
                # print(z_min)
                
                z_values = np.sort(z[point_indices])
                z_min, z_max = np.mean(z_values[0:3]), np.mean(z_values[-3:])
                # print(z_max)
                # print(z_min)
                # print("======")
                z_differ = z_max - z_min
                if z_differ >= min_h_dif and z_differ <= max_h_dif:
                    indices_to_return += list(point_indices)
    
    return V[indices_to_return]

def find_steps(points, vertical_components, step_min_range, step_max_range):
    max_means, min_means, x_mean, y_mean = [], [], [], []
    for v_comp in vertical_components:
        sorted_z = sorted(points[np.array(v_comp, dtype=int)][:,2])
        max_means.append(np.mean(sorted_z[-25:]))
        min_means.append(np.mean(sorted_z[:25]))
        x_mean.append(np.mean(points[np.array(v_comp, dtype=int)][:,0]))
        y_mean.append(np.mean(points[np.array(v_comp, dtype=int)][:,1]))

    steps = []
    placed2 = np.zeros(len(max_means), dtype=int)
    # print(max_means)
    # print(min_means)
    for i in range(len(max_means)):
        # print("We will start checking step {}".format(i))
        # print("min mean must be smaller then {}".format(max_means[i]+step_min_range))
        # print("min mean must be larger then {}".format(max_means[i]-step_min_range))
        near_step = np.where( (min_means < max_means[i]+step_min_range) & (min_means > max_means[i]-step_min_range) & ( np.abs(x_mean[i] - x_mean) < step_max_range) & ( np.abs(y_mean[i] - y_mean) < step_max_range))[0]
        near_step = np.delete(near_step, np.where((near_step == i))[0])
        n_near_steps = len(near_step)
        if  n_near_steps > 0 and near_step[0] != i:
            # print("Step(s) found is step {}".format(near_step))
            placed = False
            for j in range(len(steps)):
                if i in steps[j]:
                    placed = True
                    placed2[near_step[0]] = 1
                    steps[j].append(near_step[0])
                elif near_step[0] in steps[j]:
                    placed = True
                    placed2[i] = 1
                    steps[j].append(i)
            if placed == False:
                # print("Both steps where not placed yet. Will place them together now")
                steps.append([i, near_step[0]])
                placed2[i] = 1
                placed2[near_step[0]] = 1
        elif placed2[i] == 0:
            # print("no close steps found.")
            steps.append([i])
            placed2[i] = 1
    return steps, np.array(min_means), np.array(max_means)


def find_plane(points):
    ctr = points.mean(axis=1)
    x__ = points - ctr[:,np.newaxis]
    M = np.dot(x__, x__.T)
    
    plane = svd(M)[0][:,-1]
    plane_length = np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)

    return plane, plane_length


def plane_search(points, connected_components_V, V5):
    
    planes = []
    plane_lengths = []
    norm_planes = []
    for group in connected_components_V:
        sub_points = points[group].T
        plane, plane_length = find_plane(sub_points)
        planes.append(plane)
        plane_lengths.append(plane_length)
        norm_planes.append(plane / plane_length)
    
    tree = KDTree(points)
    new_connected_components_V = []
    center_points = []
    for i, group in enumerate(connected_components_V):
        sub_points = points[group].T
        plane, plane_length, norm_plane = planes[i], plane_lengths[i], norm_planes[i]

        if norm_plane[2] < 0.5:
        
            plane_similarity = np.sum(norm_planes - norm_plane, axis=0)
            paralel_planes = len(np.where( (plane_similarity < 0.05) )[0])

            maxima = sub_points.max(axis=1)
            minima = sub_points.min(axis=1)
            center_point = [minima[0]+(maxima[0]-minima[0])/2, 
                            minima[1]+(maxima[1]-minima[1])/2, 
                            minima[2]+(maxima[2]-minima[2])/2]

            neighbours = np.intersect1d(tree.query_radius([center_point], r=0.8)[0], V5)
            centered_points = points[neighbours] - center_point
            center_points.append(center_point)
            closeness_to_plane = np.abs(np.dot(centered_points, plane)) / plane_length
            indices_points_in_plane = np.where((closeness_to_plane < 0.06))[0]

            indices1 = neighbours[indices_points_in_plane]
            z_indices_1 = z[indices1]
            filtered_points_in_plane = np.where((z_indices_1 <= maxima[2]+0.1) & (z_indices_1 >= minima[2]-0.1))[0]
            indices2 = indices1[filtered_points_in_plane]

            points_in_plane = points[indices2]
            if len(points_in_plane) == 0:
                continue
            subtree = KDTree(points_in_plane)
            sub_neighboors = subtree.query_radius(points_in_plane, r=.05, count_only=True)
            to_keep = np.where((sub_neighboors >= 10))[0]
            indices3 = indices2[to_keep]

            plane_indices = np.unique(np.concatenate((group, indices3), axis=0))

            max_ratio = 0
            for j in range(len(new_connected_components_V)):
                equality_ratio = len(set(plane_indices) & set(new_connected_components_V[j])) / len(plane_indices)
                if equality_ratio > max_ratio:
                    max_ratio = equality_ratio
                    max_con_com_group = j

                    # center_point_distance = np.sum((np.array(center_points[max_con_com_group][0:2]) - np.array(center_points[-1][0:2]))**2)**0.5

            # if max_ratio > 0.2 and center_point_distance < 0.05:
            if max_ratio > 0.4:
                # print(center_point_distance)
                new_group = np.unique(np.concatenate((new_connected_components_V[max_con_com_group], plane_indices), axis=0))
                new_connected_components_V[max_con_com_group] = new_group
            else:
                new_connected_components_V.append(plane_indices)

    # tree = KDTree(points)
    # new_connected_components_V = []
    # for group in connected_components_V:
    #     sub_points = points[group].T
    #     plane, plane_length = find_plane(sub_points)

    #     maxima = sub_points.max(axis=1)
    #     minima = sub_points.min(axis=1)
    #     center_point = [minima[0]+(maxima[0]-minima[0])/2, 
    #                     minima[1]+(maxima[1]-minima[1])/2, 
    #                     minima[2]+(maxima[2]-minima[2])/2]

    #     neighbours = np.intersect1d(tree.query_radius([center_point], r=2.5)[0], V5)
    #     centered_points = points[neighbours] - center_point
    #     closeness_to_plane = np.abs(np.dot(centered_points, plane)) / plane_length
    #     indices_points_in_plane = np.where((closeness_to_plane < 0.05))[0]

    #     indices1 = neighbours[indices_points_in_plane]
    #     z_indices_1 = z[indices1]
    #     filtered_points_in_plane = np.where((z_indices_1 <= maxima[2]+0.1) & (z_indices_1 >= minima[2]-0.1))[0]
    #     indices2 = indices1[filtered_points_in_plane]

    #     points_in_plane = points[indices2]
    #     if len(points_in_plane) == 0:
    #         continue
    #     subtree = KDTree(points_in_plane)
    #     sub_neighboors = subtree.query_radius(points_in_plane, r=.25, count_only=True)
    #     to_keep = np.where((sub_neighboors >= 25))[0]
    #     indices3 = indices2[to_keep]

    #     new_connected_components_V.append(np.unique(np.concatenate((group, indices3), axis=0)))
    
    return new_connected_components_V



start = time.time()

# Obtained from venues csv file
x_min, y_max, x_max, y_min = 119336.1808743,485102.451,119341.33392,485096.1142418 # Niets
# x_min, y_max, x_max, y_min = 119295.806,485150.368,119306.363,485137.549 # Niets
# x_min, y_max, x_max, y_min = 119304.678,485139.85,119316.8687185,485126.962 # Niet erg duidelijk
# x_min, y_max, x_max, y_min = 119342.628,485155.657,119355.247,485141.759 # Deur voor helft zichtbaar
x_min, y_max, x_max, y_min = 119302.238,485150.897,119315.189,485136.658 # Winkel

x_min, y_max, x_max, y_min = 119293.44, 485131.159, 119316.461, 485108.919 # Groot
# tilecode = '2386_9702'

x_min, y_max, x_max, y_min = 119877.496,485266.533,119890.32,485254.402 # Winkel twee treden ver uit elkaar 1
x_min, y_max, x_max, y_min = 119871.711,485269.732,119883.878,485257.306 # Winkel één tree 2
x_min, y_max, x_max, y_min = 119868.123,485272.814,119878.091,485258.847 # Winkel hoge deurpost trede 3
x_min, y_max, x_max, y_min = 119859.817,485273.89,119872.515,485259.386 # Twee deuren, eentje lastig te pakken 4
x_min, y_max, x_max, y_min = 119878.251,485259.371,119892.559,485248.155 # Huis twee treedes 5
x_min, y_max, x_max, y_min = 119878.761,485254.049,119894.988,485242.584 # Orgineel twee tredes en deurpost trede 6
# x_min, y_max, x_max, y_min = 119854.431,485269.014,119864.82,485253.085 # Moeite waard om te checken, weinig datapunten
# x_min, y_max, x_max, y_min = 119853.955,485305.265,119866.608,485291.101 # Geen trap
# x_min, y_max, x_max, y_min = 119849.126,485266.596,119860.267,485252.242 # Te veel noise
# x_min, y_max, x_max, y_min = 119849.382,485301.656,119860.355,485288.353 # Ccclusion
# x_min, y_max, x_max, y_min = 119843.099,485302.261,119854.204,485285.986 # Geen deur
# x_min, y_max, x_max, y_min = 119891.0358053,485282.773,119892.5767026,485280.6975532 # Niets
tilecode = '2397_9705'

# 1,2,3,6
max_nn = 25
radius = 0.03
V_normal_threshold = 0.3
H_normal_threshold = 0.9
con_com_dis_thresh_H = 0.08
con_com_dis_thresh_V = 0.1
gridify_block_size_1 = 0.1
gridify_block_size_2 = 0.075
step_min_range = 0.3
step_max_range = 0.8
small_area_t = 0.45
con_com_n_points_t = 300
min_h_dif=0.05
max_h_dif=0.35

in_file = 'Data/Pointcloud/filtered_' + tilecode + '.laz'
pointcloud = las_utils.read_las(in_file)
points = np.vstack((pointcloud.x, pointcloud.y, pointcloud.z))
x, y, z = points[0], points[1], points[2] # Isolate x, y and z coordinates
venue_indices = np.where((x>x_min) & (x<x_max) & (y>y_min) & (y<y_max) & (z<3.0) )[0] # Remove all points

# Center points such that minimum value for each axis is zero
x = x[venue_indices] - np.min(x[venue_indices])
y = y[venue_indices] - np.min(y[venue_indices])
z = z[venue_indices] - np.min(z[venue_indices])

points = np.dstack((x, y, z))[0] # Stack the coordinates
venue_pcd = o3d.geometry.PointCloud() # Initialize point cloud
venue_pcd.points = o3d.utility.Vector3dVector(points) # Give points

venue_pcd2 = o3d.geometry.PointCloud() # Initialize point cloud
venue_pcd2.points = o3d.utility.Vector3dVector(points) # Give points

# Compute normal vectors of every point in cloud based on 25 neurest neighours found in radius of 1000
venue_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

venue_pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=max_nn))
# normals_x = np.asarray(venue_pcd.normals)[:,0] # Isolate Z normals
# normals_y = np.asarray(venue_pcd.normals)[:,1] # Isolate Z normals
normals_z = np.abs(np.asarray(venue_pcd.normals)[:,2]) # Isolate Z normals
normals_z2 = np.abs(np.asarray(venue_pcd2.normals)[:,2]) # Isolate Z normals

# Nx = np.degrees(np.abs(np.arctan(normals_x/normals_z)))
# Ny = np.degrees(np.abs(np.arctan(normals_y/normals_z)))
# V = np.where((Nx>20) & (Nx<=90) & (Ny>20) & (Ny<=90))[0]
# H = np.where((Nx>0) & (Nx<=5) & (Ny>0) & (Ny<=5))[0]

V = np.where((normals_z <= V_normal_threshold))[0] # Indices of vertical normals
I = np.where((normals_z > V_normal_threshold) & (normals_z < H_normal_threshold ))[0] # Indices of tilted normals
H = np.abs(np.where((normals_z2 >= H_normal_threshold ))[0]) # Indices of horizontal normals

V1 = gridify(points, V, step_size=gridify_block_size_1, min_h_dif=min_h_dif, max_h_dif=max_h_dif) # Only vertical points near height differences are important
V2 = gridify(points, V, step_size=gridify_block_size_2, min_h_dif=min_h_dif, max_h_dif=max_h_dif)
V = np.unique(np.concatenate((V1,V2), axis=0))

V_ = np.arange(0, len(points), 1)
V3 = gridify(points, V_, step_size=gridify_block_size_1, min_h_dif=min_h_dif, max_h_dif=max_h_dif) # Only vertical points near height differences are important
V4 = gridify(points, V_, step_size=gridify_block_size_2, min_h_dif=min_h_dif, max_h_dif=max_h_dif)
V5 = np.unique(np.concatenate((V3,V4), axis=0))

connected_components_H = connected_comps(points, distance_threshold=con_com_dis_thresh_H, V_H=H) # Get connected components horizontal
connected_components_V = connected_comps(points, distance_threshold=con_com_dis_thresh_V, V_H=V) # Get connected components vertical

connected_components_V = plane_search(points, connected_components_V, V5)

# connected_components_H = filter_small_areas(connected_components_H, x, y, z, t=small_area_t, con_com_n_points_t=con_com_n_points_t) # Filter small areas
connected_components_V = filter_small_areas(connected_components_V, x, y, z, t=small_area_t, con_com_n_points_t=con_com_n_points_t) # Filter small areas

# Find the sidewalk area
sidewalk_road, sidewalk_road_indices = np.array([], dtype=int), []
lowest = 10
for i, set in enumerate(connected_components_H):
    if len(set) > 2000:
        if lowest > np.mean(z[set]):
            lowest = np.mean(z[set])
            sidewalk_road = np.array(set, dtype=int)
        # if len(sidewalk_road) == 0:
        #     sidewalk_road = np.array(set,dtype=int)
        # else:
        #     sidewalk_road = np.concatenate((sidewalk_road, np.array(set,dtype=int)), axis=0)
            sidewalk_road_indices = i
sidewalk_height_mean = lowest


# Remove sidewalk area from connected components and combine rest
# of horizontal connected components with vertical connected components
connected_components_H = np.delete(connected_components_H, sidewalk_road_indices)

# Get colors of venue points
red, green, blue = pointcloud.red[venue_indices], pointcloud.green[venue_indices], pointcloud.blue[venue_indices]
c_max, c_min = 255**2, 255
reds = [c_max, c_min, c_min, c_min, c_max, c_max, c_max/2, c_max, c_max/2, c_max/4]
blues = [c_min, c_max, c_min, c_max, c_min, c_max, c_max/2, c_max/2, c_max, c_max/2]
greens = [c_min, c_max, c_max, c_min, c_max, c_min, c_max, c_max/2, c_max/2, c_max]

connected, min_means, max_means = find_steps(points, connected_components_V, step_min_range, step_max_range)
stairs = []
print(connected)
for steps in connected:
    stair = []
    min_sorted = np.sort(min_means[steps])
    if min_sorted[0] < sidewalk_height_mean+0.15:
        max_sorted = np.sort(max_means[steps])
        print("Stair has {} steps.".format(len(min_sorted)))
        for j in range(len(steps)):
            stair += list(connected_components_V[np.array(steps[j])])
            print("step {} height: {} cm".format(j+1, np.round((max_sorted[j]-min_sorted[j])*100, decimals=2)))
        stairs.append(stair)

counter = 0
for i in range(len(stairs)):
    if counter >= 10:
        counter = 0
    red[stairs[i]] = reds[counter]
    green[stairs[i]] = greens[counter]
    blue[stairs[i]] = blues[counter]
    counter += 1

# counter = 0
# for i in range(len(connected_components_V)):
#     temp = np.array(connected_components_V[i], dtype=int)
#     # if np.min(z[connected_components_V[i]]) < sidewalk_height_mean+0.4:
#     if counter >= 10:
#         counter = 0
#     red[temp] = reds[counter]
#     green[temp] = greens[counter]
#     blue[temp] = blues[counter]
#     counter += 1

red[sidewalk_road] = reds[counter]
green[sidewalk_road] = greens[counter]
blue[sidewalk_road] = blues[counter]

print("Took {} seconds.".format(time.time()-start))

colors = np.dstack((red,green,blue))[0]/c_max # Stack the collors and devide by max value of green
venue_pcd.colors = o3d.utility.Vector3dVector(colors) # Give colors
o3d.visualization.draw_geometries([venue_pcd]) # Visualize
