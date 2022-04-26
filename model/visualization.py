import numpy as np
import open3d as o3d

def find_sidewalk(H_con_comps, z):
    
    # Find the sidewalk area.
    sidewalk_road, sidewalk_road_idx = np.array([], dtype=int), None
    lowest = 10
    for i, set in enumerate(H_con_comps):
        if len(set) > 2000:
            if lowest > np.mean(z[set]):
                lowest = np.mean(z[set])
                sidewalk_road = np.array(set, dtype=int)
                sidewalk_road_idx = i
    sidewalk_height_mean = lowest
    return sidewalk_road, sidewalk_road_idx, sidewalk_height_mean

def visualize_steps_and_stairs(stairs, step_heights, colors):
    red, green, blue, reds, greens, blues = colors

    # Print step heights.
    for stair in step_heights:
        print("Stair has {} steps.".format(len(stair)))
        for i, step_h in enumerate(stair):
            print("step {} height: {} cm".format(i+1, step_h))

    counter = 0
    for i in range(len(stairs)):
        if counter >= 10:
            counter = 0
        for j in range(len(stairs[i])):
            red[stairs[i][j]] = reds[counter]
            green[stairs[i][j]] = greens[counter]
            blue[stairs[i][j]] = blues[counter]
        counter += 1
    
    return red, green, blue

def visualize_all(V_con_comps, colors):
    red, green, blue, reds, greens, blues = colors
    counter = 0
    for i in range(len(V_con_comps)):
        temp = np.array(V_con_comps[i], dtype=int)
        if counter >= 10:
            counter = 0
        red[temp] = reds[counter]
        green[temp] = greens[counter]
        blue[temp] = blues[counter]
        counter += 1
    return red, green, blue

# def visualize(venue_pcd, V_con_comps, stairs, step_heights):
#     # Get colors of venue points
#     # red, green, blue = las.red[venue_idx], las.green[venue_idx], las.blue[venue_idx]
#     colors = np.array(venue_pcd.colors)
#     red, green, blue = colors[:,0], colors[:,1], colors[:,2]
#     c_max, c_min = 255**2, 255
#     c_max, c_min = 1, 0
#     reds = [c_max, c_min, c_min, c_min, c_max, c_max, c_max/2, c_max, c_max/2, c_max/4]
#     blues = [c_min, c_max, c_min, c_max, c_min, c_max, c_max/2, c_max/2, c_max, c_max/2]
#     greens = [c_min, c_max, c_max, c_min, c_max, c_min, c_max, c_max/2, c_max/2, c_max]

#     colors = (red, green, blue, reds, greens, blues)
#     if stairs is not None:
#         red, green, blue = visualize_steps_and_stairs(stairs, step_heights, colors)
#     else:
#         red, green, blue = visualize_all(V_con_comps, colors)

#     # red[sidewalk_road] = reds[counter]
#     # green[sidewalk_road] = greens[counter]
#     # blue[sidewalk_road] = blues[counter]

#     colors = np.dstack((red,green,blue))[0]/c_max # Stack the collors and devide by max value of green
#     venue_pcd.colors = o3d.utility.Vector3dVector(colors) # Give colors
#     o3d.visualization.draw_geometries([venue_pcd]) # Visualize
#     return venue_pcd

def visualize(pcd, V_con_comps, stairs, step_heights):
    # Get colors of venue points
    colors = np.array(pcd.colors)
    red, green, blue = colors[:,0], colors[:,1], colors[:,2]
    c_max, c_min = 1, 0
    reds = [c_max, c_min, c_min, c_min, c_max, c_max, c_max/2, c_max, c_max/2, c_max/4]
    blues = [c_min, c_max, c_min, c_max, c_min, c_max, c_max/2, c_max/2, c_max, c_max/2]
    greens = [c_min, c_max, c_max, c_min, c_max, c_min, c_max, c_max/2, c_max/2, c_max]

    colors = (red, green, blue, reds, greens, blues)
    if stairs is not None:
        red, green, blue = visualize_steps_and_stairs(stairs, step_heights, colors)

        lines = [[0, 1], [0,2], [0,3], [2,5], [5,3], [5,4], [2,7], [3,6], [6,4], [6,1], [1,7], [4,7]]
        geo_list = []
        bboxes = []
        for i in stairs:
            bbox_sub = []
            for step in i:
                points_2 = np.array(pcd.points)[np.array(step, dtype=int)]
                new_points = []
                for point in points_2:
                    random = np.random.randint(0,3,1)
                    random2 = np.random.randint(0,2,1)
                    if random2 == 0:
                        
                        new_point = point + np.array([[0.1,0,0],[0,0,0.1],[0,0.1,0]])[random]
                    else:
                        new_point = point +( -1 * np.array([[0.1,0,0],[0,0,0.1],[0,0.1,0]])[random])
                    new_points.append(new_point[0])

                new_points = np.array(new_points)
                points_2 = np.concatenate((points_2, new_points), axis=0)

                venue_pcd_2 = o3d.geometry.PointCloud()
                venue_pcd_2.points = o3d.utility.Vector3dVector(points_2)
                yes = o3d.geometry.OrientedBoundingBox()
                corner_box = yes.create_from_points(venue_pcd_2.points)
                bbox = corner_box.get_box_points()
                bbox_sub.append(np.asarray(bbox))

                # Use the same color for all lines
                colors = [[1, 0, 0] for _ in range(len(lines))]

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.open3d_pybind.utility.Vector3dVector(bbox)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                geo_list.append(line_set)
            bboxes.append(np.asarray(bbox_sub))
        
        colors = np.dstack((red,green,blue))[0]/c_max # Stack the collors and devide by max value of green
        pcd.colors = o3d.utility.Vector3dVector(colors) # Give colors
        geo_list.append(pcd)
        o3d.visualization.draw_geometries(geo_list) # Visualize
        return pcd, np.asarray(bboxes)
    else:
        red, green, blue = visualize_all(V_con_comps, colors)

    colors = np.dstack((red,green,blue))[0]/c_max # Stack the collors and devide by max value of green
    pcd.colors = o3d.utility.Vector3dVector(colors) # Give colors
    o3d.visualization.draw_geometries([pcd]) # Visualize
    return pcd
