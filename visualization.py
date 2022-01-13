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
        red[stairs[i]] = reds[counter]
        green[stairs[i]] = greens[counter]
        blue[stairs[i]] = blues[counter]
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

def visualize(las, venue_idx, V_con_comps, venue_pcd, stairs, step_heights):
    # Get colors of venue points
    red, green, blue = las.red[venue_idx], las.green[venue_idx], las.blue[venue_idx]
    c_max, c_min = 255**2, 255
    reds = [c_max, c_min, c_min, c_min, c_max, c_max, c_max/2, c_max, c_max/2, c_max/4]
    blues = [c_min, c_max, c_min, c_max, c_min, c_max, c_max/2, c_max/2, c_max, c_max/2]
    greens = [c_min, c_max, c_max, c_min, c_max, c_min, c_max, c_max/2, c_max/2, c_max]

    colors = (red, green, blue, reds, greens, blues)
    if stairs is not None:
        red, green, blue = visualize_steps_and_stairs(stairs, step_heights, colors)
    else:
        red, green, blue = visualize_all(V_con_comps, colors)

    # red[sidewalk_road] = reds[counter]
    # green[sidewalk_road] = greens[counter]
    # blue[sidewalk_road] = blues[counter]

    colors = np.dstack((red,green,blue))[0]/c_max # Stack the collors and devide by max value of green
    venue_pcd.colors = o3d.utility.Vector3dVector(colors) # Give colors
    o3d.visualization.draw_geometries([venue_pcd]) # Visualize