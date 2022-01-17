import numpy as np

def gridify(x_borders, y_borders, step_size, x, y):
    """
    Function that creates a grid with cells defined by the cell borders in
    two given arrays, 'x_borders' and 'y_borders'. It then divides all points
    with coordinates 'x' and 'y' over this grid.

    Parameters
    ----------
    x_borders : array
        Array containing the border for the x-axis of the cells in the grid.
    y_borders : array
        Array containing the border for the y-axis of the cells in the grid.
    step_size : float
        Float defining the step size between cell borders.
    x : array
        Array containing x-coordinates of points.
    y : array
        Array containing y-coordinates of points.

    Output
    ------
    grid : list
        A two-dimensional grid with each cell containing the indices of points
        that fall within that cell.
    """

    # Create an empty grid with correct amount of cells.
    grid = [[[] for _ in range(len(y_borders))] for _ in range(len(x_borders))]
    
    # Place each point in its corresponding cell within the grid.
    number_of_pts = len(x)
    for i in range(number_of_pts):
        x_coor, y_coor = x[i], y[i]

        # Compute border of cell points belongs too.
        x_border = int((x_coor-(x_coor%step_size))/step_size)
        y_border = int((y_coor-(y_coor%step_size))/step_size)

        # Place point in cell within grid.
        grid[x_border][y_border].append(i)
    
    return grid

def find_potential_step_points(pts, poi_idx, params):
    """
    Function to filter out small areas from given set of connected components.

    Parameters
    ----------
    pts : array
        Array of points.
    poi_idx : array
        Array containing indices of the points of interest. All other points
        will not be considered within this function.
    params : tuple
    Tuple of parameters containing:
        step_sizes : list
            List of step_sizes where a step size is the height and width of 
            each cell within the grid. If more step sizes are given, we run the 
            function for each of them and return the concatenation of the points 
            found from each iteration.
        min_h_dif : float (default: 0.03)
            Minimum height difference within grid cell for points to be considered
            as potential step/stair points. Smaller value might be caused by noise,
            and we assume objects with a height of 0.03 meters is no obstacle for 
            people with reduced mobility.
        max_h_dif : float (default: 0.35)
            Maximum height difference within grid cell for points to be considered
            potential step/stair points. Height differences higher than this are 
            most likely walls, cars, etc.

    Output
    ------
    potential_step_points : array
        Indices of points that have the potential to be part of a step or stair.
    """
    step_sizes, min_h_dif, max_h_dif = params

    # If 'poi_idx' is not given, all points should be considered.
    if poi_idx is None: poi_idx = np.arange(0,len(pts),1)
    x, y, z = pts[poi_idx][:,0], pts[poi_idx][:,1], pts[poi_idx][:,2]
    x_min, y_min, x_max, y_max = 0, 0, np.max(x), np.max(y)

    idx_to_return = []
    for step_size in step_sizes:

        # Compute the x and y borders of each cell in grid.
        x_borders = np.arange(x_min, x_max+step_size, step_size)
        y_borders = np.arange(y_min, y_max+step_size, step_size)

        grid = gridify(x_borders, y_borders, step_size, x, y)

        for i in range(len(x_borders)):
            for j in range(len(y_borders)):
                pt_idx = grid[i][j]

                # Ignore if too few points within the cell.
                if len(pt_idx) > 15:

                    # Compute average minimum and maximum height of points within cell.          
                    z_values = np.sort(z[pt_idx])
                    z_min, z_max = np.mean(z_values[0:3]), np.mean(z_values[-3:])

                    # If the difference between maximum and minimum is between minimum
                    # and maximum height, we add the points within the cell to the 
                    # 'idx_to_return'.
                    z_differ = z_max - z_min
                    if z_differ >= min_h_dif and z_differ <= max_h_dif:
                        idx_to_return += list(pt_idx)
        
    potential_step_pts = poi_idx[idx_to_return]
    return np.unique(potential_step_pts)