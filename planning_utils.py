from enum import Enum
from queue import PriorityQueue
import numpy as np

import networkx as nx
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue

"""
def create_grid(data, drone_altitude, safety_distance):
    """ """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """ """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """ """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """ """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """ """
    Returns a list of valid actions given a grid and current node.
    """ """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions


def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost
"""

def extract_polygons(data, SAFETY_DISTANCE):
    print("Extracting polygons ...")
    polygons = []
    for i in range(data.shape[0]):
        x, y, alt, d_x, d_y, d_alt = data[i, :]

        #Extract the 4 corners of the obstacle
        point1 = (np.int32(x - d_x - SAFETY_DISTANCE), np.int32(y - d_y - SAFETY_DISTANCE))
        point2 = (np.int32(x + d_x + SAFETY_DISTANCE), np.int32(y - d_y - SAFETY_DISTANCE))
        point3 = (np.int32(x + d_x + SAFETY_DISTANCE), np.int32(y + d_y + SAFETY_DISTANCE))
        point4 = (np.int32(x - d_x - SAFETY_DISTANCE), np.int32(y + d_y + SAFETY_DISTANCE))

        corners = [point1, point2, point3, point4]

        #Compute the height of the polygon
        height = np.int32(alt + d_alt + SAFETY_DISTANCE)

        #Defining polygons
        p = Polygon(corners)
        polygons.append((p, height))

    #print(polygons[0][0])
    return polygons


def collides(polygons, point):
    #Determining if the point collides with any obstacles or not.

    p = Point(point[:2])
    for (poly, height) in polygons:
        if poly.contains(p) and height >= point[2]:
            break

    return poly.contains(p) and height >= point[2]


def can_connect (p1, p2, polygons):
    line = LineString([p1, p2])
    for polygon in polygons:
        if polygon[0].crosses(line) and polygon[1] >= min(p1[2], p2[2]):
            return False
    return True


def create_graph (nodes, k, polygons):
    print("Creating graph ...")
    from sklearn.neighbors import KDTree
    import numpy.linalg as LA
    import networkx as nx
    G = nx.Graph()
    tree = KDTree(nodes)
    for node in nodes:
        idxs = tree.query([node], k=k, return_distance=False)[0]

        for idx in idxs:
            node2 = nodes[idx]
            if node2 == node:
                continue

            if can_connect(node, node2, polygons):
                dist = LA.norm(np.array(node2) - np.array(node))
                G.add_edge(node, node2, weight=dist)
    return G


def a_star(graph, h, start, goal):

    """
    Modified A* to work with NetworkX graphs.
    """
    print("Finding best route ...")
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for node in graph[current_node]:
                next_node = node
                branch_cost = current_cost + graph[current_node][node]['weight']
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!, please try again')
        print('**********************')
    return path[::-1], path_cost


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def polygon_for_landing(local_position):
    corners = [(-37.4, -52), (-62, -28.5), (148, 181), (172, 157)]
    polygon = Polygon(corners)
    point = Point(local_position[:2])
    if polygon.contains(point):
        return True
    else:
        return False
