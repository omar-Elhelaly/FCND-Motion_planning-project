import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import a_star, heuristic, extract_polygons, collides, create_graph, polygon_for_landing #create_grid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local, local_to_global

#import sys
#import pkg_resources
#pkg_resources.require("networkx==2.1")
import networkx as nx
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue

class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if polygon_for_landing([self.local_position[0], self.local_position[1]]):
                if self.global_position[2] - self.global_home[2] < 3.1:
                    if self.local_position[2] < 3.01:
                        self.disarming_transition()
            else:
                if self.global_position[2] - self.global_home[2] < 0.1:
                    if abs(self.local_position[2]) < 0.01:
                        self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        print("this may take a few seconds ...")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        home_coord_data = np.genfromtxt('colliders.csv', delimiter=',', dtype='str', replace_space=',', max_rows=1)
        lon0 = float(home_coord_data[1].split()[1])
        lat0 = float(home_coord_data[0].split()[1])
        #print(lon0, lat0, type(lon0), type(lat0))
        # TODO: set home position to (lon0, lat0, 0)

        self.set_home_position(lon0, lat0, 0)

        # TODO: retrieve current global position

        start_global = self.global_position

        # TODO: convert to current local position using global_to_local()

        local_pos = global_to_local(start_global, self.global_home)
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        #grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        #print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        #grid_start = (-north_offset, -east_offset)
        # TODO: convert start position to current position rather than map center
        start = (int(local_pos[0]), int(local_pos[1]), int(local_pos[2]))

        # Set goal as some arbitrary position on the grid
        goal_global = (-122.399144, 37.793597, TARGET_ALTITUDE)
        # TODO: adapt to set goal as latitude / longitude position and convert
        goal = global_to_local(goal_global, self.global_home)

        polygons = extract_polygons(data, SAFETY_DISTANCE)

        print("sampling points ...")
        map_xmin = np.min(data[:, 0] - data[:, 3])
        map_xmax = np.max(data[:, 0] + data[:, 3])

        map_ymin = np.min(data[:, 1] - data[:, 4])
        map_ymax = np.max(data[:, 1] + data[:, 4])

        north_max = np.max(np.array([start[0], goal[0]]))
        north_min = np.min(np.array([start[0], goal[0]]))

        east_max = np.max(np.array([start[1], goal[1]]))
        east_min = np.min(np.array([start[1], goal[1]]))

        bigger_side = np.max(np.array([(north_max - north_min)/4, (east_max - east_min)/4]))

        n_max = north_max + bigger_side
        n_min = north_max - bigger_side

        e_max = east_max + bigger_side
        e_min = east_min - bigger_side

        scope_nmax = np.min(np.array([map_xmax, n_max]))
        scope_nmin = np.max(np.array([map_xmin, n_min]))

        scope_emax = np.min(np.array([map_ymax, e_max]))
        scope_emin = np.max(np.array([map_ymin, e_min]))

        zmin = np.int32(TARGET_ALTITUDE)
        zmax = np.int32(TARGET_ALTITUDE)

        num_samples = 20
        xvals = np.random.uniform(scope_nmin, scope_nmax, num_samples).astype(int)
        yvals = np.random.uniform(scope_emin, scope_emax, num_samples).astype(int)
        zvals = np.random.uniform(zmin, zmax, num_samples).astype(int)

        samples = list(zip(xvals, yvals, zvals))
        #print(samples)

        to_keep = []
        to_keep.append((np.int32(start[0]), np.int32(start[1]), np.int32(TARGET_ALTITUDE))) #np.int32(z)
        to_keep.append((np.int32(goal[0]), np.int32(goal[1]), np.int32(-goal[2])))
        for point in samples:
            if not collides(polygons, point):
                to_keep.append(point)
        print(to_keep)

        g = create_graph(to_keep, 4, polygons)
        print("Number of edges", len(g.edges))


        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        #print('Local Start and Goal: ', grid_start, grid_goal)
        #path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!

        a_start = (np.int32(start[0]), np.int32(start[1]), np.int32(TARGET_ALTITUDE))
        a_goal = (np.int32(goal[0]), np.int32(goal[1]), np.int32(-goal[2]))
        #print(type(a_start), type(a_goal))
        path, cost = a_star(g, heuristic, a_start, a_goal)
        print("Number of nodes in the Path:", len(path))
        print("Path Cost: ", cost)

        # Convert path to waypoints
        #waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        waypoints = [[int(p[0]), int(p[1]), int(p[2]), 0] for p in path]
        print(waypoints)
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        #self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        #self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
