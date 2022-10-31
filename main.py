import time

import pandas as pd
import plotly.express as px
import numpy as np
from osgeo import gdal
import plotly.graph_objects as go
import heapq
from geopy import distance
from typing import Dict, List, Iterator, Tuple, TypeVar, Optional
from typing_extensions import Protocol

USE_PATH_FOR_GDAL_PYTHON = 'YES'

# TODO : modify here
shipSpeed = 14  # 10kn = 18.52km/h
fuelConsum = (shipSpeed / 14 * 98.1) / 24  # 14kn in 98.10t/day unit:h
DraftDesign = 5


np.random.seed(0)
T = TypeVar('T')
global global_water_depth, waterDepth
global_water_depth = gdal.Open("data/ETOPO1_Bed_c_geotiff.tif")
band1 = global_water_depth.GetRasterBand(1)
waterDepth = np.array(pd.DataFrame(band1.ReadAsArray()))[::10, ::10]
all_nodes = band1.ReadAsArray()
all_nodes_pd = pd.DataFrame(all_nodes)
nodes_pd_index = all_nodes_pd.index
nodes_pd_columns = all_nodes_pd.columns
all_node_np = np.array(all_nodes)
GridLocation = Tuple[int, int]
Location = TypeVar('Location')
MAX_COST = 100000
land_np = np.array(np.where(all_node_np >= DraftDesign, MAX_COST, all_node_np))
all_node_np = np.array(np.where(all_node_np < DraftDesign, 0.23156, land_np))


def cal_time(p1, p2):
    ship_speed = shipSpeed
    return distance.distance(i2dm(p1[0], p1[1]), i2dm(p2[0], p2[1])).nm / ship_speed  # unit: h


def cal_time_dms(p1, p2):
    ship_speed = shipSpeed
    return distance.distance((p1[0], p1[1]), (p2[0], p2[1])).nm / ship_speed  # unit: h


def point_validity(lat, lon):
    lon_ = int((lon + 180) * 6)
    lat_ = int((90 - lat) * 6)
    dep = waterDepth[lat_, lon_]
    if dep > -DraftDesign:
        return False
    else:
        return True


def point_validity_d(lat, lon):
    lon_ = lon
    lat_ = lat
    dep = waterDepth[lat_, lon_]
    if dep > -DraftDesign:
        return False
    else:
        return True


def dms2d(degree=0, minute=0, second=0):
    result = 0.
    if degree:
        result = degree
    if minute:
        result += minute / 60
    if second:
        result += second / 3600
    return round(result, 5)


def i2dm(i=0, c=0):
    if i >= 0:
        lat = 90 - i / 6
    if c >= 0:
        lon = c / 6 - 180
    return lat, lon


def neighbors_dm(node):
    """
        node example : [dm2d(12,0),dm2d(32,0)]
    """
    dirs = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    # dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    node = [int((90 - node[0]) * 60), int((node[1] + 180) * 60)]
    result = []
    for dir in dirs:
        # 处理跨过180度经线的情况
        nei_lat = node[0] + dir[0]
        nei_lon = node[1] + dir[1]
        if nei_lon >= 2160:
            nei_lon = nei_lon - 2160
        if nei_lon <= -1:
            nei_lon = nei_lon + 2160
        else:
            nei_lon = nei_lon
        neighbors = [nei_lat, nei_lon]
        if neighbors[0] in nodes_pd_index and neighbors[1] in nodes_pd_columns:
            result.append(neighbors)
    return result


def neighbors_d1(node):
    """
        node example : [468, 1272]
    """
    dirs = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1], [-1, 2], [-2, 1], [-2, -1], [-1, -2],
            [1, -2], [2, -1], [2, 1], [1, 2]]
    node = [node[0], node[1]]
    result = []
    for dir in dirs:
        # 处理跨过180度经线的情况
        nei_lat = node[0] + dir[0]
        nei_lon = node[1] + dir[1]
        if nei_lon >= 2160:
            nei_lon = nei_lon - 2160
        elif nei_lon <= -1:
            nei_lon = nei_lon + 2160
        else:
            nei_lon = nei_lon
        neighbors = [nei_lat, nei_lon]
        if neighbors[0] in nodes_pd_index and neighbors[1] in nodes_pd_columns:
            result.append(neighbors)
    return result


def neighbors_d10(node):
    """
        node example : [468, 1272]
    """
    dirs = [[10, 0], [0, 10], [-10, 0], [0, -10], [10, 10], [10, -10], [-10, 10], [-10, -10], [-10, 20], [-20, 10],
            [-20, -10], [-10, -20], [10, -20], [20, -10], [20, 10], [10, 20]]
    node = [node[0], node[1]]
    result = []
    for dir in dirs:
        # 处理跨过180度经线的情况
        nei_lat = node[0] + dir[0]
        nei_lon = node[1] + dir[1]
        if nei_lon >= 2160:
            nei_lon = nei_lon - 2160
        elif nei_lon <= -1:
            nei_lon = nei_lon + 2160
        else:
            nei_lon = nei_lon
        neighbors = [nei_lat, nei_lon]
        if neighbors[0] in nodes_pd_index and neighbors[1] in nodes_pd_columns:
            result.append(neighbors)
    return result


class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> T:
        return heapq.heappop(self.elements)[1]


class Graph(Protocol):
    def neighbors(self, id: Location) -> List[Location]: pass


class SquareGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.walls: List[GridLocation] = []

    def in_bounds(self, id: GridLocation) -> bool:
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id: GridLocation) -> bool:
        return id not in self.walls

    def neighbors(self, id: GridLocation) -> Iterator[GridLocation]:
        (x, y) = id
        neighbors = [(x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1)]  # E W N S
        # see "Ugly paths" section for an explanation:
        if (x + y) % 2 == 0: neighbors.reverse()  # S N W E
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        return results


class GridWithWeights(SquareGrid):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self.weights: Dict[GridLocation, float] = {}

    def cost(self, from_node: GridLocation, to_node: GridLocation) -> float:
        # return self.weights.get(to_node, 1)
        return cal_time(from_node, to_node)


def reconstruct_path_(came_from: Dict[Location, Location],
                      start: Location, goal: Location) -> List:
    current: Location = goal
    path: List[Location] = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)  # optional
    path.reverse()  # optional
    return path


def heuristic_(a: GridLocation, b: GridLocation) -> float:
    c = cal_time(a, b)
    return c


def astar_search(graph: GridWithWeights, start: Location, goal: Location):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from: Dict[Location, Optional[Location]] = {}
    cost_so_far: Dict[Location, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0
    times = 0
    print("graph.width * graph.height =", graph.width * graph.height)
    while not frontier.empty():
        times += 1
        current: Location = frontier.get()
        if current == goal:
            break
        # for next in neighbors_d10(current):
        #     if point_validity_d(next[0],next[1]):
        #         next = tuple(next)
        #         new_cost = cost_so_far[current] + graph.cost(current, next)
        #         if (next not in cost_so_far or new_cost < cost_so_far[next]) :
        #             cost_so_far[next] = new_cost
        #             priority = new_cost + heuristic_(next, goal)
        #             frontier.put(next, priority)
        #             came_from[next] = current
        #             print("heuristic_(next, goal) = ", heuristic_(next, goal))
        #             if heuristic_(next, goal) < 10:
        #                 break
        #             continue
        for next in neighbors_d1(current):
            if point_validity_d(next[0], next[1]):
                next = tuple(next)
                new_cost = cost_so_far[current] + graph.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic_(next, goal)
                    frontier.put(next, priority)
                    came_from[next] = current
                    print("heuristic_(next, goal) = ", heuristic_(next, goal))
                    continue
    return came_from, cost_so_far


def make_start2goal(lat1, lon1, lat2, lon2):
    lat_start = lat1
    lon_start = lon1
    lat_goal = lat2
    lon_goal = lon2
    start = (int((90 - dms2d(lat_start)) * 6), int((dms2d(lon_start) + 180) * 6))
    goal = (int((90 - dms2d(lat_goal)) * 6), int((dms2d(lon_goal) + 180) * 6))
    return start, goal



def make_path_pd(path):
    lat = []
    lon = []
    color = []
    fuel = []
    speed = []
    time = []
    for i in range(len(path) - 1):
        ship_speed = np.random.randint(10, 14)
        lat_1 = 90 - path[i][0] / 6
        lat_2 = 90 - path[i + 1][0] / 6
        lon_1 = path[i][1] / 6 - 180
        lon_2 = path[i + 1][1] / 6 - 180
        lat.append(lat_1)
        lon.append(lon_1)
        color.append("pa")
        time.append(cal_time_dms([lat_1, lon_1], [lat_2, lon_2]))  # unit : h
        fuel.append(fuelConsum * cal_time_dms([lat_1, lon_1], [lat_2, lon_2]))  # unit : h
        speed.append(ship_speed)
    data = pd.DataFrame({'lat': lat, 'lon': lon, 'fuel': fuel, 'time': time, 'speed': speed, 'color': color},
                        index=range(len(path) - 1))
    data.to_csv("data/path.csv")
    return data


def make_came_pd(came_from):
    lat = []
    lon = []
    color = []
    for i in came_from:
        lat.append(90 - i[0] / 6)
        lon.append(i[1] / 6 - 180)
        color.append("ca")
    data = pd.DataFrame({'lat': lat, 'lon': lon, "color": color}, index=range(len(came_from)))
    data.to_csv("data/path_came.csv")
    return data


def make_cost_pd(cost_so_far):
    lat_list = list(cost_so_far.keys())
    lon_list = list(cost_so_far.keys())
    lat = []
    lon = []
    color = []
    for i in range(len(lat_list)):
        lat.append(90 - lat_list[i][0] / 6)
        lon.append(lon_list[i][1] / 6 - 180)
        color.append("cs")
    data = pd.DataFrame({'lat': lat, 'lon': lon, "color": color}, index=range(len(lat_list)))
    data.to_csv("data/path_cost.csv")
    return data


def weatherRouting(lat1, lon1, lat2, lon2):
    if point_validity(lat1, lon1) and point_validity(lat2, lon2):
        print("start and goal valid!")
        start, goal = make_start2goal(lat1, lon1, lat2, lon2)
        leftupy = min(start[0], goal[0])
        leftupx = min(start[1], goal[1])
        rightdowny = max(start[0], goal[0])
        rightdownx = max(start[1], goal[1])
        node_need_np = all_node_np[leftupy:rightdowny, leftupx:rightdownx]
        node_need_pd = pd.DataFrame(node_need_np, index=range(leftupy, rightdowny), columns=range(leftupx, rightdownx))
        DiagramTest = GridWithWeights(node_need_np.shape[1], node_need_np.shape[0])
        WALL_ARRAY2D = np.array(np.where(node_need_np == MAX_COST))
        WALLS = []
        for i in range(len(WALL_ARRAY2D[0])):
            WALLS.append(tuple([WALL_ARRAY2D[0][i], WALL_ARRAY2D[1][i]]))
        DiagramTest.walls = WALLS
        weights_need = {}
        for lat_index in node_need_pd.index:
            for lon_index in node_need_pd.columns:
                weights_need[(lat_index, lon_index)] = all_node_np[lat_index, lon_index]
        DiagramTest.weights = weights_need
        came_from, cost_so_far = astar_search(DiagramTest, start, goal)
        path = reconstruct_path_(came_from, start=start, goal=goal)
        return path, came_from, cost_so_far
    else:
        raise Exception("start or goal point not valid! check the point lat and lon")

def main():
    t_start = time.time()
    # TODO : modify here
    # 上海港 CNSHA 至阿拉伯 达曼港 SADMA
    # path, came_from, cost_so_far = weatherRouting(dms2d(31, 2.799), dms2d(122, 7.588), dms2d(26, 51.797), dms2d(50, 40.137))
    # 渤海 ： 天津至青岛
    path, came_from, cost_so_far = weatherRouting(dms2d(38, 47), dms2d(118, 17), dms2d(35, 58), dms2d(120, 43))
    make_path_pd(path)
    make_came_pd(came_from)
    make_cost_pd(cost_so_far)
    t_end = time.time()
    print("[route find time cost] : ", t_end - t_start, ' s')
    return None


if __name__ == '__main__':
    main()
