from path_extraction.prune import valid_neighbors, seed_neighbors, root_neighbors, is_tip
from path_extraction.path import Path
from path_extraction.vector_utils import Vector

from collections import namedtuple
from enum import Enum
import numpy as np
from queue import Queue
import math

import networkx as nx
import matplotlib.pyplot as plt

class PointType(Enum):
    TIP = 0,
    NODE = 1,
    SOURCE = 2


WalkedPath = namedtuple('WalkedPath', ['point_type', 'path'])

queue = Queue()
skel = None
G = nx.DiGraph()

"""
def __init__(skeleton, distances):
    self._skel = skeleton
    self._dist = skeleton * distances
    self._visited = set()
    self._WalkedPath = namedtuple('WalkedPath', ['point_type', 'path'])
"""

def walk_to_node(sender_p, start_p):
    # TODO: Cambiare il nome alla funzione
    """
    Returns a list of points (path) that are between the point `p` and the first available node or tip

    Paramteres:
        sender_p (tuple): The `father` of the starting point
        start_p (tuple): The starting point, which is the `son` (or neighbor) of `sender_p` 

    Returns:
        path: A list containing the points walked
    """

    global skel

    prev_point = sender_p
    cur_point = start_p
    path = []
    res = PointType.NODE

    while True:
        n = valid_neighbors(cur_point, root_neighbors, skel)
        n.remove(prev_point)

        path.append(cur_point)

        # The point is a node
        if len(n) > 1:
            break

        if is_tip(cur_point, skel):
            res = PointType.TIP
            break

        prev_point = cur_point
        cur_point = n[0]

    return WalkedPath(res, Path(path))

def draw_graph(G, node_attribute='pos', edge_attribute='weight', with_labels=True, node_size=100, rad=0.1):
    reverse_coords = lambda t: t[::-1]

    # Reverses the coordinates of the point (x, y) -> (y, x)
    pos = dict(map(lambda n: (n[0], reverse_coords(n[1])), nx.get_node_attributes(G, node_attribute).items()))

    plt.rc('font', size=8)          # controls default text sizes
    # Write text on top of the corresponding edge
    for edge in G.edges:
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]

        # Angle between the two nodes
        d = math.degrees(math.atan2(y2-y1, x2-x1))
        """
        print(d)
        tmp = (180+d)+180 if d < 0 else d
        sign = -1 if tmp > 90 and tmp < 270 else 1
        d = tmp + sign * 90
        """

        edge_val = G.edges[edge][edge_attribute]

        x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
        dx, dy = x2 - x1, y2 - y1

        f = rad

        # Corresponds to the center between edge 1 and edge 2
        t = 0.5

        # Control point of the bezier curve
        cx, cy = x12 + f * dy, y12 - f * dx

        x, y = quadratic_bezier((x1,y1), (cx, cy), (x2,y2), t)

        t = plt.text(x, y, edge_val, rotation=d, backgroundcolor='white')
        t.set_bbox(dict(facecolor='white', alpha=0.0))

    nx.draw(G, pos, with_labels=with_labels, connectionstyle=f'arc3, rad = {rad}', node_size=node_size)
    # TODO: fare lim automaticamente
    #plt.xlim([0, 500])
    #plt.ylim([0, 500])
    plt.gca().invert_yaxis()
    plt.show()

def quadratic_bezier(p0, p1, p2, t):
    """
    Computes a point on a quadratic Bezier curve.
    https://it.wikipedia.org/wiki/Curva_di_B%C3%A9zier

    Parameters:
    p0 (tuple): Starting point.
    p1 (tuple): Control point.
    p2 (tuple): End point.
    t (float): Parameter value in the range [0, 1].
    
    Returns:
    tuple: Coordinates of the point on the curve.
    """
    x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    return (x, y)
    
def add_nodes_and_edges(G, source_node, walked_path, max_res_err, min_points):
    if source_node not in G:
        G.add_node(source_node)
    
    prev_path_endpoint = source_node

    for path in walked_path.get_lstsq_paths(max_res_err, min_points):
        G.add_node(path.endpoint)
        G.add_edge(prev_path_endpoint, path.endpoint, weight=path.vector.angle)
        inv_vector = Vector.invert(path.vector)
        G.add_edge(path.endpoint, prev_path_endpoint, weight=inv_vector.angle)
        #H = nx.convert_node_labels_to_integers(G, label_attribute='pos')
        #draw_graph(H)
        prev_path_endpoint = path.endpoint

def create_graph(seeds, skeleton, distances, max_residual_err=1.0, min_points_lstsq=4):
    """
    Creates a networkx graph of the plant
    """

    global skel
    global queue
    global G

    skel = skeleton
    visited_nodes_neighbours = set()

    """
    G.add_node('Hamburg', pos=(300, 100), lol=True)
    G.add_node('Berlin', pos=(150, 200), lol=True)
    G.add_node('Stuttgart', pos=(0, 0), lol=True)
    G.add_node('Munich', pos=(500,500), lol=True)

    G.add_edge('Hamburg', 'Berlin', weight=2.0)
    G.add_edge('Berlin', 'Hamburg', weight=5.0)
    G.add_edge('Berlin', 'Munich', weight=5.0)
    G.add_edge('Munich', 'Berlin', weight=5.0)
    """


    for seed in seeds:
        # TODO: Riscriverla un po' meglio
        n = valid_neighbors(seed, root_neighbors, skel, return_idx=True)
        start_point = list(map(lambda n: n[1] if n[0] > 5 else None, n))[-1]

        _, path = walk_to_node(seed, start_point)

        cur_node = path.endpoint

        # TODO: spiegare meglio
        # If the node is already in the graph, it means that the previous seed
        # exploration already visited all the current plant
        if cur_node in G:
            G.nodes[cur_node]['ntype'] = PointType.SOURCE
            continue

        ntype = PointType.SOURCE
        G.add_node(cur_node, ntype=ntype)
        queue.put((ntype, cur_node))

        while not queue.empty():
            ntype, cur_node = queue.get()

            #G.add_node(cur_node, pos=cur_node_num, ntype=ntype)

            neighbours = valid_neighbors(cur_node, root_neighbors, skel)
            neighbours = [ n for n in neighbours if n not in visited_nodes_neighbours ]

            for n in neighbours:
                visited_nodes_neighbours.add(n)
                endpoint_type, walked_path = walk_to_node(cur_node, n)
                if walked_path.penultimate_point not in visited_nodes_neighbours:
                    add_nodes_and_edges(G, cur_node, walked_path, max_residual_err, min_points_lstsq)
                    if endpoint_type is not PointType.TIP:
                        queue.put((endpoint_type, walked_path.endpoint))


    H = nx.convert_node_labels_to_integers(G, label_attribute='pos')
    draw_graph(H, with_labels=False, node_size=20)
