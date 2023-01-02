from path_extraction.prune import valid_neighbors, seed_neighbors, root_neighbors, is_tip
from path_extraction.path import Path
from path_extraction.vector_utils import Vector

from collections import namedtuple
from enum import Enum
import numpy as np
import queue
import math

import networkx as nx
import matplotlib.pyplot as plt

class PointType(Enum):
    TIP = 0,
    NODE = 1,
    INTERNAL = 2


WalkedPath = namedtuple('WalkedPath', ['point_type', 'path'])

visited = set()
skel = None

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

    # Write text on top of the corresponding edge
    for edge in G.edges:
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]

        # Angle between the two nodes
        d = math.degrees(math.atan2(y2-y1, x2-x1)) - 90.0

        edge_val = G.edges[edge][edge_attribute]

        x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
        dx, dy = x2 - x1, y2 - y1

        f = rad

        # Corresponds to the center between edge 1 and edge 2
        t = 0.5

        # Control point of the bezier curve
        cx, cy = x12 + f * dy, y12 - f * dx

        x, y = quadratic_bezier((x1,y1), (cx, cy), (x2,y2), t)

        plt.text(x, y, edge_val, rotation=d, backgroundcolor='white')

    nx.draw(G, pos, with_labels=with_labels, connectionstyle=f'arc3, rad = {rad}')
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
    
def create_graph(seeds, skeleton, distances, max_residual_err=1.0, min_points_lstsq=4):
    """
    Creates a networkx graph of the plant
    """
    
    global visited
    global skel

    G = nx.DiGraph()

    G.add_node('Hamburg', pos=(300, 100), lol=True)
    G.add_node('Berlin', pos=(150, 200), lol=True)
    G.add_node('Stuttgart', pos=(0, 0), lol=True)
    G.add_node('Munich', pos=(500,500), lol=True)

    G.add_edge('Hamburg', 'Berlin', weight=2.0)
    G.add_edge('Berlin', 'Hamburg', weight=5.0)
    G.add_edge('Berlin', 'Munich', weight=5.0)
    G.add_edge('Munich', 'Berlin', weight=5.0)

    draw_graph(G)
