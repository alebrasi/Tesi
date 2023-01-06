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

skel = None
WalkedPath = namedtuple('WalkedPath', ['point_type', 'path'])

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
    
def add_nodes_and_edges(G, source_node, walked_path, max_res_err, min_points):
    if source_node not in G:
        G.add_node(source_node, node_type=PointType.NODE)
    
    prev_path_endpoint = source_node

    for path in walked_path.get_lstsq_paths(max_res_err, min_points):
        G.add_node(path.endpoint, node_type=PointType.NODE)
        G.add_edge(prev_path_endpoint, path.endpoint, weight=path.vector.angle, path=path.points)
        inv_vector = Vector.invert(path.vector)
        G.add_edge(path.endpoint, prev_path_endpoint, weight=inv_vector.angle, path=path.points[::-1])

        prev_path_endpoint = path.endpoint

def create_graph(seeds, skeleton, distances, max_residual_err=1.0, min_points_lstsq=4):
    """
    Creates a networkx graph of the plant
    """

    global skel
    queue = Queue()
    G = nx.DiGraph()

    skel = skeleton
    visited_nodes_neighbours = set()

    for seed in seeds:

        # Walks to the first node that is below the given seed point 
        # (we don't know if the point is actually the seed node)
        # which will be the seed node
        n = valid_neighbors(seed, root_neighbors, skel, return_idx=True)
        start_point = list(map(lambda n: n[1] if n[0] > 5 else None, n))[-1]
        _, path = walk_to_node(seed, start_point)
        cur_node = path.endpoint

        # If the seed node is already in the graph, it means that the previous
        # exploration (started from the previous seed) already visited all the plant
        # that grown up from the current seed
        if cur_node in G:
            G.nodes[cur_node]['node_type'] = PointType.SOURCE
            continue

        G.add_node(cur_node, node_type=PointType.SOURCE)
        queue.put(cur_node)

        while not queue.empty():
            cur_node = queue.get()

            node_neighbours = valid_neighbors(cur_node, root_neighbors, skel)
            node_neighbours = [ n for n in node_neighbours if n not in visited_nodes_neighbours ]

            for n in node_neighbours:
                visited_nodes_neighbours.add(n)
                endpoint_type, walked_path = walk_to_node(cur_node, n)
                if walked_path.penultimate_point not in visited_nodes_neighbours:
                    add_nodes_and_edges(G, cur_node, walked_path, max_residual_err, min_points_lstsq)
                    if endpoint_type is not PointType.TIP:
                        queue.put(walked_path.endpoint)
                    else:
                        G.nodes[walked_path.endpoint]['node_type'] = PointType.TIP

    H = nx.convert_node_labels_to_integers(G, label_attribute='pos')
    
    return H
