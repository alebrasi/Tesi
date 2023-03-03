from path_extraction.prune import valid_neighbors, seed_neighbors, root_neighbors, is_tip, get_nodes
from path_extraction.path import Path
from path_extraction.vector_utils import Vector

from collections import namedtuple
from enum import Enum
import numpy as np
from queue import Queue
import math

import networkx as nx
import matplotlib.pyplot as plt

import cv2 as cv

class PointType(Enum):
    TIP = 0,
    NODE = 1,
    SOURCE = 2

skel = None
WalkedPath = namedtuple('WalkedPath', ['point_type', 'path'])

def walk_to_node(sender_p, start_p):
    # TODO: Cambiare il nome alla funzione
    """
    Returns a list of points (path) that are between the point `p` and the first 
    available node or tip

    Parameteres:
        sender_p (tuple): The `father` of the starting point
        start_p (tuple): The starting point, which is the `son` (or neighbor) of `sender_p` 

    Returns:
        path: A list containing the points walked
    """

    global skel

    prev_point = sender_p
    cur_point = start_p
    path = [sender_p]
    res = PointType.NODE

    while True:
        n = valid_neighbors(cur_point, root_neighbors, skel)
        if prev_point in n: n.remove(prev_point)

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
    """
    Add nodes and edges given a path that has lstsq paths in it.
    Note: the vector associated to the edge is flipped horizontally, 
          so the calculated angle is not right if the graph is plotted

    Parameters:
        G (networkx.DiGraph) : the directed graph
        source_node (tuple) : coordinates of the source node in (y, x) format
        walked_path (Path) : the path
        max_res_err (float) : maximum residual error for obtaining a least squared path
        min_point (int) : minimum number that must contains a least squared path
    """
    if source_node not in G:
        G.add_node(source_node, node_type=PointType.NODE)
    
    prev_path_endpoint = source_node

    for path in walked_path.get_lstsq_paths(max_res_err, min_points):
        # TODO: the edge and its opposite have the vectors swapped in order to account 
        # for the (y, x) coordinate system, where (0, 0) is located in the top-left corner.
        # Modify this code or the vector class in order to account for it. Or leave as it is.

        G.add_node(path.endpoint, node_type=PointType.NODE)
        inv_vector = Vector.invert(path.vector)
        G.add_edge(prev_path_endpoint, 
                    path.endpoint, 
                    weight=inv_vector.angle, 
                    path_points=path.points, 
                    length=len(path.points),
                    vector=inv_vector)

        # Adds the opposite edge
        G.add_edge(path.endpoint, 
                    prev_path_endpoint, 
                    weight=path.vector.angle, 
                    path_points=path.points[::-1],     # Just reverse the order of the points
                    length=len(path.points),
                    vector=path.vector)

        prev_path_endpoint = path.endpoint


def find_forking_node(seed, skel):
    """
    Finds the forking node, where roots will start.
    It works by walking all the seed neighbors until a node/tip is found.
    From the walked paths, the lowest point (in this case higher because the y axis grows going down)
    is searched in the paths and, from the path that it is contained, the endpoint is taken.
    The endpoint will be the forking node

    Parameters:
    seed (tuple): point, in (y, x) format, that describes where the seed is located
    skel (numpy.ndarray): boolean matrix containing the skeletonized image
    """
    neighbors = valid_neighbors(seed, root_neighbors, skel)    

    lowest_y = float('-inf')
    lowest_point = None

    for n in neighbors:
        _, path = walk_to_node(seed, n)
        # The y axis grows going down
        lowest_y_p = max(path.points, 
                         key=lambda p: float(p[0])  # key = y coordinate of point
                         )[0]   # grabs only the y coordinate

        if lowest_y_p > lowest_y:
            lowest_point = path.endpoint
            lowest_y = lowest_y_p

    return lowest_point

def create_graph2(seeds, skeleton, distances, max_residual_err=1.0, min_points_lstsq=4):
    """
    Creates a networkx directed graph of the plant.
    The plant is approximated with least squared lines.

    Nodes attributes:
        - pos (tuple) : coordinates of the node
        - node_type (PointType) : type of the node
    
    Edges attributes:
        - weight (int) : angle between the two connected nodes
        - lenght (int) : number of points between the two nodes
        - path_points (list) : points, in (y, x) format, between the two nodes
        - vector (Vector) : vector describing the direction of the ege

    Parameters:
        seeds (list) : positions of the seeds, in (y, x) format
        skeleton (numpy.ndarray) : boolean matrix that is the skeletonization of the segmented image
        max_residual_err (float) : maximum residual error for obtaining a least squared path
        min_points_lstsq (int) : minimum number of points that must contains a least squared path
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
        cur_node = find_forking_node(seed, skel)

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
                endpoint_type, walked_path = walk_to_node(cur_node, n)
                if walked_path.penultimate_point not in visited_nodes_neighbours:
                    add_nodes_and_edges(G, 
                                        cur_node, 
                                        walked_path, 
                                        max_residual_err, 
                                        min_points_lstsq)

                    if endpoint_type is not PointType.TIP:
                        if walked_path.penultimate_point not in node_neighbours:
                            queue.put(walked_path.endpoint)
                    else:
                        G.nodes[walked_path.endpoint]['node_type'] = PointType.TIP
                
                visited_nodes_neighbours.add(n)
                #visited_nodes_neighbours.add(walked_path.penultimate_point)

    """
    Very cheap labeling fix when there are loops like this:
    
            #
           #
          # #
         #  #
        #  #
         ##
    """
    # TODO: Fare un fix migliore
    for node in G.nodes:
        if G.nodes[node]['node_type'] == PointType.NODE and len(list(G.neighbors(node))) == 1:
            G.nodes[node]['node_type'] = PointType.TIP

    all_edges_len = [ G.edges[e]['length'] for e in G.edges ]
    max_edge_len = max(all_edges_len)
    G.graph['max_edge_len'] = (max_edge_len/0.8) - max_edge_len

    H = nx.convert_node_labels_to_integers(G, label_attribute='pos')
    
    return H

def create_graph(seeds, skeleton, distances, max_residual_err=1.0, min_points_lstsq=4):
    """
    Creates a networkx directed graph of the plant.
    The plant is approximated with least squared lines.

    Nodes attributes:
        - pos (tuple) : coordinates of the node
        - node_type (PointType) : type of the node
    
    Edges attributes:
        - weight (int) : angle between the two connected nodes
        - lenght (int) : number of points between the two nodes
        - path_points (list) : points, in (y, x) format, between the two nodes
        - vector (Vector) : vector describing the direction of the ege

    Parameters:
        seeds (list) : positions of the seeds, in (y, x) format
        skeleton (numpy.ndarray) : boolean matrix that is the skeletonization of the segmented image
        max_residual_err (float) : maximum residual error for obtaining a least squared path
        min_points_lstsq (int) : minimum number of points that must contains a least squared path
    """

    global skel
    queue = Queue()
    G = nx.DiGraph()
    visited_neighbors = set()

    skel = skeleton

    nodes_idx = get_nodes(skel)
    bruh = cv.cvtColor(skel.astype(np.uint8), cv.COLOR_GRAY2BGR)
    bruh[skel, ...] = [255, 255, 255]

    for node in nodes_idx:
        y, x = node
        bruh[y, x, :] = [0, 255, 0]
        node = tuple(node)
        neighbors = valid_neighbors(node, root_neighbors, skel)
        neighbors = [ n for n in neighbors if n not in visited_neighbors ]

        for n in neighbors:
            endpoint_type, path = walk_to_node(node, n)
            points = path.points[1:]        # Skips the first point, which is the node
            visited_neighbors.add(path.penultimate_point)
            add_nodes_and_edges(G, node, path, max_residual_err, min_points_lstsq)
            if endpoint_type == PointType.TIP:
                G.nodes[path.endpoint]['node_type'] = PointType.TIP
    plt.imshow(bruh)
    plt.show()
    for seed in seeds:

        # Walks to the first node that is below the given seed point 
        # (we don't know if the point is actually the seed node)
        # which will be the seed node
        cur_node = find_forking_node(seed, skel)

        # If the seed node is already in the graph, it means that the previous
        # exploration (started from the previous seed) already visited all the plant
        # that grown up from the current seed
        if cur_node in G:
            G.nodes[cur_node]['node_type'] = PointType.SOURCE
            continue


    all_edges_len = [ G.edges[e]['length'] for e in G.edges ]
    max_edge_len = max(all_edges_len)
    #G.graph['max_edge_len'] = (max_edge_len/0.8) - max_edge_len
    G.graph['max_edge_len'] = (max_edge_len/0.8)

    H = nx.convert_node_labels_to_integers(G, label_attribute='pos')
    
    return H
