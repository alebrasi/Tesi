import queue
import numpy as np
import cv2 as cv
from utils import show_image
from path_extraction.path import Path
from enum import Enum

class PointType(Enum):
    TIP = 0,
    NODE = 1,
    SOURCE = 2
# TODO: Spostare tutto nella classe RootsExtraction

# Moore neighbor
def root_neighbors(p):
    y, x = p
    return [(y-1, x-1), (y-1, x), (y-1, x+1), (y, x-1), (y, x), (y, x+1), (y+1, x-1), (y+1, x), (y+1, x+1)]

def seed_neighbors(p):
    y, x = p
    return [(y, x-1), (y, x+1), (y+1, x-1), (y+1, x), (y+1, x+1)]

def idx(y, x, s=3):
    return (y * s) + x

# TODO: Fare refactor
def valid_neighbors(p, neighbor_func, skel, return_idx=False):
    """
    Returns the valid node neighbors

    Parameters:
        p (tuple): The point
    """
    h, w = skel.shape[:2]
    is_valid = lambda p: p[0] < h and p[0] >= 0 and p[1] < w and p[1] >= 0

    n = neighbor_func(p)
    #valid = [ skel[i] for i in n ]
    valid = [ skel[i] if is_valid(i) else False for i in n ]
    valid[idx(1, 1)] = False

    if valid[idx(0, 1)]:
        valid[idx(0, 0)] = valid[idx(0, 2)] = False
    if valid[idx(1, 0)]:
        valid[idx(0, 0)] = valid[idx(2, 0)] = False
    if valid[idx(2, 1)]:
        valid[idx(2, 0)] = valid[idx(2, 2)] = False
    if valid[idx(1, 2)]:
        valid[idx(0, 2)] = valid[idx(2, 2)] = False
    #return [ i for i in neighbor_func(p) if skel[i] ]
    if return_idx:
        return [ (i, n1) for i, n1 in enumerate(n) if valid[i] ]
        
    return [ n1 for i, n1 in enumerate(n) if valid[i] ]

def is_tip(n, skel):
    """
    Checks if a node is a tip

    Parameters:
        n (tuple): The node coordinate in (y, x) format
        skel (numpy.ndarray): Numpy boolean matrix containing the skeleton of the plants
    """
    return len(valid_neighbors(n, root_neighbors, skel)) == 1

def prune_skeleton_branches(seeds, skel, branch_threshold_len=6):
    """
    Returns the pruned skeleton 

    Parameters:
        seeds (list)              : List of the seeds points in (y, x) format
        skel (numpy.ndarray)      : Numpy boolean matrix containg the skeleton of the plants
        branch_threshold_len (int): Minimum lenght for a branch

    Returns:
        skel (numpy.ndarray): Numpy boolean matrix containing the pruned skeleton
    """
    q = queue.Queue()
    visited = set()
    tips = set()
    for s in seeds:
        n1 = valid_neighbors(s, root_neighbors, skel)
        # Get a random neighbor (in this case the last one in the array) and use it as starting point.
        # Adds the others to the queue to explore
        cur_node = n1.pop()
        for n2 in n1:
            q.put(n2)

        # Check if the first seed neighbor is already visited
        if cur_node in visited:
            break

        cur_path = [cur_node]

        visited.add(s)

        # Pruning
        while True:
            n = valid_neighbors(cur_node, root_neighbors, skel)
            n = [ i for i in n if i not in visited ]
           
            visited.add(cur_node)

            # Pixel is an element of the root when has only a valid neighbour
            if len(n) == 1:
                cur_path.append(cur_node)
                cur_node = n[0]
                continue

            # Check if the pixel is a tip of the branch
            if is_tip(cur_node, skel):
                cur_path.append(cur_node)
                tips.add(cur_node)
                # Prune of the branch path if it matches the length threshold
                if len(cur_path) < branch_threshold_len:
                    for point in cur_path:
                        skel[point] = False
                cur_path = []
                if not q.empty():
                    cur_node = q.get()
                    cur_path.append(cur_node)
                else:
                    break
                continue
        
            # Pixel is a node
            cur_path = []
            for node in n:
                if node not in q.queue:
                    q.put(node)

            if not q.empty():
                cur_node = q.get()
                cur_path.append(cur_node)

    return tips, skel


def prune2(skel, thr):
    s = skel.copy()
    s = s.astype(np.uint8)
    orig = s.copy()
    kers = []
    kers.append(np.array([[0, -1, -1],
                          [ 1, 1, -1],
                          [0, -1, -1]]))
    
    kers.append(np.array([[0, 1, 0],
                          [ -1, 1,  -1],
                          [ -1, -1,  -1]]))

    kers.append(np.array([[-1, -1, 0],
                          [-1, 1,  1],
                          [-1, -1, 0]]))

    kers.append(np.array([[ -1, -1,  -1],
                          [ -1, 1,  -1],
                          [0, 1, 0]]))

    kers.append(np.array([[1, -1, -1],
                          [-1, 1, -1],
                          [-1, -1, -1]]))

    kers.append(np.array([[-1, -1, 1],
                          [-1, 1, -1],
                          [-1, -1, -1]]))

    kers.append(np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [-1, -1, 1]]))

    kers.append(np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [1, -1, -1]]))

    # Thinning
    for i in range(thr):
        for ker in kers:
            morphed = cv.morphologyEx(s, cv.MORPH_HITMISS, ker)
            s -= morphed

    # Find end points
    endpoints = np.zeros_like(s)
    for ker in kers:
        tmp = cv.morphologyEx(s, cv.MORPH_HITMISS, ker)
        endpoints += tmp

    # Dilate end points
    dilate_ker = np.ones((3, 3))
    endpoints = cv.dilate(endpoints, dilate_ker, iterations=thr)
    endpoints = cv.bitwise_and(endpoints, endpoints, mask=orig)
    # Union
    s += endpoints
    return s

def walk_to_node(skel, sender_p, start_p):
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

    #global skel

    prev_point = sender_p
    cur_point = start_p
    path = [sender_p]
    res = PointType.NODE

    while True:
        n = valid_neighbors(cur_point, root_neighbors, skel)
        #n.remove(prev_point)
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

    return res, Path(path)

def get_nodes(skel):
    kers = list()
    skel = skel.copy().astype(np.uint8)
    nodes = np.zeros_like(skel)

    """
    Kernels took from:
    https://stackoverflow.com/questions/43037692/how-to-find-branch-point-from-binary-skeletonize-image
    """
    kers.append(np.array([[0, 1, 0], 
                          [1, 1, 1], 
                          [0, 0, 0]]))

    kers.append(np.array([[1, 0, 1], 
                          [0, 1, 0], 
                          [1, 0, 0]]))

    kers.append(np.array([[1, 0, 1], 
                          [0, 1, 0], 
                          [0, 1, 0]]))

    kers.append(np.array([[0, 1, 0], 
                          [1, 1, 0], 
                          [0, 0, 1]]))

    kers.append(np.array([[0, 0, 1], 
                          [1, 1, 1], 
                          [0, 1, 0]]))

    kers = [np.rot90(kers[i], k=j) for i in range(5) for j in range(4)]

    kers.append(np.array([[0, 1, 0], 
                          [1, 1, 1], 
                          [0, 1, 0]]))
    
    kers.append(np.array([[1, 0, 1], 
                          [0, 1, 0], 
                          [1, 0, 1]]))

    for ker in kers:
        """
        When the hitmiss transformation is applied, the border type 
        is set to constant with value=0 in order to remove false positive nodes
        when there are pixels close to the border.

        E.g.
        Skeleton before finding the nodes:

        BORDER --->  ##########################
                                   ******     #
                           *********          #
                         **         *         #
                        *            *        #

        Where '*' indicates the pixels of the skeleton

        After finding the nodes with border value = 255 (or True):

        BORDER --->  ##########################
                                   x*****     #
                           ********°          #
                         *°         *         #
                        *            *        #

        Where:
            - '°' indicate the correct identified nodes
            - 'x' indicate the wrong identified node
        """
        nodes += cv.morphologyEx(skel, 
                                 cv.MORPH_HITMISS, 
                                 ker, 
                                 borderType=cv.BORDER_CONSTANT,
                                 borderValue=0
                                )
    
    # Grab coordinates of the nodes
    nodes_idx = np.argwhere(nodes.astype(bool))

    return nodes_idx

def prune3(skel, branch_threshold, work_on_copy=True):
    """
    Prunes the branches from the skeleton that are equal or shorter than the threshold.

    Parameters:
        - skel (numpy.ndarray): Boolean numpy matrix that describes the skeleton
        - branch_threshold (int): Branch threshold
        - work_on_copy (bool): Whether or not the prune must be done on the original skeleton
    """
    pruned = skel
    if work_on_copy:
        pruned = skel.copy()

    nodes_idx = get_nodes(skel)
    
    for node in nodes_idx:
        node = tuple(node)
        neighbors = valid_neighbors(node, root_neighbors, pruned)
        for n in neighbors:
            endpoint_type, path = walk_to_node(pruned, node, n)
            points = path.points[1:]        # Skips the first point, which is the node
            if endpoint_type == PointType.TIP and len(points) <= branch_threshold:
                for point in points:
                    pruned[point] = False

    return pruned