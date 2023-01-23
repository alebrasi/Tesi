import queue
import numpy as np
import cv2 as cv
from utils import show_image

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

    n = neighbor_func(p)
    valid = [ skel[i] for i in n ]
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