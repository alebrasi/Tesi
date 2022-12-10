import networkx as nx
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.graph import pixel_graph
from scipy.sparse import coo_matrix
import numpy as np
from utils import show_image
from skimage.morphology import skeletonize, medial_axis
import queue


def ravel_idx(x, y, width):
    return (y * width) + x

# Moore neighbor
def root_neighbors(p):
    y, x = p
    return [(y-1, x-1), (y-1, x), (y-1, x+1), (y, x-1), (y, x+1), (y+1, x-1), (y+1, x), (y+1, x+1)]

def seed_neighbors(p):
    y, x = p
    return [(y, x-1), (y, x+1), (y+1, x-1), (y+1, x), (y+1, x+1)]

def valid_neighbors(p, neighbor_func, skel):
    return [ i for i in neighbor_func(p) if skel[i] ]

def is_tip(n, skel):
    return len(valid_neighbors(n, root_neighbors, skel)) == 1


skel = cv.imread('skel.png', cv.COLOR_BGR2GRAY)
skel, distance = medial_axis(skel, return_distance=True)

skel1 = skel.copy().astype(np.uint8)
skel1[skel1 == 1] = 255

a = skel * distance

skel1 = cv.cvtColor(skel1, cv.COLOR_GRAY2BGR)

#seeds = [ (140, 162), (130, 252) ]

s = (140, 162)
#s = (90, 166)
s = (130, 252)
skel1[s] = (255, 0, 0)

prev_node = s
cur_node = valid_neighbors(s, seed_neighbors, skel)[0]

q = queue.Queue()
cur_path = [cur_node]
PATH_LEN_THR = 6

visited = set()
visited.add(prev_node)

# Pruning
while True:
    n = valid_neighbors(cur_node, root_neighbors, skel)
    tmp = [i for i in n if i in visited] 

    for i in tmp:
        n.remove(i)

    visited.add(cur_node)

    # Pixel is an element of the root when has only a valid neighbour
    if len(n) == 1:
        cur_path.append(cur_node)
        cur_node = n[0]
        continue

    if is_tip(cur_node, skel):
        cur_path.append(cur_node)
        if len(cur_path) < PATH_LEN_THR:
            for point in cur_path:
                skel1[point] = (0, 0, 0)
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



show_image([skel1, skel])

"""
print(len(skel[skel != 0]))
G, nodes = pixel_graph(skel, connectivity=2)
print(nodes)
print(G)
print(np.argwhere(nodes == ravel_idx(165, 145, 500)))
"""




"""
print(ravel_idx(499, 499, 500))

pt1 = ravel_idx(165, 145, 500)
pt2 = ravel_idx(165, 91, 500)
print(G[pt1])
"""



