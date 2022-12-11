import networkx as nx
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.graph import pixel_graph
from scipy.sparse import coo_matrix
import numpy as np
from utils import show_image
from skimage.morphology import skeletonize, medial_axis
import queue
from path_extraction.prune import prune_skeleton_branches


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

seeds = [ (140, 162), (130, 252), (132, 367) ]

pruned = prune_skeleton_branches(seeds, skel.copy())

show_image([pruned, skel])



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



