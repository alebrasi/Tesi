import networkx as nx
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.graph import pixel_graph
from scipy.sparse import coo_matrix
import numpy as np
from utils import show_image
from skimage.morphology import skeletonize, medial_axis
import queue
import math
from path_extraction.prune import prune_skeleton_branches
from path_extraction.extract_roots import extract_plants_roots
from path_extraction.algorithm import RootsExtraction


def ravel_idx(x, y, width):
    return (y * width) + x

def boolean_matrix_to_rgb(m):
    m = m.copy().astype(np.uint8)
    m[m == 1] = 255
    return cv.cvtColor(m, cv.COLOR_GRAY2BGR)


skel = cv.imread('skel.png', cv.COLOR_BGR2GRAY)
# TODO: Controllare il comportamento di medial_axis
skel, distance = medial_axis(skel, return_distance=True)

skel1 = boolean_matrix_to_rgb(skel)

a = skel * distance

seeds = [(164, 172), (166, 255), (166, 340)]

# Skeleton branch prune
_, pruned = prune_skeleton_branches(seeds, skel.copy())

show_image(pruned * distance, cmap='magma')
# Done just for retrieving the valid root tips.
# Can be swapped with a more efficient function
tips, pruned = prune_skeleton_branches(seeds, pruned) 
tips, pruned = prune_skeleton_branches(seeds, pruned) 

skel2 = boolean_matrix_to_rgb(pruned)

for tip in tips:
    skel2[tip] = (0, 255, 0)

show_image([skel2, skel1])

#extract_plants_roots(seeds, pruned)

s = seeds[2]
s = (170, 337)
alg = RootsExtraction(pruned, distance)
stem_path, paths = alg.extract_roots(s)
colors = [(255, 0, 0), (0,255,0), (0, 0, 255)]
skel1 = boolean_matrix_to_rgb(pruned)

for i, p in enumerate(stem_path.points):
    skel1[p] = colors[0]

show_image(skel1)

for path in paths:
    for i, p in enumerate(path):
        for point in p.points:
            skel1[point] = colors[i%len(colors)]


show_image(skel1)
#show_image(pruned * distance, cmap='magma')

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



