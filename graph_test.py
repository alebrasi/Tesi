import networkx as nx
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.graph import pixel_graph
import numpy as np
from utils import show_image
from skimage.morphology import skeletonize, medial_axis
import queue
import math

from path_extraction.prune import prune_skeleton_branches
from path_extraction.extract_roots import extract_plants_roots
from path_extraction.algorithm import RootsExtraction
from graph.graph_creation import create_graph, PointType
from graph.graph_drawing import draw_graph
from graph.algorithm import extract

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

#show_image(pruned * distance, cmap='magma')
# Done just for retrieving the valid root tips.
# Can be swapped with a more efficient function
tips, pruned = prune_skeleton_branches(seeds, pruned) 
#tips, pruned = prune_skeleton_branches(seeds, pruned) 

skel2 = boolean_matrix_to_rgb(pruned)

for tip in tips:
    skel2[tip] = (0, 255, 0)

#show_image([skel2, skel1])

G = create_graph(seeds, pruned, distance)
color_map = { 
                PointType.NODE:'lightgreen', 
                PointType.SOURCE:'red', 
                PointType.TIP:'orange' 
            }

node_color = [ color_map[G.nodes[node]['node_type']] for node in G ]

draw_graph(G, with_labels=True, node_color=node_color, node_size=20)

extract(G)

draw_graph(G, with_labels=True, node_color=node_color, node_size=20, invert_xaxis=False)
