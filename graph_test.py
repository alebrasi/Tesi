import networkx as nx
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.graph import pixel_graph
import numpy as np
from utils import show_image
from skimage.morphology import skeletonize, medial_axis
import queue
import math
import time

from path_extraction.prune import prune_skeleton_branches
from path_extraction.extract_roots import extract_plants_roots
from path_extraction.algorithm import RootsExtraction

from graph.graph_creation import create_graph, PointType
from graph.graph_drawing import draw_graph
from graph.algorithm import extract
from graph.plant import Plant

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

plants = extract(G)

all_p = np.ones((500, 500, 3))
cv.namedWindow('image', cv.WINDOW_FULLSCREEN)

for i, plant in enumerate(plants, 0):
    print(f'Plant: {i}')
    print(f'Num roots: {len(plant.roots)}')
    mask = np.zeros((500, 500, 3))
    for root in plant.roots:
        print(root.edges)
        print(root._edges)
        print(root._split_node)
        print('\n\n\n')
        color1 = (list(np.random.choice(range(256), size=3)))  
        points = np.array(root.points)
        for point in points:
            #print(point)
            y, x = point
            
            mask[y, x, i] = 255
            all_p[y, x, :] = color1
            cv.imshow('image', all_p.astype(np.uint8))
            cv.waitKey(1)
            time.sleep(0.01)
        #plt.imshow(mask.astype(np.uint8))
        #plt.show()
print('Done!')
while True:
    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()
#plt.imshow(all_p)
#plt.show()

draw_graph(G, with_labels=True, node_color=node_color, node_size=20, invert_xaxis=False)