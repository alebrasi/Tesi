import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import filterfalse
import random

from graph.graph_creation import PointType
from graph.plant import Plant
from graph.root import Root

def extract(G, angle_par_name='weight'):
    ANGLE = angle_par_name

    seeds = [ k for k, v in nx.get_node_attributes(G, 'node_type').items() 
                if v == PointType.SOURCE ]

    plants = []
    all_plants = []

    nx.set_edge_attributes(G, False, 'walked')

    Plant.attach_graph(G)
    Root.attach_graph(G)

    for seed in seeds:
        tip_start_node = min(G.neighbors(seed), 
                             key=lambda n: G.edges[(seed, n)][ANGLE])
        tmp = Plant(seed, tip_start_node)
        plants.append(tmp)
        all_plants.append(tmp)


    print("Dita incrociate")

    while len(plants) > 0:
        """
        for i in range(len(plants)):
            if not (plants[i]).compute():
                plant = plants[i]
                plants.remove(plant)
                #continue
        """
        plants[:] = filterfalse(lambda p: not p.compute(), plants)

    all_p = np.zeros((500, 500, 3))

    for i, plant in enumerate(all_plants, 0):
        print(f'Plant: {i}')
        print(f'Num roots: {len(plant.roots)}')
        mask = np.zeros((500, 500, 3))
        for root in plant.roots:
            print(root.edges)
            print(root._edges)
            print(root._split_node)
            print('\n\n\n')
            points = np.array(root.points)
            for point in points:
                #print(point)
                y, x = point
                
                mask[y, x, i] = 255
                all_p[y, x, i] = 255
            plt.imshow(mask.astype(np.uint8))
            plt.show()
            #mask[points] = 255

    plt.imshow(all_p)
    plt.show()

    print("Ohhhhhhh my godddd")

    #print(G.edges(data=True))


    # funzioni utili: mapped queue (min heap) in networkx.utils

    """
    for p in plants:
        n = list(nx.neighbors(G, p))
        #print(nx.edges(G, p))
        for n1 in n:
            print(G.adj[n1][p]['weight'])
        print('\n\n')
    """

    """
    while len(seeds) > 0:
        for seed in seeds:

            pass
    """