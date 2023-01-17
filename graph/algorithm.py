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

    return all_plants


    print("Ohhhhhhh my godddd")

    #print(G.edges(data=True))


    # funzioni utili: mapped queue (min heap) in networkx.utils