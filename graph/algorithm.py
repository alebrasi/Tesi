import networkx as nx

from graph.create import PointType
from graph.plant import Plant
from graph.root import Root


def extract(G):
    seeds = [k for k, v in nx.get_node_attributes(G, 'node_type').items()
             if v == PointType.SOURCE]

    plants = []
    all_plants = []

    nx.set_edge_attributes(G, False, 'walked')

    Plant.attach_graph(G)
    Root.attach_graph(G)

    for seed in seeds:
        stem_start_node = min(G.neighbors(seed),
                              key=lambda n: G.nodes[n]['pos'][0]  # y coordinate of node
                              )
        tmp = Plant(seed, stem_start_node)
        plants.append(tmp)
        all_plants.append(tmp)

    while True:
        if all(map(lambda p: p.has_finished, all_plants)):
            break
        for plant in all_plants:
            plant.compute()

    return all_plants
