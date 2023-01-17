import networkx as nx
from itertools import filterfalse
from graph.graph_creation import PointType
from graph.plant import Plant
from graph.root import Root

def extract(G, angle_par_name='weight'):
    ANGLE = angle_par_name

    seeds = [ k for k, v in nx.get_node_attributes(G, 'node_type').items() 
                if v == PointType.SOURCE ]

    plants = []

    Plant.attach_graph(G)
    Root.attach_graph(G)

    for seed in seeds:
        tip_start_node = min(G.neighbors(seed), 
                             key=lambda n: G.edges[(seed, n)][ANGLE])
        plants.append(Plant(seed, tip_start_node))

    nx.set_edge_attributes(G, False, 'walked')

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