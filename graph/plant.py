import networkx as nx
import copy
from graph.root import Root

class Plant:

    G = None

    @staticmethod
    def attach_graph(G):
        Plant.G = G

    def __init__(self, seed, tip_start_node):
        self._seed = seed
        self._roots = set()
        self._current_root = None
        self._tip_start_node = tip_start_node
        self._seed_node_neighbors = list(type(self).G.neighbors(seed))
        self._seed_node_neighbors.remove(tip_start_node)

        self._init_new_root()

    def _init_new_root_starting_from_seed(self):
        n = self._seed_node_neighbors.pop()
        return Root(self, (self._seed, n))

    def _init_new_root(self):
        new_root = None
        if len(self._seed_node_neighbors) == 0:
            """
            root = min(self._roots, 
                        key=lambda r: r.highest_non_walked_node[2][0], 
                        default=None)
            """

            root = min(
                        filter(lambda r: r[1] != None, 
                            map(lambda r: (r, r.highest_non_walked_node), self._roots)), 
                    key=lambda k: k[1][2][0],
                    default=None)

            if root == None:
                return False

            node = root[0].highest_non_walked_node
            print('plant: ', node)
            # FIXME: Fare la deepcopy degli edge della  root fino a quel nodo
            #new_root = copy.deepcopy(root)
            new_root = root[0].copy_until_node(node[0])
            new_root._add_edge((node[0], node[1]))
        else:
            new_root = self._init_new_root_starting_from_seed()

        self._roots.add(new_root)
        self._current_root = new_root

        return True
        
        #return new_root

    def compute(self):
        if not self._current_root.explore():
            if not self._init_new_root():
                return False

        return True
        