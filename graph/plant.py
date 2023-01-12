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
        return Root((self._seed, n))

    def _init_new_root(self):
        if len(self._seed_node_neighbors) == 0:
            root = min(self._roots, 
                        key=lambda r: r.highest_non_walked_node, 
                        default=None)

            if root == None:
                return False

            node = root.highest_non_walked_node
            new_root = copy.deepcopy(root)
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
        