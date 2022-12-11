import queue

# Moore neighbor
def root_neighbors(p):
    y, x = p
    return [(y-1, x-1), (y-1, x), (y-1, x+1), (y, x-1), (y, x+1), (y+1, x-1), (y+1, x), (y+1, x+1)]

def seed_neighbors(p):
    y, x = p
    return [(y, x-1), (y, x+1), (y+1, x-1), (y+1, x), (y+1, x+1)]

def valid_neighbors(p, neighbor_func, skel):
    """
    Returns the valid node neighbors

    Parameters:
        p (tuple): The point
    """
    return [ i for i in neighbor_func(p) if skel[i] ]

def is_tip(n, skel):
    """
    Checks if a node is a tip

    Parameters:
        n (tuple): The node coordinate in (y, x) format
        skel (numpy.ndarray): Numpy boolean matrix containing the skeleton of the plants
    """
    return len(valid_neighbors(n, root_neighbors, skel)) == 1

def prune_skeleton_branches(seeds, skel, branch_threshold_len=6):
    """
    Returns the pruned skeleton 

    Parameters:
        seeds (list)              : List of the seeds points in (y, x) format
        skel (numpy.ndarray)      : Numpy boolean matrix containg the skeleton of the plants
        branch_threshold_len (int): Minimum lenght for a branch

    Returns:
        skel (numpy.ndarray): Numpy boolean matrix containing the pruned skeleton
    """
    q = queue.Queue()
    visited = set()
    for s in seeds:
        cur_node = valid_neighbors(s, seed_neighbors, skel)[0]
        # Check if the first seed neighbor is already visited
        if cur_node in visited:
            break

        cur_path = [cur_node]

        visited.add(s)

        # Pruning
        while True:
            n = valid_neighbors(cur_node, root_neighbors, skel)
            tmp = [i for i in n if i in visited] 

            for i in tmp:
                n.remove(i)

            visited.add(cur_node)

            # Pixel is an element of the root when has only a valid neighbour
            if len(n) == 1:
                cur_path.append(cur_node)
                cur_node = n[0]
                continue

            # Check if the pixel is a tip of the branch
            if is_tip(cur_node, skel):
                cur_path.append(cur_node)
                # Prune of the branch path if it matches the length threshold
                if len(cur_path) < branch_threshold_len:
                    for point in cur_path:
                        skel[point] = False
                cur_path = []
                if not q.empty():
                    cur_node = q.get()
                    cur_path.append(cur_node)
                else:
                    break
                continue
        
            # Pixel is a node
            cur_path = []
            for node in n:
                if node not in q.queue:
                    q.put(node)

            if not q.empty():
                cur_node = q.get()
                cur_path.append(cur_node)

    return skel
