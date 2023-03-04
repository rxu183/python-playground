# Richard Xu
# rgx1
# COMP 182 Spring 2023 - Homework 4, Problem 3

# You may NOT import anything apart from already imported libraries.
# You can use helper functions from comp182.py and provided.py, but they have
# to be copied over here.

from collections import *
def compute_largest_cc_size(g: dict) -> int:
    """
    Compute largest component size.

    Arguments:
    g        -- a graph

    Returns:
    Size of largest component
    """
    # This function takes in a dictionary, and is expected to return an int.
    # How is the graph represented as a graph?
    # It must be an adjacency list - node as keys, values are sets of neighbors reportedly
    # For each node? Ok, I'll run with that assumption
    
    max = 0
    current = 0
    v = {}
    for j in g.keys():
        v[j] = False
    for i in g.keys():
        current = compute_num_unvisited_nodes(g, i, v)
        if current > max:
            max = current
    return max

def compute_num_unvisited_nodes(g, i, v):
    """
    Computes number of unvisited node within a particular component

    Arguments:
    g        -- a graph
    i -- key value of node who's component is to be explored
    v -- a visited array, denoting which nodes have already been traversed

    Returns:
    size of component if component has not been visited, 0 otherwise.
    """
    if v[i] == True:
        return 0
    result = 1
    v[i] = True
    q = deque()
    q.append(i)
    while len(q) != 0:
        j = q.pop()
        for h in g[j]:
            if v[h] == False:
                result = result + 1
                v[h] = True
                q.append(h)
    return result

#def main():
#    g = {0:[1,2,3], 1: [0, 2], 2: [0, 1], 3: [0]}
#    g_1 = {0:[1,2], 1:[0], 2:[0], 3:[]}
#    print(compute_largest_cc_size(g))
#    print(compute_largest_cc_size(g_1))
#    return 0
#main()