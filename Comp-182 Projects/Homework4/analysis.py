# Richard Xu
# rgx1
# COMP 182 Spring 2023 - Homework 4, Problem 3

# You can import any standard library, as well as Numpy and Matplotlib.
# You can use helper functions from comp182.py, provided.py, and autograder.py,
# but they have to be copied over here.
import random
import numpy
import pylab
import copy
import matplotlib.pyplot as plt
from collections import *

def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.
 
    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is positive.
    An empty graph will be returned in all other cases.
 
    Arguments:
    num_nodes -- The number of nodes in the returned graph.
 
    Returns:
    A complete graph in dictionary form.
    """
    result = {}
         
    for node_key in range(num_nodes):
        result[node_key] = set()
        for node_value in range(num_nodes):
            if node_key != node_value: 
                result[node_key].add(node_value)
 
    return result

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

def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)

def read_graph(filename):
    """
    Read a graph from a file.  The file is assumed to hold a graph
    that was written via the write_graph function.

    Arguments:
    filename -- name of file that contains the graph

    Returns:
    The graph that was stored in the input file.
    """
    with open(filename) as f:
        g = eval(f.read())
    return g

def total_degree(g):
    """
    Compute total degree of the undirected graph g.

    Arguments:
    g -- undirected graph

    Returns:
    Total degree of all nodes in g
    """
    return sum(map(len, g.values()))

def upa(n, m):
    """
    Generate an undirected graph with n node and m edges per node
    using the preferential attachment algorithm.

    Arguments:
    n -- number of nodes
    m -- number of edges per node

    Returns:
    undirected random graph in UPAG(n, m)
    """
    g = {}
    if m <= n:
        g = make_complete_graph(m)
        for new_node in range(m, n):
            # Find <=m nodes to attach to new_node
            totdeg = float(total_degree(g))
            nodes = list(g.keys())
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)

            # Add new_node and its random neighbors
            g[new_node] = set()
            for idx in mult:
                node = nodes[idx]
                g[new_node].add(node)
                g[node].add(new_node)
    return g            

def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.

    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1

    Returns: 
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.  
    """
    ### select ntrials elements randomly
    mult = numpy.random.multinomial(ntrials, probs)

    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    return result

def erdos_renyi(n, p):
    """
    Generate a random Erdos-Renyi graph with n nodes and edge probability p.

    Arguments:
    n -- number of nodes
    p -- probability of an edge between any pair of nodes

    Returns:
    undirected random graph in G(n, p)
    """
    g = {}

    ### Add n nodes to the graph
    for node in range(n):
        g[node] = set()

    ### Iterate through each possible edge and add it with 
    ### probability p.
    for u in range(n):
        for v in range(u+1, n):
            r = random.random()
            if r < p:
                g[u].add(v)
                g[v].add(u)

    return g

def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.  

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals

def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)

def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments: 
    data     -- a list of dictionaries, each of which will be plotted 
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

def show():
    """
    Do not use this function unless you have trouble with figures.

    It may be necessary to call this function after drawing/plotting
    all figures.  If so, it should only be called once at the end.

    Arguments:
    None

    Returns:
    None
    """
    plt.show()

def remove_node(g, i):
    """
    Inputs: Graph, node to be removed
    Outputs: Nothing. Graph is modified already.
    """
    for set_neighbor in g.values():
        if i in set_neighbor:
            set_neighbor.remove(i)
    del g[i] #I don't think this is a big deal, but maybe if you remove an element from graph based on something that already exists? Might break future code.

def random_attack(g):
    """
    Inputs:
    Takes in a graph in the form of a dictionary that maps nodes to sets of neighbors.
    Outputs:
    Dictionary of counter mapped to number representing largest component of graph as nodes are removed one by one randomly
    """
    result = {}
    index = 0
    threshold = len(g.keys()) * .8
    while len(g) > threshold:
        result[index] = compute_largest_cc_size(g)
        rand_node = random.choice(list(g.keys()))
        remove_node(g,rand_node)
        index = index + 1
    return result

def targeted_attack(g):
    """
    Inputs:
    Takes in a graph that maps nodes to sets of neighbors.
    Outputs:
    Dictionary of counter mapped to number representing size of largest component in graph as nodes are removed one by one with greatest degree.
    """
    result = {}
    index = 0
    threshold = len(g) * .8
    while len(g) > threshold:
        result[index] = compute_largest_cc_size(g)
        max = 0
        target_node = 0
        for node in g.keys():
            if len(g[node]) > max:
                target_node = node
                max = len(g[node])
        remove_node(g, target_node)
        index = index + 1

    return result

def main (): 
    
    test_rewo = read_graph("rf7.repr")
    total_nodes = len(test_rewo.keys())
    total_edges = int(total_degree(test_rewo)/2)
    #print(total_nodes, total_edges)
    test_erdo = erdos_renyi(total_nodes, float(total_edges/((total_nodes*(total_nodes-1))/2)))
    test_upa = upa(total_nodes, int(total_edges/total_nodes))
    
    test_tar_erdo = copy_graph(test_erdo)
    test_tar_upa = copy_graph(test_upa)
    test_tar_rewo = copy_graph(test_rewo)

    e = random_attack(test_erdo)
    u = random_attack(test_upa)
    r = random_attack(test_rewo)
    
    e_t = targeted_attack(test_tar_erdo)
    u_t = targeted_attack(test_tar_upa)
    r_t = targeted_attack(test_tar_rewo)

    plot_lines([e, u, r, e_t, u_t, r_t], "Analysis of Graphs", "Number of Nodes Removed", "Size of Largest Connected Component", ["Random Attack on Erdo", "Random Attack on Upa", "Random Attack on Real World Graph", "Targeted Attack on Erdo", "Targeted Attack on Upa", "Targeted Attack on Real World Graph"])

    show()

main()
