# Richard Xu
# rgx1
# COMP 182 Spring 2021 - Homework 6, Problem 2

# You can import any standard library, as well as Numpy and Matplotlib.
# You can use helper functions from provided.py, and autograder.py,
# but they have to be copied over here.

# Your code here...
import os

import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple
from collections import *
from copy import *

index = [0]
def reverse_digraph_representation(graph: dict) -> dict: 
    """
    Reverses the representation so keys are incoming but values are dictionary of nodes coming into the node along with weights
    Inputs:
    graph: Standard weighted digraph representation

    Output:
    res: Reversed weighted digraph representation
    """
    #Ok, now that we understand the problem, let us try and think about how we'd do it.
    
    res = {}
    #First, initialize all nodes to map to an empty dictionary(the keys of the two representations remains the same)
    for key in graph.keys():
        res[key] = {}
    for destDict in graph.items(): #Outer Map
        sourceNode = destDict[0] #This should be the source node we need
        for toBeAdded in destDict[1].items(): #This is the second dictionary, the one of nodes connected and weights
            res[toBeAdded[0]][sourceNode] = toBeAdded[1]  #The head of the arrow is pointed at toBeAdded[0], and set weight in internal
    return res

def minimizeNode(rgraph, node):
    """
    Does the work of actually modifying by calculating the minimum. Note that it rounds to avoid floating errors.
    Inputs:
    rgraph: Reversed weighted digraph representation (dictionary)
    node: The node who's incoming edges we seek to minimize
    
    Output: 
    Nothing - it modifies the rgraph entries directly (rounding to avoid floating-point arithmetic errors)
    """
    if len(rgraph[node].values()) == 0:
        return None
    minW = min(rgraph[node].values())
    for element in rgraph[node].items():
        rgraph[node][element[0]]  = round(rgraph[node][element[0]] - minW, 10)

def modify_edge_weights(rgraph: dict, root: int) -> None:
    """
    Modifies the weights on the graph by subtracting based on the minimal edge length.
    Inputs: 
    rgraph: Reversed digraph representation (dictionary)
    root: integer representing the root node

    Outputs:
    Nothing - It modifies the original rgraph in the following way:
    Finds smallest length edge value, then subtracts this from every incoming edge to the node

    """
    #rev = reverse_digraph_representation(rgraph)
    #For all nodes with incoming stuff from other nodes
    for node in rgraph.keys():
        if node == root:
            continue
        minimizeNode(rgraph, node)

def compute_rdst_candidate(rgraph: dict, root: int) -> dict:
    """
    Computes rdst_candidate by selecting minimum incoming edge from all edges that exist.
    Input: 
    rgraph: Reversed weighted digraph representation
    root: integer denoting which node is the root of the representation

    Output:
    Reversed weighted digraph containing only the shortest incoming edge
    """
    res = {}
    for node in rgraph.items():
        res[node[0]] = {}
    for income in rgraph.items():
        if income[0] == root:
            continue
        minEdge = min(income[1].values())
        #Ok, we've found the minimum edge, now search for node associated with that
        for searchMin in income[1].items():
            if(searchMin[1] == minEdge):
                res[income[0]][searchMin[0]] = searchMin[1]
                break
    #rConvert(res, index)
    return res
 
def parentedBFS(rdst_candidate:dict, root) -> tuple:
    """
    Searches through the nodes of the cycle via BFS rooted from the input root. Checks if a neighbor is a visited
    by property of the function, it's guaranteed to be a cycle by property of 1 or 0 property.
    Input: 
    rdst_candidate - reversed digraph representation
    root - Node we are trying to find a cycle from

    Output:
    Tuple representating cycle 
    """
    #THis method just searches for path that leads to an already visited node
    #If that occurs, save it as a cycle, by retracing parents until you find the node you were trying to visit
    parents = {}
    visited = []
    for item in rdst_candidate.items():
        parents[item[0]] = None
    queue = deque()
    queue.append(root)
    res = []
    while len(queue) != 0:
        current = queue.pop()
        visited.append(current)
        for neighbor in rdst_candidate[current].keys():
            if neighbor in visited:
                search = current
                while search != neighbor:
                    res.append(search)
                    search = parents[search]
                res.append(neighbor)
                res.reverse()
                return tuple(res)
            #Otherwise, just standard BFS
            parents[neighbor] = current
            queue.append(neighbor)
    #This case is the one where there is no cycle, so please don't edit this (empty tuple)
    return tuple(res)

def compute_cycle(rdst_candidate: dict) -> tuple:
    """
    This function computes a cycle if one exists, and returns the members of that cycle in order as a tuple

    Inputs:
    rdst_candidate: Reversed digraph representation

    Outputs:
    A tuple representing the nodes in the cycle
    """
    copy = rdst_candidate
    res = []
    for item in copy.items():
        #For every node (some nodes will just terminate promptly, e.g. edge 3-5), search for cycle
        test = parentedBFS(copy, item[0])
        if test != tuple(res):
            return test
    #If we didn't find any cycles, just return empty tuple 
    return tuple(res)
        
def contract_cycle(graph: dict, cycle: tuple) -> Tuple[dict, int]:
    """
    This function contracts the cycle in the graph by condensing it down to a singular node, and then pruning the parallel edges
    Inputs: 
    graph: Standard digraph representation - this is complete graph, not pruned in any manner
    cycle: tuple denoting nodes in a cycle

    Output:
    Contracted graph in standard digraph representation - all nodes in cycle are not included, only new node cstar
    """
    cstar = max(graph.keys()) + 1
    res = {}
    res[cstar] = {}
    for key in graph.keys():
        if key not in cycle:
            res[key] = {}
    #in the graph, I have the members of the cycle - I want to find the minimum incoming, and minimum exiting from cycle
    for source in graph.items():
        for dest in source[1].items():
            if source[0] in cycle and dest[0] not in cycle:
                #Minimize parallel outgoing edges as well
                if dest[0] in res[cstar]:
                    res[cstar][dest[0]] = min(res[cstar][dest[0]], dest[1]) 
                else:
                    res[cstar][dest[0]] = dest[1]
            elif source[0] not in cycle and dest[0] in cycle:
                if cstar in res[source[0]]:
                    #Take the minimum of the parallel edges that we encounter if there are multiple
                    res[source[0]][cstar] = min(res[source[0]][cstar], dest[1])
                else:
                    res[source[0]][cstar] = dest[1]
            elif source[0] not in cycle and dest[0] not in cycle and source[0] != dest[0]:
                res[source[0]][dest[0]] = dest[1] #Include it ?
    #convert(res,index)
    
    return (res, cstar)

def calculateOutwardDestNode(graph, cycle, outNode):
    """
    This calculates the minimum length edge whose destination is a node is outside of the cycle

    Inputs:
    graph: Standard weighted digraph representation
    cycle: Tuple containing elements of the cycle (inorder)
    outNode: The node outside of the cycle that has tail node from within the cycle (head of edge with tail in cstar)

    Outputs:
    The node in the cycle that is the tail of outNode
    """
    outMin = float('inf')
    ans = -1
    if outNode != []: #If we even have an outNode
        for element in cycle:
            for outDest in graph[element].items():
                if outDest[0] == outNode and outDest[1] < outMin:
                    outMin = outDest[1]
                    ans = element
    return ans

def calculateInwardFacingMinNode(graph, cycle, inNode):
    """
    Calculates the destination node within the cycle based on node that was outside of the cycle (no parallel edges)
    Inputs:
    graph: Standard weighted graph - note this one has all the edges
    cycle: tuple representing members of cycle
    inNode: The node OUTSIDE of the cycle who's head is in the cycle

    Outputs: 
    The node within the cycle which is pointed to by inNode (tail node is inNode, head node is a node in cycle)
    """
    ans = -1
    inMin = float('inf')
    for inDest in graph[inNode].items():
        if inDest[0] in cycle and inDest[1] < inMin:
            inMin = inDest[1]
            ans = inDest[0]
    return ans

def expand_graph(graph: dict, rdst_candidate: dict, cycle: tuple, cstar: int) -> dict:
    """
    Expands the graph by re-adding the nodes that were contained within cstar.
    Inputs:
    graph: Standard weighted digraph representation (dictionary of dictionaries)
    rdst_candidate: Reverwed weighted digraph representation
    cycle: Tuple containing the elements in the cycle
    cstar: Integer denoting the node that contains a collapsed cycle

    Outputs:
    Standard weighted digraph representation that 
    """
    res = {}
    #outNode = -1
    outNodes = []
    inNode = -1
    if (cycle == tuple([])):
        return rdst_candidate
    #Initialization of res (just the set of relevant empty dictionaries)
    for element in cycle:
        #Add in all elements of cycle
        res[element] = {}
    for source in rdst_candidate.keys():
        if source != cstar:
            res[source] = {}
    #Traverse through rdst_candidate - do twofold task here - copy over unrelated edges, and search for inNode/outNode of cycle (outNode might not exist)
    for source in rdst_candidate.items():
        for dest in source[1].items():
            if source[0] != cstar and dest[0] != cstar: #
                res[source[0]][dest[0]] = dest[1] 
            elif source[0] == cstar:
                outNodes.append(dest[0]) #We want all edges in rdst_candidate
                #outNode = dest[0] #Save node that the arrow arrives at from within the cycle
            elif dest[0] == cstar:
                inNode = source[0] #Save node that the arrow comes from to arrive at cycle (you can check these conditions pretty simply)
    cycleSource = calculateInwardFacingMinNode(graph, cycle, inNode)
    #There's only one inwardFacingNode, but multiple outward possible connections - not just one. We want to ensure we get all of them.
    for node in outNodes:
        cycleOut = calculateOutwardDestNode(graph, cycle, node)
        if cycleOut != -1:
            res[cycleOut][node] = rdst_candidate[cstar][node]
    #cycleOut = calculateOutwardDestNode(graph, cycle, outNode)
    if cycleSource != -1:
        res[inNode][cycleSource] = rdst_candidate[inNode][cstar]
    #if cycleOut != -1:
    #    res[cycleOut][outNode] = rdst_candidate[cstar][outNode]
    for index in range(len(cycle) - 1):
        if(cycle[index] != cycleSource):
            res[cycle[index+1]][cycle[index]] = graph[cycle[index+1]][cycle[index]]

    if cycle[len(cycle)-1] != cycleSource:
        res[cycle[0]][cycle[len(cycle)-1]] = graph[cycle[0]][cycle[len(cycle)-1]]
    return res

###START OF PROVIDED FUNCTIONS ###
def bfs(graph, startnode):
    """
        Perform a breadth-first search on digraph graph starting at node startnode.
        
        Arguments:
        graph -- directed graph
        startnode - node in graph to start the search from
        
        Returns:
        The distances from startnode to each node
    """
    dist = {}
    
    # Initialize distances
    for node in graph:
        dist[node] = float('inf')
    dist[startnode] = 0
    
    # Initialize search queue
    queue = deque([startnode])
    
    # Loop until all connected nodes have been explored
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            if dist[nbr] == float('inf'):
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist

def compute_rdmst(graph, root):
    """
        This function checks if:
        (1) root is a node in digraph graph, and
        (2) every node, other than root, is reachable from root
        If both conditions are satisfied, it calls compute_rdmst_helper
        on (graph, root).
        
        Since compute_rdmst_helper modifies the edge weights as it computes,
        this function reassigns the original weights to the RDMST.
        
        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node id.
        
        Returns:
        An RDMST of graph rooted at r and its weight, if one exists;
        otherwise, nothing.
    """
    
    if root not in graph:
        print ("The root node does not exist")
        return
    
    distances = bfs(graph, root)
    for node in graph:
        if distances[node] == float('inf'):
            print ("The root does not reach every other node in the graph")
            return

    rdmst = compute_rdmst_helper(graph, root)
    
    # reassign the original edge weights to the RDMST and computes the total
    # weight of the RDMST
    rdmst_weight = 0
    for node in rdmst:
        for nbr in rdmst[node]:
            rdmst[node][nbr] = graph[node][nbr]
            rdmst_weight += rdmst[node][nbr]

    return (rdmst,rdmst_weight)

def compute_rdmst_helper(graph,root):
    """
        Computes the RDMST of a weighted digraph rooted at node root.
        It is assumed that:
        (1) root is a node in graph, and
        (2) every other node in graph is reachable from root.
        
        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node in graph.
        
        Returns:
        An RDMST of graph rooted at root. The weights of the RDMST
        do not have to be the original weights.
        """
    
    # reverse the representation of graph
    rgraph = reverse_digraph_representation(graph)
    
    # Step 1 of the algorithm
    modify_edge_weights(rgraph, root)
    
    # Step 2 of the algorithm
    rdst_candidate = compute_rdst_candidate(rgraph, root)
    #convert(rdst_candidate, index)
    #print(rdst_candidate)
    # compute a cycle in rdst_candidate
    cycle = compute_cycle(rdst_candidate)
    
    # Step 3 of the algorithm
    if not cycle:
        return reverse_digraph_representation(rdst_candidate)
    else:
        # Step 4 of the algorithm
        
        g_copy = deepcopy(rgraph)
        g_copy = reverse_digraph_representation(g_copy)
        
        # Step 4(a) of the algorithm
        (contracted_g, cstar) = contract_cycle(g_copy, cycle)
        #cstar = max(contracted_g.keys())
        
        # Step 4(b) of the algorithm
        new_rdst_candidate = compute_rdmst_helper(contracted_g, root)
        
        # Step 4(c) of the algorithm
        rdmst = expand_graph(reverse_digraph_representation(rgraph), new_rdst_candidate, cycle, cstar)
        #convert(rdmst, index)
        
        return rdmst


### Functions for use in Problem 4

def infer_transmap(gen_data, epi_data, patient_id):
    """
        Infers a transmission map based on genetic
        and epidemiological data rooted at patient_id
        
        Arguments:
        gen_data -- filename with genetic data for each patient
        epi_data -- filename with epidemiological data for each patient
        patient_id -- the id of the 'patient 0'
        
        Returns:
        The most likely transmission map for the given scenario as the RDMST 
        of a weighted, directed, complete digraph
        """
    
    complete_digraph = construct_complete_weighted_digraph(gen_data, epi_data)
    convert(complete_digraph, index)
    return compute_rdmst(complete_digraph, patient_id)

def read_patient_sequences(filename):
    """
        Turns the bacterial DNA sequences (obtained from patients) into a list containing tuples of
        (patient ID, sequence).
        
        Arguments:
        filename -- the input file containing the sequences
        
        Returns:
        A list of (patient ID, sequence) tuples.
        """
    sequences = []
    with open(filename) as f:
        line_num = 0
        for line in f:
            if len(line) > 5:
                patient_num, sequence = line.split("\t")
                sequences.append( (int(patient_num), ''.join(e for e in sequence if e.isalnum())) )
    return sequences

def read_patient_traces(filename):
    """
        Reads the epidemiological data file and computes the pairwise epidemiological distances between patients
        
        Arguments:
        filename -- the input file containing the sequences
        
        Returns:
        A dictionary of dictionaries where dict[i][j] is the
        epidemiological distance between i and j.
    """
    trace_data = []
    patient_ids = []
    first_line = True
    #print(os.getcwd())
    with open(filename) as f:
        for line in f:
            if first_line:
                patient_ids = line.split()
                patient_ids = list(map(int, patient_ids))
                first_line = False
            elif len(line) > 5:
                trace_data.append(line.rstrip('\n'))
    return compute_pairwise_epi_distances(trace_data, patient_ids)

def compute_pairwise_gen_distances(sequences, distance_function):
    """
        Computes the pairwise genetic distances between patients (patients' isolate genomes)
        
        Arguments:
        sequences -- a list of sequences that correspond with patient id's
        distance_function -- the distance function to apply to compute the weight of the 
        edges in the returned graph
        
        Returns:
        A dictionary of dictionaries where gdist[i][j] is the
        genetic distance between i and j.
        """
    gdist = {}
    cultures = {}
    
    # Count the number of differences of each sequence
    for i in range(len(sequences)):
        patient_id = sequences[i][0]
        seq = sequences[i][1]
        if patient_id in cultures:
            cultures[patient_id].append(seq)
        else:
            cultures[patient_id] = [seq]
            gdist[patient_id] = {}
    # Add the minimum sequence score to the graph
    for pat1 in range(1, max(cultures.keys()) + 1):
        for pat2 in range(pat1 + 1, max(cultures.keys()) + 1):
            min_score = float("inf")
            for seq1 in cultures[pat1]:
                for seq2 in cultures[pat2]:
                    score = distance_function(seq1, seq2)
                    if score < min_score:
                        min_score = score
            gdist[pat1][pat2] = min_score
            gdist[pat2][pat1] = min_score
    return gdist

### HELPER FUNCTIONS. ###

def find_first_positives(trace_data):
    """
        Finds the first positive test date of each patient
        in the trace data.
        Arguments:
        trace_data -- a list of data pertaining to location
        and first positive test date
        Returns:
        A dictionary with patient id's as keys and first positive
        test date as values. The date numbering starts from 0 and
        the patient numbering starts from 1.
        """
    first_pos = {}
    for pat in range(len(trace_data[0])):
        first_pos[pat + 1] = None
        for date in range(len(trace_data)):
            if trace_data[date][pat].endswith(".5"):
                first_pos[pat + 1] = date
                break
    return first_pos

def compute_epi_distance(pid1, pid2, trace_data, first_pos1, first_pos2, patient_ids):
    """
        Computes the epidemiological distance between two patients.
        
        Arguments:
        pid1 -- the assumed donor's index in trace data
        pid2 -- the assumed recipient's index in trace data
        trace_data -- data for days of overlap and first positive cultures
        first_pos1 -- the first positive test day for pid1
        first_pos2 -- the first positive test day for pid2
        patient_ids -- an ordered list of the patient IDs given in the text file
        
        Returns:
        Finds the epidemiological distance from patient 1 to
        patient 2.
        """
    first_overlap = -1
    assumed_trans_date = -1
    pid1 = patient_ids.index(pid1)
    pid2 = patient_ids.index(pid2)
    # Find the first overlap of the two patients
    for day in range(len(trace_data)):
        if (trace_data[day][pid1] == trace_data[day][pid2]) & \
            (trace_data[day][pid1] != "0"):
            first_overlap = day
            break
    if (first_pos2 < first_overlap) | (first_overlap < 0):
        return len(trace_data) * 2 + 1
    # Find the assumed transmission date from patient 1 to patient 2
    for day in range(first_pos2, -1, -1):
        if (trace_data[day][pid1] == trace_data[day][pid2]) & \
            (trace_data[day][pid1] != "0"):
            assumed_trans_date = day
            break
    sc_recip = first_pos2 - assumed_trans_date

    if first_pos1 < assumed_trans_date:
        sc_donor = 0
    else:
        sc_donor = first_pos1 - assumed_trans_date
    return sc_donor + sc_recip

def compute_pairwise_epi_distances(trace_data, patient_ids):
    """
        Turns the patient trace data into a dictionary of pairwise 
        epidemiological distances.
        
        Arguments:
        trace_data -- a list of strings with patient trace data
        patient_ids -- ordered list of patient IDs to expect
        
        Returns:
        A dictionary of dictionaries where edist[i][j] is the
        epidemiological distance between i and j.
        """
    edist = {}
    proc_data = []
    # Reformat the trace data
    for i in range(len(trace_data)):
        temp = trace_data[i].split()[::-1]
        proc_data.append(temp)
    # Find first positive test days and remove the indication from the data
    first_pos = find_first_positives(proc_data)
    for pid in first_pos:
        day = first_pos[pid]
        proc_data[day][pid - 1] = proc_data[day][pid - 1].replace(".5", "")
    # Find the epidemiological distance between the two patients and add it
    # to the graph
    for pid1 in patient_ids:
        edist[pid1] = {}
        for pid2 in patient_ids:
            if pid1 != pid2:
                epi_dist = compute_epi_distance(pid1, pid2, proc_data,
                                                first_pos[pid1], first_pos[pid2], patient_ids)
                edist[pid1][pid2] = epi_dist
    return edist

#Does everything

def compute_genetic_distance(seq1, seq2):
    res = abs(len(seq1) - len(seq2))
    if len(seq1) != len(seq2):
        if len(seq2) == min(len(seq1),len(seq2)):
            temp = seq1
            seq1 = seq2
            seq2 = temp
            #We want minimum length function to be seq1
    
    for index in range(len(seq1)):
        if seq1[index] != seq2[index]:
            res += 1
    return res

def construct_complete_weighted_digraph(gen_data, epi_data):
    #gen_data will be a filename with genetic data for each patient, and so will epi_data
    #We just need to build the digraph
    res = {}
    epidemiological_dicts = read_patient_traces(epi_data)
    patient_tuples = read_patient_sequences(gen_data)
    for patient in patient_tuples:
        #ids.append(patient[0])
        res[patient[0]] = {} # Add an empty dictionary
    genetic_data = compute_pairwise_gen_distances(patient_tuples, compute_genetic_distance)
    #List comprehension: Create a list based on the following things - like e.g. [blah for blah in other list + 2]]
    max_E = max([max(dictionary.values()) for dictionary in epidemiological_dicts.values()])
    for element in genetic_data.items():
        for neighbor in element[1].items():
            res[element[0]][neighbor[0]] = neighbor[1] + (999 * (epidemiological_dicts[element[0]][neighbor[0]]/max_E) /100000)
    #print(res)
    return res

def convert(graph, index):
    """
    Inputs:
    graph: Standard weighted digraph representation
    index: Array with element 0

    Outputs:
    A figure containing nodes/edges shown
    """
    plt.figure(index[0])
    G = nx.Graph()
    elarge = []
    for first in graph.items():
        for element in  first[1].items():
            G.add_edge(first[0], element[0], weight=element[1])
            elarge.append(tuple([first[0], element[0]]))

    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, arrows = True, arrowstyle="->", arrowsize=20, width=6, alpha = 0.5)
    #nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    index[0] += 1
    #print(index)
    #plt.show()

def rConvert(graph, index):
    """
    Inputs:
    graph: Standard weighted digraph representation
    index: Array with element 0

    Outputs:
    A figure containing nodes/edges shown
    """
    plt.figure(index[0])
    G = nx.Graph()
    #temp = []
    #res = []
    elarge = []
    for first in graph.items():
        for element in  first[1].items():
            G.add_edge(first[0], element[0], weight=element[1])
            elarge.append(tuple([first[0], element[0]]))
    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, arrows = True, arrowstyle="<-", arrowsize=20, width=6, alpha = 0.5)
    #nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    index[0] += 1

def dijkstra(graph, root):
    distances = {}
    for node in graph.keys():
        distances[node] = -1
    queue = deque()
    queue.append(root)
    distances[root] = 0
    while len(queue) != 0:
        curr = queue.pop()
        for neighbor in graph[curr]:
            queue.append(neighbor) #Technically speaking we don't need a visited array or anything.
            if distances[neighbor] == -1 or distances[curr] + graph[curr][neighbor] < distances[neighbor]:
                distances[neighbor] = distances[curr] + graph[curr][neighbor]
    reverse = {}
    for element in distances.items():
        reverse[element[1]] = element[0]
    
    print(sorted(reverse.items()))
    

def main():
    #NON scuffed standard testing:
    g0 = {0: {1: 2, 2: 2, 3: 2}, 1: {2: 2, 5: 2}, 2: {3: 2, 4: 2}, 3: {4: 2, 5: 2}, 4: {1: 2}, 5: {}}
    # Results for compute_rdmst(g0, 0):
    # ({0: {1: 2, 2: 2, 3: 2}, 1: {5: 2}, 2: {4: 2}, 3: {}, 4: {}, 5: {}}, 10) 

    g1 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}
    # Results for compute_rdmst(g1, 0):
    # ({0: {2: 4}, 1: {}, 2: {3: 8}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}, 28)

    g2 = {0: {1: 5, 2: 4}, 1: {2: 2}, 2: {1: 2}}
    # Results for compute_rdmst(g2, 0):
    # ({0: {2: 4}, 1: {}, 2: {1: 2}}, 6)

    g3 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0}, 3: {1: 1.0, 2: 1.0, 4: 16.0, 5: 0.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0}}
    # Results for compute_rdmst(g3, 1):
    # ({1: {3: 1.0, 4: 9.1}, 2: {}, 3: {2: 1.0, 5: 0.0}, 4: {}, 5: {}}, 11.1)

    g4 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1, 6: 10.1, 7: 10.1,  8: 6.1, 9: 11.0, 10: 10.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0, 6: 18.1, 7: 18.1, 8: 14.1, 9: 19.1, 10: 18.0}, 3: {1: 1.0, 2: 1.0, 4: 16.0, 5: 0.0, 6: 17.0, 7: 17.0, 8: 13.1, 9: 18.1, 10: 17.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0, 6: 5.1, 7: 5.1, 8: 15.1, 9: 6.1, 10: 5.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0, 6: 17.1, 7: 17.1, 8: 13.1, 9: 18.1, 10: 17.0}, 6: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 7: 0.0, 8: 16.1, 9: 7.1, 10: 0.0}, 7: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 6: 0.0, 8: 16.0, 9: 7.1, 10: 0.0}, 8: {1: 6.1, 2: 14.1, 3: 13.1, 4: 15.1, 5: 13.1, 6: 16.1, 7: 16.0, 9: 17.1, 10: 16.1}, 9: {1: 11.1, 2: 19.1, 3: 18.1, 4: 6.1, 5: 18.1, 6: 7.1, 7: 7.1, 8: 17.1, 10: 7.0}, 10: {1: 10.1, 2: 18.1, 3: 17.1, 4: 5.1, 5: 17.0, 6: 0.0, 7: 0.0, 8: 16.1, 9: 7.0}}
    # Results for compute_rdmst(g4, 1):
    #g4Ans = ({1: {8: 6.1, 3: 1.0, 4: 9.1}, 2: {}, 3: {2: 1.0, 5: 0.0}, 4: {9: 6.1, 10: 5.0}, 5: {}, 6: {7: 0.0}, 7: {}, 8: {}, 9: {}, 10: {6: 0.0}}, 28.3)
    
    g5 = {1: {2: 3}, 2:{3:5}, 3:{5:1, 4:1}, 4:{2:2}, 5:{}} #10

    g6 = {0:{}, 1:{2:10}, 2:{3:10}, 3:{1:10}}
    print(compute_rdmst(g0, 0))
    print(compute_rdmst(g1, 0))
    print(compute_rdmst(g2, 0))
    print(compute_rdmst(g3, 1))
    print(compute_rdmst(g4, 1))
    print(compute_rdmst(g5, 1))
    convert(compute_rdmst(g2, 0)[0], index)
    convert(g2, index)
    #dirname = os.path.dirname(__file__)
    #filename = os.path.join(dirname, 'patient_traces.txt')
    ans = infer_transmap("patient_sequences.txt", "patient_traces.txt", 1)
    convert(ans[0], index)
    print(ans)
    dijkstra(ans[0], 1)
    plt.show()
    return 0

main()