# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:43:53 2020

@author: Colin
"""

from tqdm import trange, tqdm
from joblib import Parallel, delayed
from igraph import *
import multiprocessing
import numpy as np
import cvxpy as cp


def generateERGraph(num_vertices, prob_edges, min_weight, max_weight,
                  symbols=None, max_symbols=1):
    # make random graph
    g = Graph.Erdos_Renyi(num_vertices, prob_edges, directed=True, loops=False)
    return _fix_graph(g, min_weight, max_weight, symbols, max_symbols)


def generateGraph(graph_generator, min_weight, max_weight, symbols, max_symbols, **kwargs):
    g = graph_generator(**kwargs)
    return _fix_graph(g, min_weight, max_weight, symbols, max_symbols)


def _fix_graph(g, min_weight, max_weight, symbols=None, max_symbols=1):
    # remove terminal vertices from graph
    g.vs["od"] = g.vs.degree(mode=OUT, loops=False)
    for vertex in g.vs.select(od_lt=2):
        vertex["current"] = True
        # connect a random vertex to this one and this one to another one
        from_vert = np.random.choice(g.vs.select(current_ne=True))
        g.add_edge(from_vert, vertex)
        to_vert = np.random.choice(g.vs.select(current_ne=True))
        g.add_edge(vertex, to_vert)
        vertex["current"] = False
        
    # min and max weight values
    w_min = min_weight
    w_max = max_weight
    
    # find the number of edges, and assign weights
    g.ecount()
    # change to np.random.random for non-integer weights
    weights = np.random.randint(w_min, w_max, g.ecount())
    # weights = (np.random.random(g.ecount()) - 0.5) * 20000.0
    g.es["weight"] = weights
    g.es["label"] = weights

    # choose random labels to put on each state
    if symbols:
        if max_symbols > 1:
            for v in g.vs:
                # determine the number of labels to put on each state
                num_symbols = np.random.randint(1, max_symbols+1)
                # sample those labels, and turn them into a string
                label = ', '.join(np.random.choice(symbols, num_symbols, replace=False))
                v["label"] = label
        else:
            g.vs["label"] = np.random.choice(symbols, g.vcount())

    state_names = [str(v.index) for v in g.vs]
    g.vs["name"] = state_names

    # this is inefficient right now, but I might want to add random action assignments in the future, so I'm leaving it.
    actions = {}
    for i, edge in enumerate(g.es):
        actions[i] = [edge.index]
    for action in actions:
        for edge in actions[action]:
            edge = g.es[edge]
            edge["action"] = action

    return g



def generateHistories(graph, num_histories, history_len, discount_factor):
    # generate some histories
    weights = graph.es["weight"]
    num_cores = multiprocessing.cpu_count()
    histories = Parallel(n_jobs=num_cores,verbose=5)(delayed(generateHistory)(graph, weights, history_len, discount_factor) for i in trange(num_histories))

    return histories
    
    
def generateHistory(graph, weights, history_len, discount_factor):
    # perform random walk on given graph, starting at node "1"
    walk = graph.random_walk(0, history_len)
    # perform random walk on given graph, starting from random node
    # walk = graph.random_walk(np.random.randint(0, graph.vcount()))
    edges = []
    labels = []
    if 'label' in graph.vs.attributes():
        labels.append(graph.vs["label"][0])
    # compute accumulated value of the walk
    value = 0
    for depth in range(len(walk)-1):
        from_vertex = walk[depth]
        to_vertex = walk[depth + 1]
        edge_tuple = ([from_vertex], [to_vertex])
        edge = graph.es.find(_between=edge_tuple)
        factor = discount_factor ** depth
        edge_weight = weights[edge.index]
        value += factor * edge_weight
        edges.append(edge.index)
        if 'label' in graph.vs.attributes():
            labels.append(graph.vs["label"][to_vertex])

    return [edges, value, labels]
    
    
def solveWeights(graph, histories, discount_factor, verbose=False):
    # make list of edges, whose weights will be solved for
    weights = cp.Variable(graph.ecount())
    
    # create value equations for each history
    values = np.zeros((len(histories), graph.ecount()))
    for h, history in tqdm(enumerate(histories)):
        word = history[0]
        for depth, edge in enumerate(word):
            factor = discount_factor ** depth
            values[h, edge] += factor

    sums = np.array(histories)[:, 1]

    val_w = values @ weights
    cost = cp.sum_squares(val_w - sums)
    
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    norm = cp.norm(weights.value - graph.es["weight"], p=2).value
    if verbose:
        # Print result.
        print("\nThe optimal value is", prob.value)
        print("The optimal x is")
        print(weights.value)
        print("The norm of the residual is ", norm)
        print(prob.status)

    return {"optimal": prob.value, "weights": weights.value, "values": values,
            "norm": norm, "status": prob.status}


if __name__ == "__main__":
    # g = generateERGraph(5, 0.5, -5.0, 5.0, ['a', 'b', 'c'])
    generator = Graph.Barabasi
    gen_kwargs = {'n': 5, 'm': 2, 'directed': True}
    g = generateGraph(generator, -5.0, 5.0, ['a', 'b', 'c'], 1, **gen_kwargs)
    h = generateHistories(g, 5, 10, 0.5)
    print(h)
    solveWeights(g, h, 0.5)
