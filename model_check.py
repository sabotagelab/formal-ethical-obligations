"""
Created June 2020

@author: Colin
"""

import pickle
import re
import subprocess
import numpy as np
import mdptoolbox.mdp as mdp
from copy import copy, deepcopy
from itertools import product
from collections import defaultdict

import stormpy
from tqdm import tqdm
from igraph import *
from random_weighted_automaton import *


# TODO: refactor checkObligation, checkConditional, and generateFragments as Automaton class functions
# ^ the Obligation class could have validatedBy(Automaton)
# ^ also add a get_optimal_actions function to Automaton that returns the optimal actions
# TODO: asserts on the construction logic so malformed automata aren't constructed
# TODO: constructor that takes a list of "action classes" and associated probabilities
# ^ maybe even a gridworld constructor
# TODO: separate function of 'label', 'name' and atomic propositions
# ^ allow multiple propositions on a given vertex, and be able to check them.
# TODO: refactor lots of the functions just to simplify them
class Automaton(object):
    def __init__(self, graph, initial_state, atomic_propositions=None):
        """
        Create a new Automaton object from an igraph graph object and an initial state.
        The graph should have vertex properties "label" and "name".
        The graph should have the edge properties "action" and "weight.
        The graph may optionally have the edge property "prob".
        The initial_state parameter should the id of a vertex in graph.
        Adds a "delete" property to the edges; if edge["delete"] = 1, then it is marked for deletion later.

        :param graph:
        :param initial_state:
        """
        self.graph = graph
        self.graph.es["delete"] = [0] * self.graph.ecount()
        self.q0 = initial_state

        self.qn = self.q0
        self.t = 0
        self.q_previous = []
        self.t_previous = []
        self.num_clones = 0
        self.counter = False
        self.policy = None
        self.has_policy = False
        self.propositions = atomic_propositions
        self.prob = "prob" in self.graph.es.attribute_names()

    @classmethod
    def from_matrix(cls, adj_matrix, atomic_propositions, label_asg, weight_asg=None, initial_state=0):
        """
        Create an automaton from an adjacency matrix and the labels assigned to each state

        :param adj_matrix:
        :param atomic_propositions: a list of strings, e.g. ['a', 'b', 'p', 'q']
        :param label_asg: a list of label assignments in order of vertices, e.g. ["a, b", "b, q", "p, q"]
        :param weight_asg: a list of weight assignments in order of edges, e.g. [1, 0.3, -2.5, 6, 8, 10]
        :param initial_state: a vertex number to be the starting state of the automaton
        :return:
        """
        graph = Graph.Adjacency(adj_matrix)
        if not weight_asg:
            weight_asg = np.zeros(graph.ecount())
        graph.es["weight"] = weight_asg
        graph.es["label"] = weight_asg

        state_names = [str(v.index) for v in graph.vs]
        graph.vs["name"] = state_names
        graph.vs["label"] = label_asg
        return cls(graph, initial_state, atomic_propositions)

    @classmethod
    def with_actions(cls, graph, actions, q0=0, probs=None, labels=False):
        """
        graph is a directed igraph graph object
        actions is a dictionary that maps actions to edges
        key(action), values(list of edges)
        e.g. {0:[0, 1, 3], 1:[2, 4], 2:[5], ... }
        probs is a list of probabilities such that probs[k] is the
        probability of following edge[k] when the action containing
        edge[k] is taken.

        :param graph:
        :param actions:
        :param q0:
        :param probs:
        :param labels:
        """

        state_names = [str(v.index) for v in graph.vs]
        graph.vs["name"] = state_names
        if not labels:
            graph.vs["label"] = state_names
        for action in actions:
            for edge in actions[action]:
                edge = graph.es[edge]
                edge["action"] = action

        if probs:
            graph.es["prob"] = probs
        return cls(graph, q0)

    # TODO: make a gridworld cell class?
    @classmethod
    def as_gridworld(cls, x_len, y_len, start=(0, 0), action_successes=0.8, cells=None, default_reward=-1,
                     state_property='label'):
        """
        Construct an automaton for an x_len-by-y_len gridworld, starting from the 'start' position, with actions up,
        down, left, and right.
        Each action, when taken, has a 'action_success' chance of effecting. Otherwise, another action is effected with
        probability (1-action_success)/3. That is, by default, the 'up' action has a probability of 0.7 to transition
        the automaton from (0,0) to (0,1). Taking the 'up' action has a probability of 0.1 to move the automaton down.
        Because (0,0) is in a corner, however, moving down leaves the automaton in its same state.

        The cells parameter is a list of tuples [(type, positions, reward, absorbing, accessible), ...].
        Each tuple represents one class of cells in the gridworld, relays the positions of those cells, the reward
        received for entering those cells, whether those cells can be exited, and whether those cells can
        be entered.
        The 'type' entry in the tuple is a string that denotes the class of cell; e.g. "goal", "pit", or "wall".
        The 'positions' entry is a list of 2-tuples (x, y) that denotes the locations of the cells of the given type in
        this gridworld. E.g. [(0,0), (2,2), (1,3)].
        The 'reward' entry is a real value that denotes the reward for entering a cell of the given type; e.g. 10.7.
        The 'absorbing' entry is a boolean value that denotes if cells of the given type are absorbing states.
        The 'accessible' entry is a boolean value that denotes if cells of the given type can be entered.
        If cells is left as None, then no cells are specified, and all cells are accessible, non-absorbing, and have a
        reward of 'default_reward'.
        An example cells input for a basic 4x3 gridworld is as follows:
        cells=[("goal", [(3, 2)], 10, True, True),
               ("pit", [(2, 2)], -50, True, True),
               ("wall", [(1, 1)], -1, False, False)]
        This places a goal in the upper-right of the grid with reward 10, and is an absorbing state,
        a pit just below the goal with a reward of -50, and is an absorbing state,
        and a wall just north-east of the starting position that is inaccessible. Note that because the wall is
        not accessible, its 'reward' and 'absorbing' entries are irrelevant.

        If a cell is not included among the positions in the 'cells' parameter, its type is "default", it is accessible,
        and not absorbing, and the reward for entering it is 'default_reward'.

        :param x_len:
        :param y_len:
        :param start:
        :param action_success:
        :param default_reward:
        :param cells:
        """
        n = x_len * y_len
        g_new = Graph(directed=True)
        pos_to_type = {}
        type_to_spec = {}
        default_spec = ("default", [], default_reward, False, True)
        # cache information about cell positions and types for future use
        for spec in cells:
            cell_type = spec[0]
            type_to_spec[cell_type] = spec
            cell_poss = spec[1]
            for cell_pos in cell_poss:
                pos_to_type[cell_pos] = cell_type

        positions = product(range(x_len), range(y_len))
        # set up the attribute dictionary so all vertices can be added in one go
        v_attr_dict = {"x": [], "y": [], "pos": [], "label": [], "name": [], "type": [],
                       "absorbing": [], "accessible": [], "reward": [], "property": []}
        k = 0
        for y in range(y_len):
            for x in range(x_len):
                v_attr_dict["x"].append(x)
                v_attr_dict["y"].append(y)
                v_attr_dict["pos"].append((x, y))
                v_attr_dict["label"].append(str((x, y)))
                v_attr_dict["name"].append(str(k))
                cell_type = pos_to_type.get((x, y), "default")
                cell_spec = type_to_spec.get(cell_type, default_spec)
                v_attr_dict["type"].append(cell_type)
                v_attr_dict["reward"].append(cell_spec[2])
                v_attr_dict["absorbing"].append(cell_spec[3])
                v_attr_dict["accessible"].append(cell_spec[4])
                k += 1
        # add a vertex for every position, and set its attributes
        v_attr_dict["property"] = v_attr_dict[state_property]
        g_new.add_vertices(n, attributes=v_attr_dict)

        # set the four actions and the effect of actually following that action
        actions = ["up", "right", "down", "left"]
        effects = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # define the probability that an effect is effected in the case that the effect is not the assumed consequence
        # of a taken action
        if not isinstance(action_successes, list):
            off_prob = (1.0 - action_successes) / 3.0
        else:
            off_prob = False
        # set up attribute dictionary and edge list so all edges can be added in one go
        edge_tuples = []
        signatures = []
        e_attr_dict = {"action": [], "weight": [], "prob": []}
        # for each vertex...
        for v in g_new.vs:
            effect_targets = []
            # if that vertex is inaccessible
            if not v["accessible"]:
                # just add a self edge for consistency reasons
                effect_targets = [v] * len(effects)
            elif v["absorbing"]:
                # if it's absorbing, set the target of each effect to itself
                effect_targets = [v] * len(effects)
            else:
                # otherwise, for each effect...
                for effect in effects:
                    # get the x and y positions that the automaton would enter under that effect
                    next_x = v["x"] + effect[0]
                    next_y = v["y"] + effect[1]
                    # if those x and y positions are inside the bounds...
                    if 0 <= next_x < x_len and 0 <= next_y < y_len:
                        # get the target of the effect...
                        next_v = g_new.vs.find(pos=(next_x, next_y))
                        # if the target isn't accessible...
                        if not next_v["accessible"]:
                            # then the effect leads to the same state
                            next_v = v
                    else:
                        # the target of the effect is out of bounds, so the effect would lead to the same state
                        next_v = v
                    # record the target of each effect
                    effect_targets.append(next_v)

            # for each action that can be taken...
            for i, action in enumerate(actions):
                # for each effect that action can have...
                for j, effect in enumerate(effects):
                    # get the target of this effect
                    next_v = effect_targets[j]
                    signature = (v.index, next_v.index, action)
                    # if probabilities are classic gridworld...
                    if off_prob:
                        # in the case that the effect matches the action taken...
                        if i == j:
                            prob = action_successes
                        else:
                            # otherwise, the probability of this effect is the off_prob probability
                            prob = off_prob
                    else:
                        # otherwise action_successes defines probability of an action moving fore, starboard, aft,
                        # or port
                        prob = action_successes[(4-i+j) % 4]

                    if signature not in signatures:
                        signatures.append(signature)
                        # record an edge from v to next_v
                        edge_tuples.append((v.index, next_v.index))
                        # record the attributes of this edge
                        e_attr_dict["action"].append(action)
                        e_attr_dict["weight"].append(next_v["reward"])
                        e_attr_dict["prob"].append(prob)
                    else:
                        sig_index = signatures.index(signature)
                        e_attr_dict["prob"][sig_index] += prob
        # add an edge for each (vertex*action*effect), with associated attributes
        g_new.add_edges(edge_tuples, e_attr_dict)
        v_0 = g_new.vs.find(pos=start)
        return cls(g_new, v_0.index)

    def setPolicy(self, policy):
        self.policy = policy
        self.has_policy = True


    def k(self, i):
        """
        get all actions available from vertex i.

        :param i:
        :return:
        """
        es = self.graph.es.select(_source=i)
        actions = []
        for edge in es:
            action = edge["action"]
            if action not in actions:
                actions.append(action)
        actions.sort()
        return actions

    def setCounter(self, var_name='c', start=0, term=1000):
        """
        create a simple counter in this automaton

        :param var_name:
        :param start:
        :param term:
        :return:
        """
        self.counter = (var_name, start, term)

    def forceKn(self, kn, source=0):
        """
        delete all edges from source vertex with edges in kn that are not
        themselves in kn.

        :param kn:
        :param source:
        :return:
        """
        # find all edges with this source
        candidates = self.graph.es.select(_source=source)
        # find all candidates not in kn
        selections = candidates.select(action_ne=kn)
        # remove these edges from the graph
        self.graph.delete_edges(selections)
        return self

    def forceEn(self, en, source=0):
        """
        delete all edges from source vertex that are not
        themselves in en.

        :param en:
        :param source:
        :return:
        """
        # find all edges with this source.
        # when an index is deleted, the indices change
        # so I need to delete all edges at once
        # I can't query the edge indices here, so I'll give the edges temporary names
        # and grab the ones with names different from en
        candidates = self.graph.es.select(_source=source)
        for edge in candidates:
            if edge.index != en.index:
                edge["delete"] = 1
            else:
                edge["delete"] = 0
        candidates = candidates.select(delete=1)
        self.graph.delete_edges(candidates)
        return self

    def forceQn(self, qn, source=0):
        """
        delete all edges from source vertex with edges that do not lead to given
        vertex.

        :param qn:
        :param source:
        :return:
        """
        # find all edges with this source
        candidates = self.graph.es.select(_source=source)
        # find all candidates not in qn
        selections = candidates.select(_target_ne=qn)
        # remove these edges from the graph
        self.graph.delete_edges(selections)
        return self

    def max_action_outcomes(self):
        """
        What is the largest number of states any action in the automaton could lead to?
        :return:
        """
        actions = list(set(self.graph.es['action']))
        max_deg = 0
        for action in actions:
            for v in self.graph.vs:
                action_edges = self.graph.es.select(_source=v, action=action)
                outdegree = len(action_edges)
                max_deg = max(max_deg, outdegree)
        return max_deg

    def union(self, g, target=0):
        """
        modify this automaton such that transitions in itself to the target
        state are replaced with transitions to automaton g.

        :param g:
        :param target:
        :return:
        """
        # recall certain properties of the given graphs
        v_mod = self.graph.vcount() + target % g.graph.vcount()

        # find the transitions to the target state not from previous state
        if len(self.q_previous) > 0:
            # es = self.graph.es.select(_target=target, _source_notin=[self.q_previous[-1]])
            es = self.graph.es.select(_target=target)
        else:
            es = None
        # if no
        if not es:
            return self
        else:
            self.num_clones += 1

        labels = self.graph.vs["label"] + [label + "-" + str(self.num_clones)
                                           for label in g.graph.vs["label"]]
        names = self.graph.vs["name"] + g.graph.vs["name"]
        weights = self.graph.es["weight"] + g.graph.es["weight"]
        actions = self.graph.es["action"] + g.graph.es["action"]
        if self.prob:
            probs = self.graph.es["prob"] + g.graph.es["prob"]
        else:
            probs = None
        # take the disjoint union of this graph and the given graph
        self.graph = self.graph.disjoint_union(g.graph)
        # reinstate edge and vertex attributes
        self.graph.vs["label"] = labels
        self.graph.vs["name"] = names
        self.graph.es["weight"] = weights
        self.graph.es["action"] = actions
        if probs:
            self.graph.es["prob"] = probs
        # properties = [(e.source, e["action"], e["weight"]) for e in es]
        # for each edge, make a replacement edge to new graph
        new_edges = []
        new_edge_attr = {"action":[], "weight":[]}
        if probs:
            new_edge_attr["prob"] = []
        for edge in tqdm(es):
            new_edges.append((edge.source, self.graph.vs[v_mod]))
            new_edge_attr["action"].append(edge["action"])
            new_edge_attr["weight"].append(edge["weight"])
            if probs:
                new_edge_attr["prob"].append(edge["prob"])
        self.graph.add_edges(new_edges, new_edge_attr)
        # delete the edges
        if len(self.q_previous) > 0:
            # self.graph.delete_edges(_target=target, _source_notin=[self.q_previous[-1]])
            self.graph.delete_edges(_target=target)
            # self.graph.delete_vertices(VertexSeq(self.graph, target))
        else:
            self.graph.delete_edges(_target=target)

        return self

    def optimal(self, discount, best=True, punish=-1000, steps=100):
        mod = 1
        if not best:
            mod = -1
        tr = self.to_mdp(best, punish)
        sol = mdp.ValueIteration(tr[0], tr[1], discount)
        sol.run()
        return sol.V[self.q0] * mod

    def to_mdp(self, best=True, punish=-1000):
        """
        solve graph as MDP for most (or least) optimal strategy and return value

        :param best:
        :param punish:
        :return:
        """
        vcount = self.graph.vcount()
        ecount = self.graph.ecount()
        # t represents the transition probabilities for each "action" from one
        # "state" to another "state, where every action is associated with a
        # transition, and every state is represented by a vertex.
        # The matrix may be considered, along the "action" vertex, as specifying
        # the probability that action has of moving the process from state A
        # to state B. As we are treating each transition as a sure thing,
        # in the case that we are evaluating a DAU automaton, all
        # probabilities are 1. E.g. if edge 2 in the graph points from vertex 3
        # to vertex 1, then the entry t[2, 3, 1] = 1. The t matrix must be row
        # stochastic, so there must be an entry of 1 in each row; i.e. for each
        # source state, there must be some effect on the process if the process
        # takes that edge - even if the edge is not connected to the source.
        # In the given example, t[2, 3, j] = 0 where j != 1, since it is clear
        # that taking edge 2 from 3 to any edge other than 1 is impossible,
        # however t[2, j, j] = 1 where j != 3, since there *must* be an action
        # for each edge-state pair. Due to this discrepancy in representations,
        # the only reasonable choice is to say that trying to take edges that do
        # not begin from the current vertex leaves you in the current vertex.
        # Letting the process "wait" by "taking" an edge not connected to the
        # current state can be problematic when there are negative weights.
        # If there are negative weights, the MDP seems to want to wait unless
        # it gets punished for doing so, since 0 reward is better than -x.
        # To prevent this behavior, rewards for taking actions that correspond
        # with moving on edges not connected to the current vertex are deeply
        # negative. If the range of rewards for the automaton are lower than
        # this value, it will have to be changed, so it's a parameter with
        # default of -1000.
        # The r matrix, itself, has the same shape as the t matrix, and each
        # entry in r provides the reward for taking the transition in t that has
        # the same index. E.g. if moving from vertex 3 to vertex 1 via edge 2
        # has the reward of 5, then r[2, 3, 1] = 5. For "wait" actions, the
        # reward is equal to the punish value, e.g. r[2, j, j] = -1000 where
        # j != 1. NOTE: Currently, all elements of r are set to a value (either
        # a valid weight, or the punish value). Rewards associated with elements
        # of t where the transition probability is 0 may also be set to 0 if we
        # want to switch to sparse matrix representations.
        if self.prob:
            actions = set(self.graph.es.get_attribute_values("action"))
            t = np.zeros((len(actions), vcount, vcount))
            r = np.full((len(actions), vcount, vcount), punish)
        else:
            t = np.zeros((ecount, vcount, vcount))
            r = np.full((ecount, vcount, vcount), punish)
        # mod negates the weights of the system if we're looking for the worst
        # possible execution (except punishment weights, otherwise the system
        # would do nothing at all).
        mod = 1
        if not best:
            mod = -1

        if self.prob:
            # we're dealing with an actual MDP! Construct the mdp differently
            # start by getting a list of all actions
            # TODO: a more stable way of ordering actions. This gets all the actions, but sets don't maintain order, so
            # if you run this on the same Automaton twice, its matrices will be shuffled along the action axis, making
            # it difficult to keep track of the identities of the actions the Automaton should be taking. None of the
            # model checking code here relies on the actions maintaining their identities (I use the Value output of
            # next states instead of the policy for determining optimal actions), but it would be nice to reliably
            # retrieve the policy from the Automaton, such that the actions in the policy have the same ids as actions
            # in the Automaton.
            actions = set(self.graph.es.get_attribute_values("action"))
            # for each action...
            for i, action in enumerate(actions):
                # find the edges that are in that action
                edges = self.graph.es.select(action_eq=action)
                # for each such edge...
                for j, edge in enumerate(edges):
                    tup = edge.tuple
                    # set the transition probability for action i for this edge to its prob
                    t[i, tup[0], tup[1]] = edge["prob"]
                    r[i, tup[0], tup[1]] = edge["weight"] * mod
                # sometimes an action may not be conditionally possible, so check for rows in t that are all zeros
                # in which case create a self-transition with probability 1 and reward = punish
                for k in range(vcount):
                    if sum(t[i, k]) == 0:
                        t[i, k, k] = 1
                        r[i, k, k] = punish

        else:
            # This loop iterates through the edges in the graph so each transition
            # matrix can be provided for every edge.
            # for each edge...
            for i, edge in enumerate(self.graph.es):
                tup = edge.tuple
                # ... for each vertex considered as source...
                for j in range(vcount):
                    # ... if this vertex actually is the source of this edge...
                    if j == tup[0]:
                        # ... the transition probability from source to target is 1
                        t[i, tup[0], tup[1]] = 1
                    else:
                        # ... otherwise, taking this edge is a "wait" action.
                        t[i, j, j] = 1
                # ... change the reward corresponding to actually taking the edge.
                r[i, tup[0], tup[1]] = edge["weight"] * mod
        return (t, r)

    def checkCTL(self, file, x, verbose=False):
        """
        Checks the automaton for a given CTL specification

        :param file:
        :param x:
        :param verbose:
        :return:
        """
        # convert graph to nuXmv model
        self.convertToNuXmv(file, x)
        # nuxmv = "nuXmv"
        # TODO: extract this and make it easier to change
        # nuxmv = "E:\\Programs\\nuXmv-2.0.0-win64\\bin\\nuXmv.exe"
        # nuxmv = "/home/colin/Downloads/nuXmv-2.0.0-Linux/bin/nuXmv"
        nuxmv = "nuXmv"

        # with open("cmd.txt", 'w') as f:
        #     f.write("read_model -i " + file + "\n")
        #     f.write("flatten_hierarchy\n")
        #     f.write("encode_variables\n")
        #     f.write("build_model\n")
        #     f.write("check_ctlspec -p \"" + x + "\"")

        # out = subprocess.run([nuxmv, "-source", "cmd.txt", file], shell=True, stdout=subprocess.PIPE)
        # out = subprocess.run([nuxmv, file], shell=True, stdout=subprocess.PIPE)
        out = subprocess.run([nuxmv, file], stdout=subprocess.PIPE)
        check = "true" in str(out.stdout)
        if verbose:
            print(out.stdout)
        return check

    def checkLTL(self, file, x, verbose=False):
        """
        Checks the automaton for a given LTL specification

        :param file:
        :param x:
        :param verbose:
        :return:
        """
        # convert graph to nuXmv model
        self.convertToNuXmv(file, x, lang="LTL")
        # TODO: extract this and make it easier to change
        # nuxmv = "E:\\Programs\\nuXmv-2.0.0-win64\\bin\\nuXmv.exe"
        # nuxmv = "/home/colin/Downloads/nuXmv-2.0.0-Linux/bin/nuXmv"
        nuxmv = "nuXmv"
        out = subprocess.run([nuxmv, file], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # TODO: a more robust "api"
        #  - this shouldn't fail because nuXmv only prints "true" (lower case) when the spec is satisfied, or if a state
        #  name or label is called "true", but the way convertToNuXmv is written, no state, label, or name can be "true"
        check = "true" in str(out.stdout)
        if verbose:
            print(out.stdout)
        return check

    def checkPCTL(self, m_file, p_file, x, verbose=False, solver="prism", strategic=False):
        """
        Checks the automaton for a given PCTL specification

        :param m_file:
        :param p_file:
        :param x:
        :param verbose:
        :param strategic:
        :param solver:
        :return:
        """
        if solver == "prism":
            # convert graph to PRISM model
            self.convertToPRISM(m_file, p_file, x)
            # TODO: extract this and make it easier to change
            prism = "/home/colin/prism-4.7-linux64/bin/prism"
            out = subprocess.run([prism, m_file, p_file], stdout=subprocess.PIPE)
            check = "true" in str(out.stdout)
            if verbose:
                print(out.stdout)
            return check
        elif solver == "storm":
            # convert graph to storm model
            if strategic:
                sparse_model = self.convertToStormDTMC()
            else:
                sparse_model = self.convertToStormMDP()
            property = stormpy.parse_properties(x)
            result = stormpy.model_checking(sparse_model, property[0], True)
            check = result.at(self.q0)
            return check
        else:
            # TODO: throw exception for no good solver specified
            return None

    # TODO: checkLTL by lang="LTL", tag a spec with CTL or LTL
    def checkToCTL(self, file, x, negate=False, verbose=False):
        """
        Checks an automaton for a CTL specification, given an LTL specification.

        :param file:
        :param x:
        :param negate:
        :param verbose:
        :return:
        """
        if negate:
            return self.checkCTL(file, '!E' + x, verbose=verbose)
        else:
            return self.checkCTL(file, 'A' + x, verbose=verbose)

    def convertToNuXmv(self, file, x=None, lang="CTL", return_string=False):
        """
        Produces a NuXmv input file specifying this automaton.
        :param file:
        :param x:
        :param return_string:
        :return:
        """

        open_mode = 'w'

        out_str = ""

        with open(file, open_mode) as f:
            f.write("MODULE main\n\n")
            self._writeStatesNuXmv(f)

            self._writeNamesNuXmv(f)

            self._writeVarsNuXmv(f)

            # begin ASSIGN constraint for state and name transitions
            f.write("ASSIGN\n")

            # States:
            self._writeStateTransNuXmv(f)

            # Names:
            self._writeNameTransNuXmv(f)

            # Properties:
            self._writePropTransNuXmv(f)

            # Specification
            if x:
                f.write(lang.upper() + "SPEC " + x + ";")
                f.write("\n")

        if return_string:
            with open(file, 'r') as f:
                out_str = f.read()
            return out_str

    def _writeStatesNuXmv(self, f):
        sep = ', '
        # include each vertex as a state in the model
        states = [str(v.index) for v in self.graph.vs]
        states = sep.join(states)
        f.write("VAR state: {" + states + "};\n\n")

    def _writeNamesNuXmv(self, f):
        sep = ', '
        # since multiple states can be associated with the same state of a
        # smaller original automaton, we want to track what that original
        # state is with a name variable
        names = self.graph.vs["name"]
        # remove duplicates from names
        names = list(set(names))
        # add names variable to model
        names = sep.join(names)
        f.write("VAR name: {" + names + "};\n\n")

    def _writeStateTransNuXmv(self, f):
        sep = ', '
        # set initial state
        f.write(" init(state) := " + str(self.q0) + ";\n")
        # define state transitions
        f.write(" next(state) :=\n")
        f.write("  case\n")
        # for each vertex...
        for v in self.graph.vs:
            # ... get a string representation of all the vertex's successors
            next_v = [str(vx.index) for vx in v.neighbors(mode=OUT)]
            # and a string rep of this vertex
            state = str(v.index)
            # and write out the transitions to the case
            if next_v:
                next_v = sep.join(next_v)
                f.write("   state = " + state + " : {" + next_v + "};\n")

        # default case
        f.write("   TRUE : state;\n")
        f.write("  esac;\n")
        f.write("\n")

    def _writeNameTransNuXmv(self, f):
        # set initial name
        init_name = self.graph.vs["name"][self.q0]
        f.write(" init(name) := " + str(init_name) + ";\n")
        # define name transitions
        f.write(" next(name) :=\n")
        f.write("  case\n")
        # for each vertex...
        for v in self.graph.vs:
            # ... get that vertex's name
            v_name = v["name"]
            # and a string rep of this vertex
            state = str(v.index)
            # and write out the transitions to the case based on next state
            f.write("   next(state) = " + state + " : " + v_name + ";\n")
        # default case
        f.write("   TRUE : name;\n")
        f.write("  esac;\n")
        f.write("\n")

    def _writeVarsNuXmv(self, f):
        # if auto has a counter
        if self.counter:
            # ... then write the counter var
            c_name = str(self.counter[0])
            start = str(self.counter[1])
            end = str(self.counter[2])
            f.write("VAR " + c_name + " : " + start + " .. " + end + ";\n\n")

        # if auto has labels
        if 'label' in self.graph.vs.attributes():
            # ... then write the label var
            labels_size = str(len(self.propositions))
            f.write("VAR label: unsigned word[" + labels_size + "];\n\n")

    def _writePropTransNuXmv(self, f):
        # if auto has a counter
        if self.counter:
            # ... then write the counter transitions
            c = str(self.counter[0])
            t = str(self.counter[2])
            f.write(" init(" + c + ") := " + str(self.counter[1]) + ";\n")
            f.write(" next(" + c + ") := (" + c + "<" + t + ")?(" + c + "+1):(" +
                    c + ");\n\n")

        # if auto has labels
        if 'label' in self.graph.vs.attributes():
            # set up translation between label strings and bit strings.
            # bit strings get used in nuXmv to represent which propositions a state has.
            # e.g. if the language has 4 propositions ['p', 'q', 'r', 's'], and state 0 has propositions "p, s",
            # then the bit string representing the labels of state 0 would be 0ub8_1001.
            word_size = len(self.propositions)
            no_labels = [0] * len(self.propositions)
            prop_dict = {}
            for i, prop in enumerate(self.propositions):
                label_bits = copy(no_labels)
                label_bits[i] = 1
                prop_dict[prop] = label_bits

            # set initial label
            # get the string representation of state labels, e.g. "p, s"
            init_props = self.graph.vs["label"][self.q0].split(', ')
            # get a list of bit string representations for each proposition, e.g. [[1, 0, 0, 0], [0, 0, 0, 1]]
            init_props_bits = [prop_dict[prop] for prop in init_props]
            # combine those bit string representations into a single bit string, e.g. ['1', '0', '0', '1']
            init_label = [str(sum(bits)) for bits in zip(*init_props_bits)]
            # join those bits into a string, e.g. "1001"
            init_label_str = ''.join(init_label)
            f.write(" init(label) := 0ub" + str(word_size) + "_" + str(init_label_str) + ";\n")
            # define label transitions
            f.write(" next(label) :=\n")
            f.write("  case\n")
            # for each vertex...
            for v in self.graph.vs:
                # ... get that vertex's propositions
                v_props = v["label"].split(', ')
                # then a list of bits
                v_props_bits = [prop_dict[prop] for prop in v_props]
                # combine those into a bit string
                v_label = [str(sum(bits)) for bits in zip(*v_props_bits)]
                v_label_str = ''.join(v_label)
                # and get a string rep of this vertex
                state = str(v.index)
                # and write out the transitions to the case based on next state
                f.write("   next(state) = " + state + " : 0ub" + str(word_size) + "_" + v_label_str + ";\n")
            # default case
            f.write("   TRUE : label;\n")
            f.write("  esac;\n")
            f.write("\n")

            # define relationship between bit words and original propositions
            f.write("DEFINE\n")
            for prop in self.propositions:
                # build logical equivalence string
                logical_bits = prop_dict[prop]
                logical_bits = ''.join([str(bit) for bit in logical_bits])
                f.write(" " + str(prop) + " := (0ub" + str(word_size) + "_" + logical_bits + " & label) = 0ub" +
                        str(word_size) + "_" + logical_bits + ";\n")
            f.write("\n")

    def convertToPRISM(self, m_file, p_file, x):
        """
        Produces a PRISM model file specifying this automaton, and a properties file specifying the formula
        :param m_file:
        :param p_file:
        :param x:

        :return:
        """
        with open(m_file, 'w') as f:
            f.write("mdp\n")
            f.write("module main\n")

            states = [v.index for v in self.graph.vs]
            names = self.graph.vs["name"]
            names = list(set(names))
            names = [int(namei) for namei in names]
            # make a state variable that goes from 0 to number of states; init q0.id
            f.write("state : [0.." + str(max(states)) + "] init " + str(self.q0) + ";\n")
            # make a name variable that goes from 0 to maximum name; init q0.name
            f.write("name : [0.." + str(max(names)) + "] init " + self.graph.vs["name"][self.q0] + ";\n")
            # for each vertex...
            for v in self.graph.vs:
                # for each action at vertex...
                for k in self.k(v):
                    # initialize command string
                    command = "    [] state=" + str(v.index) + " -> "
                    plus = ""
                    # get the edges from this vertex that are part of this action
                    esi = self.graph.es.select(_source_eq=v, action_eq=k)
                    # for each edge in that action...
                    for e in esi:
                        prob = str(e["prob"])
                        tgt_id = str(e.target_vertex.index)
                        tgt_nm = str(e.target_vertex["name"])
                        command += plus + prob + " : (state'=" + tgt_id + ")&(name'=" + tgt_nm + ")"
                        if not plus:
                            plus = " + "
                    command += ";\n"
                    f.write(command)
                f.write("\n")

            f.write("endmodule")

        with open(p_file, 'w') as f:
            # Specification
            f.write(x)
            f.write("\n")

    def convertToStormPDTMC(self, spec=None, reward=None, return_components=False):
        import stormpy.info
        import stormpy.pars
        if stormpy.info.storm_ratfunc_use_cln():
            import pycarl.cln as pc
        else:
            import pycarl.gmp as pc

        from pycarl.core import Variable

        def create_polynomial(pol):
            num = pc.create_factorized_polynomial(pc.Polynomial(pol))
            return pc.FactorizedRationalFunction(num)

        def create_number(num):
            num = pc.FactorizedPolynomial(pc.Rational(num))
            return pc.FactorizedRationalFunction(num)

        builder = stormpy.ParametricSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                                        has_custom_row_grouping=False)

        nr_states = self.graph.vcount()
        state_labeling = stormpy.storage.StateLabeling(nr_states)
        labels = self._getPropositionsFromLabels()
        state_labeling.add_label('init')
        result = None
        for label in labels:
            state_labeling.add_label(label)
        if spec:
            storm_mdp = self.convertToStormMDP()
            spec = stormpy.parse_properties(spec)
            result = stormpy.model_checking(storm_mdp, spec[0])
        state_reward = []
        # for each state-action pair, create a Variable that will be a theta and a parameter
        theta = []
        params = []
        # for each transition probability, create a number.
        state_labeling.add_label_to_state('init', self.q0)
        for i, state in enumerate(self.graph.vs):
            actions = self.k(state)
            theta_s = []
            for k in actions:
                var = Variable(str(i)+"_"+str(k))
                params.append(var)
                theta_s.append(create_polynomial(var))
            theta.append(theta_s)
            # set state rewards
            r = create_number(0.01)
            if spec:
                # magic numbers scaled and subtracted to help with precision issues
                r = create_number((100*result.at(state.index))-101)
            elif reward is not None:
                # TODO: in the case that rewards are r(s, a) make a state-action reward vector
                # TODO: in the case that rewards are r(s, a, s') reduce them to a state-action reward vector
                if type(reward) is list:
                    r = create_number(reward[i])
                else:
                    r = create_number(state['reward'])
            state_reward.append(r)
            probs_by_successor = defaultdict(list)
            for k, action in enumerate(actions):
                # get the edges from this vertex that are part of this action
                edges = self.graph.es.select(_source_eq=state, action_eq=action)
                probs = edges['prob']
                for j, edge in enumerate(edges):
                    if probs[j] != 0:
                        theta_sa = theta[i][k] * theta[i][k]
                        denom = theta[i][0] * theta[i][0]
                        for var_idx in range(1, len(theta[i])):
                            denom += theta[i][var_idx] * theta[i][var_idx]
                        action_prob = theta_sa/denom
                        probs_by_successor[edge.tuple[1]] += [action_prob*create_number(probs[j])]
            for key in probs_by_successor.keys():
                probs_list = probs_by_successor[key]
                current_prob = probs_list[0]
                for l in range(1, len(probs_list)):
                    current_prob += probs_list[l]
                builder.add_next_value(state.index, key, current_prob)
            # get state labels and add them
            state_labels = state['property'].split(", ")
            for state_label in state_labels:
                state_labeling.add_label_to_state(self._formatLabel(state_label)[0], i)

        transition_matrix = builder.build()
        # print("matrix built")
        reward_models = {'state_reward': stormpy.SparseParametricRewardModel(optional_state_reward_vector=state_reward)}
        components = stormpy.SparseParametricModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling,
                                                             rate_transitions=False, reward_models=reward_models)
        storm_pdtmc = stormpy.storage.SparseParametricDtmc(components)
        # print("model built")
        if return_components:
            return storm_pdtmc, params, transition_matrix, state_labeling, reward_models
        else:
            return storm_pdtmc, params

    def constructStormPDTMCfromComponents(self, rewards, transition_matrix, state_labeling):
        import stormpy.info
        import stormpy.pars
        if stormpy.info.storm_ratfunc_use_cln():
            import pycarl.cln as pc
        else:
            import pycarl.gmp as pc

        from pycarl.core import Variable

        def create_polynomial(pol):
            num = pc.create_factorized_polynomial(pc.Polynomial(pol))
            return pc.FactorizedRationalFunction(num)

        def create_number(num):
            num = pc.FactorizedPolynomial(pc.Rational(num))
            return pc.FactorizedRationalFunction(num)

        nr_states = self.graph.vcount()
        state_reward = []
        for i, state in enumerate(self.graph.vs):
            r = create_number(rewards[i])
            state_reward.append(r)

        reward_models = {'state_reward': stormpy.SparseParametricRewardModel(optional_state_reward_vector=state_reward)}
        components = stormpy.SparseParametricModelComponents(transition_matrix=transition_matrix,
                                                             state_labeling=state_labeling,
                                                             rate_transitions=False, reward_models=reward_models)
        storm_pdtmc = stormpy.storage.SparseParametricDtmc(components)
        return storm_pdtmc

    def convertToDRN(self, m_file, spec=None, reward=None):
        if spec:
            storm_mdp = self.convertToStormMDP()
            spec = stormpy.parse_properties(spec)
            result = stormpy.model_checking(storm_mdp, spec[0])
        nr_states = self.graph.vcount()
        # actions = list(set(self.graph.es['action']))
        params = []
        model_string = ""
        for v in tqdm(self.graph.vs):
            model_string += "state " + str(v.index) + " "
            if self.q0 == v.index:
                model_string += " init "
            v_labels = v['label']
            v_labels = self._formatLabel(v_labels)
            model_string += ' '.join(v_labels)
            model_string += "\n"
            r = 0.01
            if spec:
                r = str(result.at(v.index))
            elif reward is not None:
                r = str(self.graph.vs[v.index]['reward'])
            model_string += "\taction 0 [" + r + "]\n"

            v_edges = self.graph.es.select(_source_eq=v)
            actions = list(set(v_edges['action']))
            actions.sort()
            v_params = ["v"+str(v.index)+"a"+str(a) for a in actions]
            params += v_params
            v_params_sq = [v_param + "*" + v_param for v_param in v_params]
            v_denom = '+'.join(v_params_sq)
            for vn in set(v.successors()):
                model_string += "\t\t" + str(vn.index) + " : "
                prob_list = []
                for a, action in enumerate(actions):
                    va_edges = v_edges.select(_target_eq=vn.index, action=action)
                    for e in va_edges:
                        prob = str(e['prob']) + "*" + v_params_sq[a] + "/(" + v_denom + ")"
                        prob_list.append(prob)
                model_string += "+".join(prob_list)
                model_string += "\n"

        params_str = ' '.join(params)

        with open(m_file, 'w') as f:
            f.write("@type: DTMC\n")
            f.write("@parameters\n")
            f.write(params_str)
            f.write("\n")
            f.write("@reward_models\n\n")
            f.write("@nr_states\n")
            f.write(str(nr_states))
            f.write("\n")
            f.write("@nr_choices\n")
            f.write(str(nr_states))
            f.write("\n")
            f.write("@model\n")
            f.write(model_string)
        return params

    def convertToStormDTMC(self):
        builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                              has_custom_row_grouping=False)

        nr_states = self.graph.vcount()
        state_labeling = stormpy.storage.StateLabeling(nr_states)
        labels = self._getPropositionsFromLabels()
        state_labeling.add_label('init')
        for label in labels:
            state_labeling.add_label(label)

        state_labeling.add_label_to_state('init', self.q0)
        # edges = self.graph.es.select(prob_gt=0)
        # for e in edges:
        #     builder.add_next_value(e.source, e.target, e['prob'])
        # edge_tuples = self.graph.to_tuple_list(edge_attrs="prob")
        # edge_tuples = sorted(edge_tuples, key=lambda x: (x[0], x[1]))
        # for tup in tqdm(edge_tuples):
        #     if tup[2] > 0:
        #         builder.add_next_value(tup[0], tup[1], tup[2])
        for i, state in tqdm(enumerate(self.graph.vs)):
            actions = self.k(state)
            for action in actions:
                # get the edges from this vertex that are part of this action
                edges = self.graph.es.select(_source_eq=state, action_eq=action)
                probs = np.array(edges['prob'])/np.sum(edges['prob'])
                for j, edge in enumerate(edges):
                    if probs[j] != 0:
                        builder.add_next_value(edge.tuple[0], edge.tuple[1], probs[j])
            # get state labels and add them
            state_labels = state['property'].split(", ")
            for state_label in state_labels:
                state_labeling.add_label_to_state(state_label, i)

        transition_matrix = builder.build()
        print("matrix built")
        components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling,
                                                   rate_transitions=False)
        storm_dtmc = stormpy.storage.SparseDtmc(components)
        print("model built")
        return storm_dtmc

    def convertToStormMDP(self):
        builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                              has_custom_row_grouping=True, row_groups=0)
        reward_builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                                     has_custom_row_grouping=True, row_groups=0)
        nr_states = self.graph.vcount()
        state_labeling = stormpy.storage.StateLabeling(nr_states)
        labels = self._getPropositionsFromLabels()
        state_labeling.add_label('init')
        for label in labels:
            state_labeling.add_label(label)

        state_labeling.add_label_to_state('init', self.q0)
        reward_models = {}
        transition_reward = []
        action_id = 0
        # builder.new_row_group(action_id)
        for i, state in tqdm(enumerate(self.graph.vs)):
            builder.new_row_group(action_id)
            reward_builder.new_row_group(action_id)
            actions = self.k(state)
            used_action = False
            for action in actions:
                # get the edges from this vertex that are part of this action
                edges = self.graph.es.select(_source_eq=state, action_eq=action)
                for edge in edges:
                    if edge["prob"] != 0:
                        builder.add_next_value(action_id, edge.tuple[1], edge["prob"])
                        reward_builder.add_next_value(action_id, edge.tuple[1], edge["weight"])
                        used_action = True
                    # transition_reward.append(edge["weight"])
                if used_action:
                    action_id += 1
            # get state labels and add them
            state_labels = self._formatLabel(state['property'])
            for state_label in state_labels:
                state_labeling.add_label_to_state(state_label, i)

        reward_matrix = reward_builder.build()
        reward_models['reward'] = stormpy.SparseRewardModel(optional_transition_reward_matrix=reward_matrix)
        transition_matrix = builder.build()
        print("matrix built")
        components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling,
                                                   reward_models=reward_models, rate_transitions=False)
        storm_mdp = stormpy.storage.SparseMdp(components)
        print("model built")
        return storm_mdp

    def _getPropositionsFromLabels(self):
        labels = self.graph.vs['property']
        label_list = []
        for string in labels:
            label_list += self._formatLabel(string)
        return set(label_list)

    def _formatLabel(self, label_string):
        label_string = label_string.replace('(', 'x')
        label_string = re.sub(r'(\d+)\)', r'y\1', label_string)
        # label_string = label_string.replace(')', 'y')
        return label_string.split(", ")

    def convertToMatrix(self, labeled_mat=False):
        """
        Get a matrix representation of the automaton augmented with the labels on each state; e.g. for a two state
        automaton with two transitions: one from the first state to the second state, and one from the second state to
        itself. The first state is labeled "a, b" and the second state is labeled "a, c".
        if labeled_mat is True, then the output is like [['0', 'a, c'], ['0', 'a, c']].
        otherwise: [("a, b", [0, 1]), ("a, c", [0, 1])]
        :return:
        """
        mat = np.array(self.graph.get_adjacency().data)
        labels = self.graph.vs["label"]
        if labeled_mat:
            return np.where(np.array(mat) > 0, labels, 0)
        mat = np.where(mat > 0, 1, 0).tolist()
        if not mat:
            return '0'
        return list(zip(labels, mat))


class Obligation(object):
    """
    Contains an obligation in Dominance Act Utilitarian deontic logic
    """

    def __init__(self, phi, is_ctls, is_neg, is_pctl=False):
        """
        Creates an Obligation object

        :param phi:
        :param is_ctls:
        :param is_neg:
        """
        self.phi = phi
        self.is_ctls = is_ctls
        self.is_neg = is_neg
        self.is_stit = not is_ctls
        self.is_pctl = is_pctl
        self.phi_neg = False

    @classmethod
    def fromCTL(cls, phi):
        """
        Creates an Obligation object from a CTL string

        :param phi:
        :return:
        """
        return cls(phi, True, False)

    @classmethod
    def fromPCTL(cls, phi):
        """
        Creates an Obligation object from a PCTL string

        :param phi:
        :return:
        """
        return cls(phi, False, False, True)

    def isCTLS(self):
        """
        Checks if obligation is a well-formed CTL* formula

        :return:
        """
        # TODO: use grammar to check this
        return self.is_ctls

    def isPCTL(self):
        """
        Checks if obligation is a well-formed PCTL formula

        :return:
        """
        # TODO: use grammar to check this
        return self.is_pctl

    def isSTIT(self):
        """
        Checks if obligation is a well-formed dstit statement

        :return:
        """
        # TODO: use grammar to check this
        return self.is_stit

    def isNegSTIT(self):
        """
        Checks if obligation is of the form ![alpha dstit: phi]

        :return:
        """
        return self.is_stit and self.is_neg

    def getPhi(self):
        """
        Gets the inner formula of the obligation

        :return:
        """
        return self.phi


def checkObligation(g, a, verbose=False):
    """
    Check an automaton for if it has a given obligation.

    :param g:
    :param a:
    :param verbose:
    :return:
    """
    # if the automaton g has a policy, then just force the optimal action and check the tense formula
    if g.has_policy:
        k_star = g.policy[g.q0]
        gn = deepcopy(g)
        gn = gn.forceKn(k_star, g.q0)
        truth = gn.checkPCTL('temp.sm', 'temp.pctl', a.getPhi(), solver='storm')
        return truth

    # return checkConditional with trivial condition params
    return checkConditional(g, a, "TRUE", 0, verbose=verbose)


def checkStrategicObligation(g, a, verbose=False):
    """
    Check an automaton for if it has a given strategic obligation

    :param g:
    :param a:
    :param verbose:
    :return:
    """
    # TODO: extract the bit that gets gn so I can call that by itself
    # if the automaton g has a policy, then just force the optimal actions and check the tense formula
    if g.has_policy:
        gn = getStrategicAutomaton(g)
        # print(len(gn.graph.vs))
        # now that edges belonging to suboptimal actions have been deleted, check the remaining model
        # this might need to be a markov chain instead of a MDP?
        truth = gn.checkPCTL('temp.sm', 'temp.pctl', a.getPhi(), solver='storm', strategic=True)
        return truth


def getStrategicAutomaton(g, policy=None):
    """
    Apply the Automaton's strategy/policy/scheduler and return an Automaton whose graph is restricted to the specified
    actions.

    :param g:
    :return:
    """
    if g.has_policy:
        policy = g.policy
    gn = deepcopy(g)
    # for each state, get the optimal actions and force them.
    for s in tqdm(gn.graph.vs):
        # TODO: Technically a policy could return multiple actions here (but it doesn't), so handle that
        k_star = policy[int(s['name'])]
        # maybe I should still treat strat model as mdp but copy the optimal action?
        add_optimal_action_again = True
        if add_optimal_action_again:
            pass
        # get the edges from this vertex that are not part of this action
        e_bad = gn.graph.es.select(_source_eq=s, action_ne=k_star)
        # mark those edges for deletion
        e_bad['delete'] = 1
    # get all the edges marked for deletion
    candidates = gn.graph.es.select(delete=1)
    # and delete them
    gn.graph.delete_edges(candidates)
    # find all states that can not be entered
    candidates = gn.graph.vs.select(_indegree=0)
    while len(candidates) != 0:
        # find edges that have no source
        e_bad = gn.graph.es.select(_source_in=candidates)
        # and delete them
        gn.graph.delete_edges(e_bad)
        # then delete the vertices
        gn.graph.delete_vertices(candidates)
        # then find if these deletions made any more orphans
        candidates = gn.graph.vs.select(_indegree=0)
    return gn



# TODO: refactor checkConditional into smaller functions so I can use some of the juicy bits elsewhere
def checkConditional(g, a, x, t, verbose=False):
    """
    Check an automaton for if it has a given obligation under a given condition.

    :param g:
    :param a:
    :param x:
    :param t:
    :param verbose:
    :return:
    """
    optimal = get_optimal_automata(g, t, x, verbose)

    for m in optimal:
        truth_n = True
        if verbose:
            print(m[0])
        if a.isCTLS():
            # truth_n = m[1].checkToCTL('temp.smv', a.getPhi(), a.phi_neg,
                                      # verbose=verbose)
            truth_n = m[1].checkCTL('temp.smv', a.getPhi(), a.phi_neg)
        elif a.isPCTL():
            truth_n = m[1].checkPCTL('temp.sm', 'temp.pctl', a.getPhi())
        elif a.isSTIT():
            phi = a.getPhi()
            if not a.isNegSTIT():
                delib = not g.checkToCTL('temp.smv', phi, a.phi_neg,
                                         verbose=verbose)
                guaranteed = m[1].checkToCTL('temp.smv', phi, a.phi_neg,
                                             verbose=verbose)
                if verbose:
                    print("deliberate: ", delib)
                    print("guaranteed: ", guaranteed)
                truth_n = delib and guaranteed
            else:
                not_delib = g.checkToCTL('temp.smv', phi, a.phi_neg,
                                         verbose=verbose)
                guaranteed = m[1].checkToCTL('temp.smv', phi, a.phi_neg,
                                             verbose=verbose)
                if verbose:
                    print("not deliberate: ", not_delib)
                    print("not guaranteed: ", not guaranteed)
                truth_n = not_delib or not guaranteed
        else:
            raise ValueError(
                'The given obligation was not a well formed (P)CTL* formula, ' +
                'nor a well formed deliberative STIT statement.',
                a)
        if not truth_n:
            return False

    return True


# TODO: consider returning a list of dictionaries
# TODO: easier evaluation of automaton value when x="TRUE", so use that case
def get_choice_automata(g, t, x="TRUE", return_fragments=False):
    """
    given an automaton g, a time horizon t, and a horizon-limited condition x, generate:
    a list of tuples (action, act_automaton, interval); where action is an action of the
    automaton g available at g.q0 (the starting state of g), act_automaton is the
    automaton generated when g can only take the corresponding action from q0, and
    interval is a list containing the highest and lowest scores of histories of length t
    produced from the act_automaton.

    :param g:
    :param t:
    :param x:
    :param return_fragments:
    :return:
    """
    root = g.q0
    choices = g.k(root)
    out = []
    frags = []
    l = len(choices)
    discount = 0.5
    # for each choice available from start...
    for n in np.arange(l):
        kn = choices[n]
        gn = deepcopy(g)
        gn = gn.forceKn(kn, source=root)
        gnr = deepcopy(gn)
        gnr.q_previous.append(-1)
        gnp = gnr.union(g, target=root)
        if x == "TRUE" and not return_fragments:
            # skip the hard stuff
            q_of_kn = gnp.optimal(discount)
            out.append((kn, gnp, q_of_kn))
            continue
        # get a list of automata whose first action is kn, and have one history
        # up to depth t, and that history satisfies X, and after that it behaves
        # like g
        if t <= 0:
            t += 1
        gns = generate_fragments(gnp, g, root, x, t)
        lows = []
        highs = []
        if gns:
            # there are condition-satisfying histories, so gnp is in choice/|x|
            if g.prob:
                # we're dealing with a probabilistic automaton, get the conditional automaton
                cond_auto = build_fragment_tree(gns, g)
                q_of_kn = cond_auto.optimal(discount)
                out.append((kn, gnp, q_of_kn))
                for gf in gns:
                    frags.append((kn, gf))
            else:
                for gf in gns:
                    lows.append(gf.optimal(discount, best=False))
                    highs.append(gf.optimal(discount, best=True))
                    frags.append((kn, gf))
                interval = [np.max(lows), np.max(highs)]
                out.append((kn, gnp, interval))
        else:
            raise RuntimeError("No fragments found: maybe the condition is not satisfiable, or transitions are missing")

    if return_fragments:
        return out, frags
    else:
        return out


def get_choice_fragments(g, t, x="TRUE"):
    """
    given an automaton g, a time horizon t, and a horizon-limited condition x, generate:
    a list of tuples (action, act_fragment); where action is an action of the
    automaton g available at g.q0 (the starting state of g), act_fragment is the
    fragment generated when g can only take the corresponding action from q0 and has only
    one history up to depth t.

    :param g:
    :param t:
    :param x:
    :return:
    """
    root = g.q0
    choices = g.k(root)
    out = []
    l = len(choices)
    # for each choice available from start...
    for n in np.arange(l):
        kn = choices[n]
        gn = deepcopy(g)
        gn = gn.forceKn(kn, source=root)
        gnr = deepcopy(gn)
        gnr.q_previous.append(-1)
        gnp = gnr.union(g, target=root)
        # get a list of automata whose first action is kn, and have one history
        # up to depth t, and that history satisfies X, and after that it behaves
        # like g
        if t <= 0:
            t += 1
        gns = generate_fragments(gnp, g, root, x, t)
        if gns:
            for gf in gns:
                out.append((kn, gf))

    return out


def get_optimal_automata(g, t, x="TRUE", verbose=False):
    """
    given an automaton g, a time horizon t, and a horizon-limited condition x, generate:
    a list of tuples (opt_action, opt_automaton) where opt_action is an action available
    to g at g.q0 (the starting state of g), and opt_automaton is the automaton generated
    when g can only take the corresponding opt_action from q0. opt_action is the
    dominance optimal action available to g.q0.

    :param g:
    :param t:
    :param x:
    :param verbose:
    :return:
    """
    choices = get_choice_automata(g, t, x)
    return choose_optimal_automata(choices, verbose)


def choose_optimal_automata(choices, verbose=False):
    """
    given a list of tuples (action, act_automaton, interval), generate:
    a list of tuples (opt_action, opt_automaton) where opt_action is an action available
    to g at g.q0 (the starting state of g), and opt_automaton is the automaton generated
    when g can only take the corresponding opt_action from q0. opt_action is the
    dominance optimal action available to g.q0.

    :param choices:
    :param verbose:
    :return:
    """
    intervals = [choice[2] for choice in choices]

    optimal = []
    if not intervals:
        if verbose:
            print("No Intervals")
        return False

    if choices[0][1].prob:
        # find all automata whose expected utility is the best
        v_max = np.max(intervals)
        for i, interval in enumerate(intervals):
            if interval >= v_max:
                if verbose:
                    print(choices[i][0], interval)
                optimal.append((choices[i][0], choices[i][1]))
    else:
        # find all un-dominated intervals
        # optimal carries tuples containing an optimal action and an automaton
        # whose first action is that optimal action.
        inf = np.max(np.min(intervals, axis=1))
        for i, interval in enumerate(intervals):
            if interval[1] >= inf:
                if verbose:
                    print(choices[i][0], interval)
                optimal.append((choices[i][0], choices[i][1]))

    return optimal


def generate_fragments(gn, g0, q0, x, t, check_at_end=True):
    """
    Given an Automaton gn, a prototype Automaton g0, a starting state q0,
    a finite horizon condition x, and the length of that horizon t, generate
    a list of all Automata that start from q0 and have only one history up to
    depth t, that history satisfies x, and after t the Automaton behaves like
    g0.

    If check_at_end is true, then generate_fragments will generate all possible
    fragments of length t, and check each for satisfaction of x. Otherwise, it
    checks each as they're built.

    :param gn:
    :param g0:
    :param q0:
    :param x:
    :param t:
    :param check_at_end:
    :return:
    """

    g = deepcopy(gn)
    # set a clock on the automaton so the condition can be horizon limited
    g.setCounter(var_name="fragmentc")
    # set up the condition to be checked in each step
    f = "E [" + x + " U " + "(fragmentc = " + str(t) + ")]"
    # f = 'E' + x
    # make sure we start from the right state
    g.qn = q0
    # initialize the list of systems with the given system
    systems = [g]
    # until we reach the given horizon...
    for i in trange(t):
        new_systems = []
        # ... for every system we have so far...
        for system in systems:
            # ... get each possible transition for that system from its current state...
            possible_edges = system.graph.es.select(_source=system.qn)
            # ... and for each possible transition...
            for edge in possible_edges:
                state = edge.tuple[1]
                # copy the system
                sys_n = deepcopy(system)
                # force the transition
                sys_n = sys_n.forceEn(edge, source=system.qn)
                sys_n_ren = deepcopy(sys_n)
                # update the list of previous states
                sys_n_ren.q_previous.append(sys_n_ren.qn)
                # tack the prototype system onto the end
                sys_n_prime = sys_n_ren.union(g0, state)
                # if this new system satisfies the condition...
                if not check_at_end:
                    if sys_n_prime.checkCTL("temp.smv", f):
                        # set the system's current state to the only possible next state
                        sys_n_prime.qn = sys_n_prime.graph.neighbors(sys_n_prime.qn, mode=OUT)[0]
                        # and add the system to our list of systems.
                        new_systems.append(sys_n_prime)
                else:
                    # set the system's current state to the only possible next state
                    sys_n_prime.qn = sys_n_prime.graph.neighbors(sys_n_prime.qn, mode=OUT)[0]
                    # and add the system to our list of systems.
                    new_systems.append(sys_n_prime)
        # all systems have been stepped through, and the satisfactory systems
        # get to make it to the next round.
        systems = new_systems
    # now that all the systems in our list are deterministic to depth t
    # we cut the "chaff" from each automaton, because we added a lot of superfluous states and transitions
    # so, for each system...
    good_systems = []
    for system in tqdm(systems):
        # find out what the identifier is for the last prototype we tacked on
        clone_no = "-" + str(system.num_clones)
        # get all the old vertices we want to be keeping
        path = system.q_previous
        path.append(system.qn)
        # and set up a list to retain the vertices we want to delete
        del_v_id = []
        # then for each vertex in our graph...
        for v in system.graph.vs:
            # if that vertex is not one we want to keep and it's not in our last prototype tacked on...
            if v.index not in path and clone_no not in v["label"]:
                # add it to our delete list
                del_v_id.append(v.index)
        # make our delete list a proper vertex sequence so it can be deleted
        del_vs = VertexSeq(system.graph, del_v_id)
        # and delete those vertices! (And any associated edges)
        system.graph.delete_vertices(del_vs)
        # set the q0 of this pared-down system to what it should be
        system.q0 = system.graph.vs.find(name=str(q0)).index
        system.t = t
        if check_at_end:
            if system.checkCTL("temp.smv", f):
                good_systems.append(system)

    if not check_at_end:
        good_systems = systems
    return good_systems


def build_fragment_tree(fragments, g0):
    """
    Build a tree from a given set of fragments such that each leaf of the tree is the end of one of the
    fragments, and each leaf leads into g0 - the original automaton.
    Return an Automaton based on the tree.

    When the fragments are generated with a condition, then the tree may be considered to be a conditional
    automaton.

    :param fragments:
    :param g0:
    :return:
    """
    q0 = fragments[0].q0
    t = fragments[0].t
    g_new = Graph(directed=True)
    v_new = g_new.add_vertex(name=fragments[0].graph.vs[q0]["name"])
    # propagate other attributes of v0
    for attr in fragments[0].graph.vs[q0].attribute_names():
        v_new[attr] = fragments[0].graph.vs[q0][attr]
    # track the fragments on each front of the tree
    frag_partitions = [[fragments, q0]]
    # track the front of the tree
    front_vs = [q0]
    visited = [q0]
    prob = False
    # for each level of the tree....
    for _ in range(t):
        # for each branch (starting from none)...
        new_front = []
        new_partitions = []
        partition_assignment = {}
        partition_idx = 0
        for frag_partition, qn in frag_partitions:
            # get each edge from this front of the tree...
            actions = []
            unique_edges = []
            signatures = []
            # for each history in this branch...
            for fragment in frag_partition:
                edge = fragment.graph.es.find(_source=fragment.q0)
                # add that edge's action to the list of actions
                actions.append(edge["action"])
                # get that edge's action
                target = edge.target_vertex
                # determine the proper signature of the edge
                if "prob" in edge.attribute_names():
                    prob = True
                    edge_sig = (qn, edge["action"], edge["weight"], edge["prob"], target["name"])
                else:
                    edge_sig = (qn, edge["action"], edge["weight"], target["name"])
                if edge_sig not in signatures:
                    # if we haven't seen this signature before, record it
                    signatures.append(edge_sig)
                    unique_edges.append(edge)
                    # remember where we put fragments with this signature
                    partition_assignment[edge_sig] = partition_idx
                    # put this fragment in its partition
                    new_partitions.append([[fragment], None])
                    # remember where to put the next fragment with a new signature
                    partition_idx += 1
                else:
                    # we've seen this signature before, so we know where to put its fragment
                    temp_part_idx = partition_assignment[edge_sig]
                    new_partitions[temp_part_idx][0].append(fragment)
                # now set the fragment's q0 to its next state
                fragment.q0 = target.index
            # get the set of actions among those edges...
            actions = set(actions)
            # for each action...
            for action in actions:
                # get the edges in that action
                act_es = [edge for edge in unique_edges if edge["action"] == action]
                act_probs = []
                if prob:
                    act_probs = [edge["prob"] for edge in act_es]
                    if np.sum(act_probs) == 0:
                        # there are no edges in this action, so

                        pass
                for edge in act_es:
                    # find target label
                    target = edge.target_vertex
                    # determine the signature of the edge
                    if prob:
                        edge_sig = (qn, edge["action"], edge["weight"], edge["prob"], target["name"])
                    else:
                        edge_sig = (qn, edge["action"], edge["weight"], target["name"])
                    temp_part_idx = partition_assignment[edge_sig]
                    # add it to the new graph, starting with the target
                    v_new = g_new.add_vertex()
                    visited.append(v_new.index)
                    new_front.append(v_new.index)
                    new_partitions[temp_part_idx][1] = v_new.index
                    # copy the attributes
                    for attr in target.attribute_names():
                        v_new[attr] = target[attr]
                    # now add the edge
                    e_new = g_new.add_edge(qn, v_new)
                    # copy the attributes
                    for attr in edge.attribute_names():
                        e_new[attr] = edge[attr]
                    # normalize the probability
                    if prob:
                        e_new["prob"] = e_new["prob"] / np.sum(act_probs)
        front_vs = new_front
        frag_partitions = new_partitions

    # turn the graph we've built into an automaton
    g_new.vs["target"] = [False] * g_new.vcount()
    g0.graph.vs["target"] = [True] * g0.graph.vcount()
    g_new = g_new.disjoint_union(g0.graph)
    move_edges = g_new.es.select(_target_in=front_vs)
    mv_edg_tuples = []
    mv_edg_attr = {}
    for attr in move_edges.attribute_names():
        mv_edg_attr[attr] = []
    for move_edge in move_edges:
        target_name = move_edge.target_vertex["name"]
        move_target = g_new.vs.select(name=target_name, target=True)[0]
        new_tuple = (move_edge.tuple[0], move_target.index)
        mv_edg_tuples.append(new_tuple)
        for attr in move_edge.attribute_names():
            mv_edg_attr[attr].append(move_edge[attr])
    g_new.delete_edges(move_edges)
    g_new.add_edges(mv_edg_tuples, mv_edg_attr)
    del_vs = VertexSeq(g_new, front_vs)
    g_new.delete_vertices(del_vs)

    cond_auto = Automaton(g_new, q0)
    cond_auto.q_previous = visited
    cond_auto.num_clones = t
    # cond_auto.graph = cond_auto.graph.disjoint_union(g0.graph)

    for fragment in fragments:
        fragment.q0 = q0

    return cond_auto
