"""
Created July 2020

@author: Colin
"""

from random_weighted_automaton import *
from model_check import *
import igraph as ig
import numpy as np
import pickle


def simpleExperiment():
    graph = pickle.load(open("example.pkl", "rb"))
    k = {0: [3], 1: [0], 2: [6], 3: [1, 4], 4: [7, 5], 5: [2]}
    g = Automaton.with_actions(graph, k)
    kn = 1
    gn = deepcopy(g)
    gn = gn.forceKn(kn)
    gnr = deepcopy(gn)
    gnp = gnr.union(g)
    gns = generate_fragments(gnp, g, 0, "F (name = 2 | name = 0)", 3)
    sol = g.optimal(0.5)
    print("optimal actions: ", sol)
    checkConditional(g, '2', 'F (name = 2)', 3)
    gns[0].graph.write_svg("graph1.svg")
    obl = Obligation.fromCTL("AF (state = 2)")
    out = gns[0].checkCTL("model.smv", obl.phi)
    print("AF (state = 2): ", out)


def originalObligations():
    ograph1 = Graph(n=10, edges=[(0, 1), (1, 2), (1, 3), (3, 2), (3, 4), (2, 2),
                                 (4, 5), (4, 6), (5, 4), (9, 9), (6, 7), (7, 8),
                                 (8, 8), (0, 9), (1, 9), (3, 9), (4, 9), (5, 9),
                                 (6, 9), (7, 9), (8, 9)],
                    directed=True)
    ograph1.es["weight"] = [5, 2, 5, 2, 7, 0, 5, 5, 3, 0, 7, 10, 5, 0, 0, 0, 0,
                            0, 0, 0, 0]
    k0 = {0: [0, 13], 1: [1, 2, 14], 2: [3, 4, 15], 3: [6, 7, 16], 4: [8, 17],
          5: [10, 18], 6: [11, 19], 7: [12, 20], 8: [9], 9: [5]}
    og = Automaton.with_actions(ograph1, k0)

    col_mission0 = og.checkCTL("temp.smv", "EG !(name = 9)")
    print("T0: Collision mission (EG !collision) = ", str(col_mission0))
    assert col_mission0

    exit_mission0 = og.checkCTL("temp.smv", "EF (name = 7)")
    print("T0: Exit mission (EF safe_to_exit) = ", str(exit_mission0))
    assert exit_mission0

    hwy_mission0 = og.checkCTL("temp.smv", "EF (name = 4)")
    print("T0: Highway mission (EF on_highway) = ", str(hwy_mission0))
    assert hwy_mission0

    safe_obl = Obligation.fromCTL("X !(name = 9)")

    og.q0 = 5
    has_safe0 = checkObligation(og, safe_obl)
    print("T0: Safety obligation (O[a cstit: X !collision]) = ", str(has_safe0))
    assert not has_safe0

    fast_obl = Obligation.fromCTL(" [TRUE U (name=6 & c<=4)]")
    fast_obl.phi_neg = True

    og.q0 = 0
    og.setCounter()
    has_fast0 = not checkObligation(og, fast_obl)
    print("T0: Fast obligation (!O[a cstit: !(True U reach_exit & c<=4)]) = ",
          str(has_fast0))
    assert has_fast0

    ast_obl = Obligation(" [! (name=4) U (name=6 | name=9)]", False, False)
    ast_obl.phi_neg = True

    check_state = 5

    og.q0 = check_state
    has_ast0 = not checkObligation(og, ast_obl)
    print("T0: Assertive obligation (!O[a cstit: [a dstit: !(!g U p)]]) = ",
          str(has_ast0))
    assert has_ast0

    agg_obl = Obligation(" [! (name=4) U (name=6 | name=9)]", False,
                         True)
    agg_obl.phi_neg = False

    og.q0 = check_state
    has_agg0 = not checkObligation(og, agg_obl)
    print("T0: Aggressive obligation (!O[a cstit: ![a dstit: (!g U p)]]) = ",
          str(has_agg0))
    assert not has_agg0


def setupAuto(safe=True):
    graph3 = Graph(n=13, edges=[(0, 1), (1, 2), (2, 2), (1, 3), (3, 2), (3, 2),
                                (3, 4), (4, 4), (4, 5), (4, 9), (5, 4), (5, 9),
                                (5, 12), (12, 12), (3, 6), (6, 6), (6, 7),
                                (7, 6), (6, 9), (9, 10), (10, 11), (11, 11)],
                   directed=True)

    if safe:
        # not assertive, not aggressive, safe:
        graph3.es["weight"] = [5, 1, 0, 5, 1, 1, 1, 5, 5, 5, 14, 5, 0, 0, 2, 5,
                               5, 5, 0, 10, 10, 5]
    else:
        # assertive, aggressive, unsafe:
        graph3.es["weight"] = [5, 1, 0, 5, 1, 1, 1, 5, 5, 5, 4, 5, 5, 0, 2, 5,
                               5, 1, 5, 10, 10, 5]

    # assertive, not aggressive,
    # k3 = {0: [0], 1: [1, 3], 2: [2], 3: [4, 6], 5: [7, 8], 7: [10, 11, 12],
    #       8: [13], 9: [14, 5], 10: [15, 16], 11: [17], 12: [18], 13: [19],
    #       14: [20], 15: [21]}

    k3 = {0: [0], 1: [1, 3], 2: [2], 3: [4, 6], 5: [7, 8], 6: [10], 7: [11, 12],
          8: [13], 9: [14, 5], 10: [15, 16], 11: [17], 12: [18], 13: [19],
          14: [20], 15: [21], 16: [9]}

    g3 = Automaton.with_actions(graph3, k3)
    return g3


def setupTest(test_no=0):
    if test_no == 0:
        graph = Graph(n=3, edges=[(0, 1), (1, 2), (2, 2)], directed=True)
        graph.es["weight"] = [1, 1, 1]
        k = {0: [0], 1: [1], 2: [2]}
        auto = Automaton.with_actions(graph, k)
        return auto
    else:
        graph = Graph(n=4, edges=[(0, 1), (1, 2), (1, 3), (2, 3), (3, 2), (3, 3)], directed=True)
        graph.es["weight"] = [1, 1, 1, 1, 1, 1]
        k = {0: [0], 1: [1, 2], 2: [3], 3: [4, 5]}
        auto = Automaton.with_actions(graph, k)
        return auto


def modifiedObligations(safe=True, verbose=False):
    g3 = setupAuto(safe)
    col_mission3 = g3.checkCTL("temp.smv", "EG !(name = 12)")
    print("T3: Collision mission (EG !collision) = ", str(col_mission3))

    g3.setCounter()
    exit_mission3 = g3.checkCTL("temp.smv", "E [TRUE U (name=9 & c<=4)]")
    print("T3: Exit mission (EF{<=4} safe_to_exit) = ", str(exit_mission3))

    hwy_mission3 = g3.checkCTL("temp.smv", "EF (name = 4 | name = 6)")
    print("T3: Highway mission (EF on_highway) = ", str(hwy_mission3))

    safe_obl_re = Obligation.fromCTL("G !(name = 12)")

    g3.q0 = 6
    has_safe3 = checkObligation(g3, safe_obl_re)
    print("T3: Safety obligation (O[a cstit: G !collision]) = ", str(has_safe3))

    fast_obl_re = Obligation.fromCTL(" [TRUE U (name=9 & c<=1)]")
    fast_obl_re.phi_neg = True

    g3.q0 = 6
    g3.setCounter()
    has_fast3 = not checkObligation(g3, fast_obl_re)
    print("T3: Fast obligation (!O[a cstit: !(True U reach_exit & c<=1)]) = ",
          str(has_fast3))

    ast_obl_re = Obligation(" [! (name=4 | name=6) U (name=9 | name=12)]",
                            False, False)
    ast_obl_re.phi_neg = True

    g3.q0 = 5
    has_ast3 = not checkObligation(g3, ast_obl_re, verbose=verbose)
    print("T3: Assertive obligation (!O[a cstit: [a dstit: !(!g U p)]]) = ",
          str(has_ast3))

    agg_obl_re = Obligation(" [! (name=4 | name=6) U (name=9 | name=12)]",
                            False, True)
    agg_obl_re.phi_neg = False

    g3.counter = False
    g3.q0 = 5
    has_agg3 = not checkObligation(g3, agg_obl_re, verbose=verbose)
    print("T3: Aggressive obligation (!O[a cstit: ![a dstit: (!g U p)]]) = ",
          str(has_agg3))

    if safe:
        assert col_mission3
        assert exit_mission3
        assert hwy_mission3
        assert has_safe3
        assert not has_fast3
        assert not has_ast3
        assert not has_agg3


def setupGridworld():
    return Automaton.as_gridworld(4, 3, cells=[("goal", [(3, 2)], 10, True, True),
                                               ("pit", [(3, 1)], -50, True, True),
                                               ("wall", [(1, 1)], -1, True, False)])


def setupGridworldSmaller():
    return Automaton.as_gridworld(3, 3, cells=[("goal", [(2, 2)], 10, True, True),
                                               ("pit", [(2, 1)], -10, True, True),
                                               ("wall", [(1, 1)], -1, True, False)])


def setupGridworldSmallerTwo():
    return Automaton.as_gridworld(3, 3, start=(0, 2), cells=[("goal", [(0, 0)], 10, True, True),
                                               ("pit", [(0, 1)], -10, True, True),
                                               ("wall", [(1, 1)], -1, True, False)])


def setupGridworldTwo():
    return Automaton.as_gridworld(5, 5, cells=[("goal", [(4, 2)], 10, True, True),
                                               ("fire", [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (1, 3), (2, 3), (3, 3),
                                                         (1, 2), (2, 2), (3, 2)], -5, False, True)])


def setupCliffworld(cliff_reward=-10):
    gw = Automaton.as_gridworld(5, 4, start=(0, 2),
                                cells=[("goal", [(4, 2)], 10, True, True),
                                       ("cliff", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], cliff_reward, True, True),
                                       ("wall", [(1, 2), (3, 2)], -1, True, False)])
    # combine cliff states
    cliffs = gw.graph.vs.select(type="cliff")
    first_cliff = gw.graph.vs.find(pos=(0, 0))
    e_to_cliff = gw.graph.es.select(_target_in=cliffs)
    for edge in e_to_cliff:
        source = edge.source
        attributes = edge.attributes()
        gw.graph.add_edges([(source, first_cliff)], attributes)
    gw.graph.delete_edges(e_to_cliff)
    bad_cliffs = cliffs.select(pos_ne=(0, 0))
    # gw.graph.delete_vertices(bad_cliffs)
    # make start state inaccessible
    start = gw.graph.vs.find(pos=(0, 2))
    e_to_start = gw.graph.es.select(_target=start, _source_ne=start.index)
    for edge in e_to_start:
        source = edge.source
        prob = edge["prob"]
        act = edge["action"]
        other_edge = gw.graph.es.find(_source=source, _target=source, action=act)
        other_edge["prob"] += prob
    gw.graph.delete_edges(e_to_start)
    return gw


def setupWindyDrone():
    gw = Automaton.as_gridworld(5, 4, start=(0, 2),
                                cells=[("goal", [(4, 3)], 10, True, True),
                                       ("wall", [(1, 1), (1, 2), (3, 1)], -1, True, False)])
    return gw


def setupHallTrap():
    gw = Automaton.as_gridworld(5, 4, start=(0, 0),
                                cells=[("trap", [(4, 3)], 10, False, True),
                                       ("gold", [(0, 3), (1, 3), (2, 3), (3, 3)], 10, False, True),
                                       ("wall", [(1, 2), (2, 2), (3, 2)], -1, True, False),
                                       ("goal", [(4, 2)], 5, True, True)],
                                state_property="type")
    return gw


def randomGridworld(x_len, y_len, pits, walls, coins, goals):
    """
    Create a gridworld of size x_len by y_len with pits number of spaces that give -10 points, walls number of spaces
    that can't be entered, coins number of spaces that give +5 points, and goals number of spaces that are absorbing and
    give +10 points.
    :param x_len:
    :param y_len:
    :param pits:
    :param walls:
    :param coins:
    :param goals:
    :return:
    """
    # TODO: how to map properties to states in a meaningful (or at least useful) way?
    # TODO: what constraints do we want on the problem generation?
    #  should the agent always start in the same place?
    #  should the goal always be in the same column?
    num_special_cells = pits + walls + coins + goals
    rng = np.random.default_rng()
    cells = np.arange(1, x_len*y_len)
    flat_idxs = rng.choice(cells, num_special_cells, replace=False)
    idxs = np.unravel_index(flat_idxs, (x_len, y_len))
    idxs = list(zip(idxs[0], idxs[1]))
    pit_cells = idxs[0:pits]
    wall_cells = idxs[pits:pits+walls]
    coin_cells = idxs[pits+walls:pits+walls+coins]
    goal_cells = idxs[pits+walls+coins:]
    gw = Automaton.as_gridworld(x_len, y_len, start=(0, 0),
                                cells=[("pit", pit_cells, -10, True, True),
                                       ("gold", coin_cells, 5, False, True),
                                       ("wall", wall_cells, -1, True, False),
                                       ("goal", goal_cells, 10, True, True)],
                                state_property="type")
    return gw


def test_fragments():
    gt = setupAuto()
    frags = generate_fragments(gt, gt, gt.q0, "EF (name=12)", t=7)
    assert len(frags) == 12

    gw = setupGridworld()
    frags = generate_fragments(gw, gw, gw.q0, "TRUE", t=1)
    assert len(frags) == 12


def test_gridworld():
    gw = setupGridworld()
    # plot(gw.graph)
    frags = generate_fragments(gw, gw, gw.q0, "EX (name = 0 | name = 1)", t=2)
    # assert len(frags) == 12
    print(len(frags))
    for frag in frags:
        # plot(frag.graph, layout=frag.graph.layout_fruchterman_reingold())
        pass
    tree = build_fragment_tree(frags, gw)
    # plot(tree.graph, layout=tree.graph.layout_fruchterman_reingold())
    actions = ["up", "down", "left", "right"]
    for v in tree.graph.vs:
        for action in actions:
            act_es = tree.graph.es.select(_source=v, action=action)
            act_prob = sum(act_es["prob"])
            np.testing.assert_allclose(act_prob, 1)
    print(tree.optimal(0.9))
    pass


if __name__ == "__main__":
    # originalObligations()
    # modifiedObligations(verbose=False)
    # enumeration()
    # test_fragments()
    # test_gridworld()
    # enum_gridworld()
    # explore_gridworld()
    pass
