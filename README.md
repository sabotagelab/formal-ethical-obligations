# autonomous-learning

Repository related to learning, modeling, and controlling behaviors exhibited by autonomous systems.

The core of this repository is model_check.py. This file contains the Automaton object that most code in this project revolves around. The Automaton object is essentially a wrapper for an igraph Graph object to allow us to treat it as an FSM, NFA, MDP, or PMC.

The random_weighted_automaton.py file provides functions to generate weighted directed graphs to cast at Automaton objects and to generate histories (executions, traces) from Automaton objects.

The complexity_experiments.py file has experiments for solving weights on Automaton objects.

The data_generation.py file contains functions for casting system models as NFAs, generating traces, and generating formulas.

The data_analysis.py file has functions for generating statistics about collections of LTL formulas.

The obenum.py file enumerates the obligations (in DAU.CTL) of a given Automaton object.

The bayes_opt.py file performs symbolic regression of a formula to meet an objective - based on "BOSS: Bayesian Optimization over String Spaces" by Moss, et al.

GrammarAcquisitionOptimizer.py extends BOSS's GrammarGeneticProgrammingOptimizer with validity checking to ensure that a generated string is satisfied by a given model.

The file label_learn.py attempts to take an Automaton object and a set of finite histories and identify what labelings for each state are consistent with the observed data.

The paired_comparisons.py file explores how to transform pairwise preferences over histories into weights for an Automaton.

The rl_utils.py file includes basic reinforcement learning functions over Automaton objects and gridworld environments.

The dac_mdp_parse.py file provides functions for parsing and solving DAC MDPs.

The policy_optimization.py file includes functions and experiments for modifying an MDP's policy to meet specified obligations.

The examples.py file has tests, experiments, examples, and demos of various functions in this project.

## requirements

Depending on what code you are interested in running, you may need some packages or executables that are not as simple to install as using `pip` or `conda`.
LTL, CTL, and CTL* model checking tasks are performed by (nuXmv)[https://es-static.fbk.eu/tools/nuxmv/]. Install nuXmv, and ensure that `nuXmv` is included in your path.
For some PCTL model checking tasks, the (PRISM)[http://www.prismmodelchecker.org/] model checking program is used. Install PRISM, and ensure that a path to the executable is set in the checkPCTL function of model_check.py.
For some PCTL model checking tasks, the (STORM)[https://www.stormchecker.org/] model checking program is used via the (stormpy)[https://moves-rwth.github.io/stormpy/index.html] bindings. Install STORM, and install stormpy bindings as instructed on their websites.
For LTL formula generation, (Spot)[https://spot.lre.epita.fr/index.html] is sometimes used. We suggest installing Spot as a conda package when possible.
Code based on "BOSS: Bayesian Optimization over String Spaces" requires their (package)[https://github.com/henrymoss/BOSS] to be accessible to python.
