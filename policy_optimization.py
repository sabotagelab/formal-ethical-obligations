"""
Created July 2023

@author: Colin
"""
import os
import sys
import re
import stormpy
import torch
import chime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import trange, tqdm
from joblib import Parallel, delayed
from pathlib import Path
from tabulate import tabulate
from gurobipy import GRB
from model_check import Obligation, checkStrategicObligation
from dac_mdp_parse import solve_mdp, load_dac_mdp, evaluate_mdp, calculate_expected_reward
from rl_utils import act, epsilon_greed, policy_from_q, safe_epsilon_greed

sys.path.extend(['/home/colin/Documents/GitHub/prmc-sensitivity'])
try:
    from core.classes import PMC
    from core.pmc_functions import pmc_load_instantiation, pmc_instantiate, pmc_derivative_LP, pmc_validate_derivative, \
        assert_probabilities
    from core.baseline_gradient import explicit_gradient
    from core.verify_pmc import pmc_verify, pmc_get_reward
    from core.io.export import export_json
    from core.io.parser import parse_main
except ImportError:
    pmc_imported = False


class DAC_PMC(PMC):
    def __init__(self, model, args, verbose=False):
        self.model_path = None
        self.verbose = verbose
        self.properties = stormpy.parse_properties(args.formula)
        self.model = model

        parameters_set = model.collect_probability_parameters()
        self.parameters = np.array(list(parameters_set))
        self.sI = {'s': np.array(self.model.initial_states),
                   'p': np.full(len(self.model.initial_states), 1/len(self.model.initial_states))}


# take a MDP
# take a property (and probability threshold)
# get the policy that maximizes the property
# take automaton
# transform into storm MDP
# run check on psi asking for max rho and scheduler
def get_property_maximizing_scheduler(auto, property_string, raw_scheduler=False):
    prop = stormpy.parse_properties(property_string)
    storm_mdp = auto.convertToStormMDP()
    result = stormpy.model_checking(storm_mdp, prop[0], only_initial_states=True, extract_scheduler=True)
    scheduler = result.scheduler
    probability = result.at(auto.q0)

    state_size = len(storm_mdp.states)
    if raw_scheduler:
        policy = scheduler
    else:
        policy = [scheduler.get_choice(state).get_deterministic_choice() for state in range(state_size)]
        policy = np.array(policy)
    return probability, policy, storm_mdp


# apply a storm scheduler to an automaton
# parse the storm scheduler into format for automaton
# call getStrategicAutomaton


# get a reward function that enforces a given policy
# implement inverse reinforcement learning for given policy

def infer_reward(mdp):
    pass


def get_factored_mdp(mdp, actions=15, outcomes=5):
    actions = len(set(mdp.graph.es['action']))
    action_list = list(set(mdp.graph.es['action']))
    action_list.sort()
    outcomes = mdp.max_action_outcomes()
    state_size = mdp.graph.vcount()
    Ti = torch.zeros((state_size, actions, outcomes), dtype=torch.int)
    Tp = torch.zeros((state_size, actions, outcomes))
    R = torch.zeros((state_size, actions, outcomes))
    # TODO: get these tensors by calling, e.g. mdp.graph.es['prob'] and shaping into 50000, 15, 5
    for v in mdp.graph.vs:
        vi = v.index
        for a, action in enumerate(action_list):
            transitions = mdp.graph.es.select(_source=vi, action=action)
            for i, e in enumerate(transitions):
                Ti[vi, a, i] = int(e.target)
                Tp[vi, a, i] = e['prob']
                R[vi, a, i] = e['weight']
    return {'Ti': Ti, 'Tp': Tp, 'R': R}


def parse_pmc(args):
    pmc = PMC(model_path=Path(args.root_dir, args.model), args=args)
    return pmc


def instantiate_pmc(pmc, params, theta):
    # TODO: make this work for in-memory PMC
    point = {}
    valuation = dict()
    flat_theta = theta.flatten()
    for i, x in enumerate(params):
        valuation[str(x)] = float(flat_theta[i].item())
        point[x] = stormpy.RationalRF(float(flat_theta[i].item()))
    inst = {'valuation': valuation, 'sample_size': None}
    instantiator = stormpy.pars.PDtmcInstantiator(pmc.model)
    inst_pmc = instantiator.instantiate(point)
    inst['point'] = point
    # assert_probabilities(inst_pmc)
    return inst_pmc, inst


def set_args(model_path, parameters_path, formula, storm_mdp, num_deriv=10, explicit_baseline=False, maximize=True):
    args = parse_main()
    args.goal_label = None
    args.mdp = storm_mdp
    args.parameters = parameters_path
    args.model = model_path
    args.formula = formula
    args.num_deriv = num_deriv
    args.explicit_baseline = explicit_baseline
    args.root_dir = os.path.dirname(os.path.abspath(__file__))
    args.validate_delta = 1e-10
    optimize = {True: GRB.MAXIMIZE, False: GRB.MINIMIZE}
    args.derivative_direction = GRB.MAXIMIZE
    return args


# get pMC for MDP w/ soft-max policy
# replace actions with soft-max probabilities

# compute the policy gradient (original reward) w.r.t. reward function
# get gradient of the policy by taking the gradient of the softmax function
# def calculate_performance_gradient(Tp, Pi, Q, V, states=50000, actions=15):
#     # Compute the advantage function
#     A = Q - V.unsqueeze(-1)
#
#     # Compute the policy gradient
#     grad = torch.einsum('ij,ijk->ijk', A, Pi.unsqueeze(-1) * (1 - Pi).unsqueeze(-1) * Tp)
#
#     grad_theta = torch.sum(grad, dim=2)
#     return grad_theta
def calculate_performance_gradient(Tp, Pi, Q):
    gradient = torch.zeros_like(Pi)
    for s, Tps in enumerate(Tp):
        for a, Tpsa in enumerate(Tps):
            log_grad = -Pi[s]
            log_grad[a] += 1  # - Pi[s, a]
            for s_p, Tpsas in enumerate(Tpsa):
                gradient[s] += Tpsas * Pi[s, a] * Q[s, a] * log_grad
    return gradient


def convert_scheduler(model, scheduler):
    prob_scheduler = np.zeros((model.graph.vcount(), len(model.k(0))))
    for v in model.graph.vs:
        prob_scheduler[v.index] = np.zeros(len(model.k(v.index)))
        action = scheduler.get_choice(v.index).get_deterministic_choice()
        prob_scheduler[v.index][action] = 1
    return torch.tensor(prob_scheduler)


def get_pmc_reward(pmc, instantiated_model, args):
    property = stormpy.parse_properties(args.formula)
    check = stormpy.model_checking(args.mdp, property[0], extract_scheduler=True)
    pmc.scheduler_raw = check.scheduler
    pmc.scheduler_prob = convert_scheduler(args.auto, pmc.scheduler_raw)
    reward = np.array(check.get_values(), dtype=float)
    return reward


def calculate_property_gradient(pmc, instantiated_model, inst, args, theta, params):
    # to silence the calls
    old_stdout = sys.stdout  # backup current stdout
    sys.stdout = open(os.devnull, "w")
    pmc.reward = pmc_get_reward(pmc, instantiated_model, args)
    # pmc.reward = get_pmc_reward(pmc, instantiated_model, args)
    pmc.reward = np.array([float(str(x.numerator))/float(str(x.denominator)) for x in pmc.model.reward_models['state_reward'].state_rewards])
    solution, J, Ju = pmc_verify(instantiated_model, pmc, inst['point'])
    optm, deriv = pmc_derivative_LP(pmc, J, Ju, args)
    # deriv = pmc_validate_derivative(pmc, inst, solution, deriv, args.validate_delta)
    grad_theta = torch.zeros_like(theta)
    param_names = [param.name for param in params]
    params_reshape = np.array(param_names).reshape(grad_theta.shape)
    for i, idx in enumerate(deriv['LP_idxs']):
        grad_s = deriv['LP'][i]
        name = pmc.parameters[idx].name
        index = np.where(params_reshape == name)
        grad_theta[index[0].item()][index[1].item()] = grad_s
    sys.stdout = old_stdout
    return grad_theta, solution


def evaluate_policy(mdp, policy, rewards):
    pass


def policy_from_theta(theta):
    pi = torch.zeros_like(theta)
    for i, row in enumerate(theta):
        denominator = torch.sum(row * row)
        for j, item in enumerate(row):
            pi[i][j] = (item.item() * item.item())/denominator
    return pi


def params_to_json(theta, fname, params):
    # TODO: I'm indexing parameters here differently than when I convert to DRN. Make sure this doesn't cause a conflict
    param_string = '{'
    i = 0
    for t_v in theta:
        for t_va in t_v:
            if i != 0:
                param_string += ', '
            param_string += '"' + str(params[i]) + '": ' + str(t_va.item())
            i += 1
    param_string += '}'
    with open(fname, 'w') as f:
        f.write(param_string)


def format_property(prop, maximize_prop=True):
    if maximize_prop:
        return 'Pmax = ? [ ' + prop + ' ]'
    else:
        return 'Pmin = ? [ ' + prop + ' ]'


def optimize_policy(mdp, prop, maximize_prop=True, threshold=1.0, alpha=0.05, pmc=None, perf_pmc=None, grad_num=None):
    # alpha = 0.9
    init_state = mdp.q0
    objective_prop = format_property(prop)
    # get the original reward
    # original_reward = deepcopy(mdp.graph.es['weight'])

    # get the policy that optimizes the property
    optimum_probability, property_policy, storm_mdp = get_property_maximizing_scheduler(mdp, objective_prop)
    # if optimum_probability is worse than threshold then get angry
    if not (optimum_probability >= threshold) == maximize_prop:
        # TODO: throw an exception instead? maybe not - just optimize, ignore threshold.
        print("Threshold can not be passed. Continuing to maximize probability.")
        # return

    model_file = f"model_files/test.drn"
    perf_file = f"model_files/perf_test.drn"
    param_file = f"model_files/test.json"
    # get the soft-max policy for the current reward
    factored_mdp = get_factored_mdp(mdp)
    quality, policy, value = solve_mdp(factored_mdp, softmax=True)
    # get the PMC args
    args = set_args(model_file, param_file, objective_prop, storm_mdp, num_deriv=grad_num, explicit_baseline=True)
    perf_args = set_args(perf_file, param_file, objective_prop, storm_mdp, num_deriv=grad_num, explicit_baseline=True)
    # args = set_args(None, None, objective_prop, storm_mdp, num_deriv=grad_num, explicit_baseline=True)
    # perf_args = set_args(None, None, objective_prop, storm_mdp, num_deriv=grad_num, explicit_baseline=True)
    # setup model and params
    # params = auto_mdp.convertToDRN(model_file, spec=objective_prop)
    # params = auto_mdp.convertToDRN(perf_file, reward=value)
    _, params = mdp.convertToStormPDTMC(reward=True)
    vals = []
    probs = []
    prop_mags = []
    perf_mags = []
    prop_grads = []
    perf_grads = []

    mdp.setPolicy(policy)
    args.auto = mdp
    theta = policy
    # params_to_json(theta, param_file, params)
    # get PMC and instantiate
    if not pmc:
        # pmc = parse_pmc(args)
        pmc, params = mdp.convertToStormPDTMC(spec=objective_prop)
        pmc = DAC_PMC(pmc, args)
        # pmc.parameters = params
    if not perf_pmc:
        perf_pmc, perf_params = mdp.convertToStormPDTMC(reward=True)
        perf_pmc = DAC_PMC(perf_pmc, perf_args)
        # perf_pmc.parameters = params
    chime.info()
    if not grad_num:
        # perf_args.num_deriv = args.num_deriv = len(pmc.parameters)
        # args.num_deriv = len(pmc.parameters)//10
        args.num_deriv = int(np.sqrt(len(pmc.parameters)))
    else:
        args.num_deriv = grad_num
    instantiated_model, inst = instantiate_pmc(pmc, params, theta)
    property = stormpy.parse_properties(objective_prop)
    check = stormpy.model_checking(instantiated_model, property[0])
    optimum_value = value[init_state]
    print("Original V[start]: ", value[init_state])
    print("Original P[start]: ", check.at(init_state))
    vals.append(value[init_state])
    probs.append(check.at(init_state))

    # current_policy = policy
    # theta = quality
    theta = torch.functional.F.softmax(quality, dim=1)
    # theta = torch.rand_like(quality)
    # params_to_json(theta, param_file, params)
    current_policy = torch.functional.F.softmax(theta, dim=1)
    # current_policy = torch.abs(theta) / torch.sum(torch.abs(theta), dim=-1, keepdim=True)
    # current_policy = theta.clamp(min=0) / torch.sum(theta.clamp(min=0), dim=-1, keepdim=True)
    mdp.setPolicy(current_policy)
    instantiated_model, inst = instantiate_pmc(pmc, params, theta)
    perf_instantiated_model, perf_inst = instantiate_pmc(perf_pmc, perf_params, theta)
    check = stormpy.model_checking(instantiated_model, property[0])
    prob = check.at(init_state)
    value, quality = evaluate_mdp(factored_mdp, current_policy)
    print("Random V[start]: ", value[init_state])
    print("Random P[start]: ", check.at(init_state))
    vals.append(value[init_state])
    probs.append(check.at(init_state))
    property_magnitude = np.inf
    toggle_ascend = True
    # get the policy gradient w.r.t. performance on original reward
    # is there a better way to use the gradients? Maybe it *is* better to use only some? Choose which randomly?

    for _ in trange(1000):
        # performance_grad = calculate_performance_gradient(current_policy, quality)
        performance_grad, perf_solution = calculate_property_gradient(perf_pmc, perf_instantiated_model, perf_inst,
                                                                      perf_args, theta, perf_params)
        performance_magnitude = performance_grad.norm(p='fro')
        perf_mags.append(performance_magnitude)
        perf_grads.append(performance_grad)
        rand_num = torch.rand(1) * 2 - 1
        # if False:
        # if True:
        if toggle_ascend and ((prob > threshold and maximize_prop) or
                              (prob < threshold and not maximize_prop)):
            theta += alpha * performance_grad
            if property_magnitude < 1e-5 or True:
                property_grad, _ = calculate_property_gradient(pmc, instantiated_model, inst, args, theta, params)
                property_magnitude = property_grad.norm(p='fro')
                prop_mags.append(property_magnitude)
                prop_grads.append(property_grad)
        elif toggle_ascend:
            # the property is not sufficed, add property gradient to update
            property_grad, _ = calculate_property_gradient(pmc, instantiated_model, inst, args, theta, params)
            property_magnitude = property_grad.norm(p='fro')
            prop_mags.append(property_magnitude)
            prop_grads.append(property_grad)
            noise = rand_num * property_magnitude
            # theta += alpha * (performance_grad/performance_magnitude + property_grad/property_magnitude)/2
            # theta += alpha * (performance_grad + property_grad) / 2
            # theta += alpha * (property_grad + noise)
            theta += alpha * property_grad
            # theta += alpha * performance_grad
        else:
            property_grad, _ = calculate_property_gradient(pmc, instantiated_model, inst, args, theta, params)
            property_magnitude = property_grad.norm(p='fro')
            prop_mags.append(property_magnitude)
            prop_grads.append(property_grad)
            # theta += alpha * (property_grad/property_magnitude + performance_grad/performance_magnitude) / 2
            theta += alpha * (performance_grad + property_grad) / 2
        # theta += alpha * property_grad / magnitude + noise
        # params_to_json(theta, param_file, params)
        # print(theta[init_state])
        # current_policy = torch.functional.F.softmax(theta, dim=-1)
        current_policy = torch.abs(theta) / torch.sum(torch.abs(theta), dim=-1, keepdim=True)
        # current_policy = theta.clamp(min=0) / torch.sum(theta.clamp(min=0), dim=-1, keepdim=True)
        mdp.setPolicy(current_policy)
        instantiated_model, inst = instantiate_pmc(pmc, params, theta)
        perf_instantiated_model, perf_inst = instantiate_pmc(perf_pmc, perf_params, theta)
        # get new Q values
        value, quality = calculate_expected_reward(factored_mdp['Ti'], factored_mdp['Tp'], factored_mdp['R'],
                                                   current_policy)
        check = stormpy.model_checking(instantiated_model, property[0])
        prob = check.at(init_state)
        # print(value[init_state].item(), check.at(init_state))
        vals.append(value[init_state])
        probs.append(check.at(init_state))
        # print(quality[init_state])
        # print(performance_grad[init_state])
        # print(current_policy[init_state])
        if property_magnitude < 1e-5 and performance_magnitude < 1e-5:
            break
    vals = [val.item() for val in vals]
    # print('optimum property satisfaction: ', optimum_probability)
    out_dict = {'pmc': pmc, 'policy': current_policy, 'vals': vals, 'probs': probs, 'prop_mags': prop_mags,
                'perf_mags': perf_mags, 'prop_grads': prop_grads, 'perf_grads': perf_grads,
                'opt_val': optimum_value.item(), 'opt_prob': optimum_probability}
    return out_dict


def policy_interpolation(mdp, prop, maximize_prop=True, threshold=1.0, alpha=0.01, grad_num=None):
    vals = []
    probs = []

    init_state = mdp.q0

    samples = int(1//alpha)

    objective_prop = format_property(prop)

    # get the policy that optimizes the property
    optimum_probability, property_policy, storm_mdp = get_property_maximizing_scheduler(mdp, objective_prop,
                                                                                        raw_scheduler=True)
    Q = convert_scheduler(mdp, property_policy).numpy()
    probability_policy = policy_from_q(Q, temperature=0.1, tensor=True)
    best_policy = probability_policy
    # if optimum_probability is worse than threshold then get angry
    if not (optimum_probability >= threshold) == maximize_prop:
        # TODO: throw an exception instead? maybe not - just optimize, ignore threshold.
        print("Threshold can not be passed. Continuing to maximize probability.")

    factored_mdp = get_factored_mdp(mdp)
    true_quality, optimal_policy, true_value = solve_mdp(factored_mdp, softmax=True)
    utility_policy = optimal_policy
    points = np.linspace(start=0.0, stop=1.0, num=samples)

    # set up pMC, so I can put the policies in it and check the probabilities and utility
    model_file = f"model_files/test.drn"
    param_file = f"model_files/test.json"
    args = set_args(model_file, param_file, objective_prop, storm_mdp, num_deriv=grad_num, explicit_baseline=True)
    args.auto = mdp
    pmc, params = mdp.convertToStormPDTMC(spec=objective_prop)
    pmc = DAC_PMC(pmc, args)
    if not grad_num:
        args.num_deriv = len(pmc.parameters)
    else:
        args.num_deriv = grad_num
    instantiated_model, inst = instantiate_pmc(pmc, params, best_policy)
    property = stormpy.parse_properties(objective_prop)
    check = stormpy.model_checking(instantiated_model, property[0])

    # get the value and prob for starting policy
    start_val, _ = calculate_expected_reward(factored_mdp['Ti'], factored_mdp['Tp'], factored_mdp['R'], best_policy)
    start_prob = check.at(init_state)
    best_val = start_val[init_state].item()

    for a in points:
        pi_a = (1 - a) * probability_policy + a * utility_policy
        # get the value and prob for pi_a and put them in vals and probs
        instantiated_model, inst = instantiate_pmc(pmc, params, pi_a)
        check = stormpy.model_checking(instantiated_model, property[0])
        val, _ = calculate_expected_reward(factored_mdp['Ti'], factored_mdp['Tp'], factored_mdp['R'], pi_a)
        val = val[init_state].item()
        prob = check.at(init_state)
        vals.append(val)
        probs.append(prob)
        if prob >= threshold and val > best_val:
            best_val = val
            best_policy = pi_a

    out_dict = {'policy': best_policy, 'vals': vals, 'probs': probs, 'opt_val': true_value[init_state].item(),
                'opt_prob': optimum_probability, 'best_val': best_val}
    return out_dict


def safe_q_learning(mdp, prop, maximize_prop=True, threshold=1.0, grad_alpha=1.0, pmc=None, grad_num=None, eps=0.65,
                    q_alpha=0.5, gamma=0.95, episodes=100, steps=100, update_every=1, grad_q=True, shield=False):
    vals = []
    probs = []

    toggle = True
    projected_q = False
    apx_utility_grad = True

    state_space = len(mdp.graph.vs)
    actions = list(set(mdp.graph.es['action']))
    init_state = mdp.q0
    actions.sort()
    # Q = np.full((state_space, len(actions)), 100)
    experiences = []
    objective_prop = format_property(prop)

    # get the policy that optimizes the property
    optimum_probability, property_policy, storm_mdp = get_property_maximizing_scheduler(mdp, objective_prop,
                                                                                        raw_scheduler=True)
    Q = convert_scheduler(mdp, property_policy).numpy()
    # if optimum_probability is worse than threshold then get angry
    if not (optimum_probability >= threshold) == maximize_prop:
        # TODO: throw an exception instead? maybe not - just optimize, ignore threshold.
        print("Threshold can not be passed. Continuing to maximize probability.")

    model_file = f"model_files/test.drn"
    param_file = f"model_files/test.json"
    args = set_args(model_file, param_file, objective_prop, storm_mdp, num_deriv=grad_num, explicit_baseline=True)
    # get the soft-max policy for the current reward
    factored_mdp = get_factored_mdp(mdp)
    true_quality, optimal_policy, true_value = solve_mdp(factored_mdp, softmax=True)
    _, params = mdp.convertToStormPDTMC(reward=True)

    # get current policy from current Q values
    # policy = policy_from_q(Q, tensor=True, temperature=0.1)
    policy = optimal_policy
    policy = torch.rand_like(policy)
    mdp.setPolicy(policy)
    args.auto = mdp
    theta = policy

    rewards = [20] * mdp.graph.vcount()
    # get PMC and instantiate
    if not pmc:
        pmc, params = mdp.convertToStormPDTMC(spec=objective_prop)
        pmc = DAC_PMC(pmc, args)
    if apx_utility_grad:
        util_args = set_args(f"model_files/perf_test.drn", param_file, objective_prop, storm_mdp,
                             num_deriv=len(pmc.parameters), explicit_baseline=True)
        util_pmc, util_params, transition_matrix, state_labeling, reward_models = mdp.convertToStormPDTMC(reward=rewards, return_components=True)
        util_pmc = DAC_PMC(util_pmc, args)
        util_instantiated_model, util_inst = instantiate_pmc(util_pmc, util_params, theta)
    if not grad_num:
        # args.num_deriv = int(np.sqrt(len(pmc.parameters)))
        args.num_deriv = len(pmc.parameters)
    else:
        args.num_deriv = grad_num
    # debuging...
    args.num_deriv = len(pmc.parameters)

    instantiated_model, inst = instantiate_pmc(pmc, params, theta)
    property = stormpy.parse_properties(objective_prop)
    check = stormpy.model_checking(instantiated_model, property[0])
    print("Optimum V[0]: ", true_value[init_state].item())
    print("Original P[0]: ", check.at(init_state))
    prob = check.at(init_state)

    for i in trange(episodes):
        current_state = mdp.q0
        t = 0
        finished = False
        while t < steps and not finished:
            if mdp.graph.vs[current_state]['absorbing']:
                finished = True
            if shield:
                action = safe_epsilon_greed(mdp.graph, current_state, Q, eps, check, threshold, maximize_prop)
            else:
                action = epsilon_greed(mdp.graph, current_state, Q, eps)
            reward, next_state = act(mdp.graph, current_state, action)
            rewards[current_state] = reward
            experiences.append((current_state, action, reward, next_state))
            current_state = next_state
            t += 1

        # Batch update after 'update_every' episodes
        # TODO: should I include a learning rate schedule? In the alternating case in particular it might be nice.
        if (i + 1) % update_every == 0 or i == episodes - 1:
            new_Q = deepcopy(Q)
            if apx_utility_grad:
                # recreate the PMC with the new reward values.
                # util_pmc, util_params = mdp.convertToStormPDTMC(reward=rewards)
                util_pmc = mdp.constructStormPDTMCfromComponents(rewards, transition_matrix, state_labeling)
                util_pmc = DAC_PMC(util_pmc, args)
                util_instantiated_model, util_inst = instantiate_pmc(util_pmc, util_params, theta)
                # get the utility grad
                util_grad, _ = calculate_property_gradient(util_pmc, util_instantiated_model,
                                                                              util_inst, util_args, theta, util_params)
            for experience in experiences:
                state, action, reward, next_state = experience
                new_Q[state][action] = Q[state][action] + q_alpha * (
                        reward + gamma * np.max(Q[next_state]) - Q[state][action])
            if projected_q:
                new_Q = projected_q_update(new_Q, Q, threshold, q_alpha, init_state, property, factored_mdp, pmc,
                                           params, maximize_prop)
            # calculate the property gradient
            elif check_safe_update(grad_q, toggle, maximize_prop, prob, threshold):
                # if threshold is not met, then follow property_grad
                # else follow Q update
                property_grad, _ = calculate_property_gradient(pmc, instantiated_model, inst, args, theta, params)
                # new_Q = grad_q_update(property_grad, new_Q, Q)
                new_Q = Q + property_grad.numpy()
                if apx_utility_grad:
                    theta += grad_alpha * property_grad
            elif apx_utility_grad:
                theta += grad_alpha * util_grad
            # clear experiences after updating
            experiences = []
            # update Q table
            Q = new_Q
            if not apx_utility_grad:
                policy = policy_from_q(Q, tensor=True)
                theta = policy
            else:
                policy = policy_from_theta(theta)
            mdp.setPolicy(policy)
            instantiated_model, inst = instantiate_pmc(pmc, params, theta)
            # evaluate updated policy for performance and conformance
            value, _ = calculate_expected_reward(factored_mdp['Ti'], factored_mdp['Tp'], factored_mdp['R'],
                                                 policy)
            vals.append(value[init_state].item())
            check = stormpy.model_checking(instantiated_model, property[0])
            prob = check.at(init_state)
            probs.append(prob)

    out_dict = {'pmc': pmc, 'policy': policy, 'vals': vals, 'probs': probs, 'opt_val': true_value[init_state].item(),
                'opt_prob': optimum_probability}
    return out_dict


def check_safe_update(grad_q, toggle, maximize_prop, prob, threshold):
    if grad_q:
        if not toggle:
            return True
        elif maximize_prop and prob < threshold:
            return True
        elif not maximize_prop and prob > threshold:
            return True
        else:
            return False
    else:
        return False


def grad_q_update(grad, new_Q, old_Q):
    del_policy = policy_from_q(new_Q, tensor=True) - policy_from_q(old_Q, tensor=True)
    # find where the sign of del_policy disagrees with the sign of grad
    policy_signs = torch.sign(del_policy)
    grad_signs = torch.sign(grad)
    indeces = torch.argwhere(policy_signs != grad_signs)
    # where there is a disagreement, keep old_Q
    new_Q[indeces[:, 0], indeces[:, 1]] = old_Q[indeces[:, 0], indeces[:, 1]]
    return new_Q


def projected_q_update(new_Q, safe_Q, threshold, alpha, init_state, property, factored_mdp, pmc, params,
                       maximize_prob=True):
    safe_policy = policy_from_q(safe_Q, tensor=True)
    new_policy = policy_from_q(new_Q, tensor=True)
    safe_value, _ = calculate_expected_reward(factored_mdp['Ti'], factored_mdp['Tp'], factored_mdp['R'],
                                              safe_policy)
    safe_value = safe_value[init_state].item()
    new_value, _ = calculate_expected_reward(factored_mdp['Ti'], factored_mdp['Tp'], factored_mdp['R'],
                                             new_policy)
    new_value = new_value[init_state].item()
    safe_instantiated_model, _ = instantiate_pmc(pmc, params, safe_policy)
    new_instantiated_model, _ = instantiate_pmc(pmc, params, new_policy)
    safe_check = stormpy.model_checking(safe_instantiated_model, property[0])
    safe_prob = safe_check.at(init_state)
    new_check = stormpy.model_checking(new_instantiated_model, property[0])
    new_prob = new_check.at(init_state)
    limit = 44
    counter = 0
    while check_safe_update(True, True, maximize_prob, new_prob, threshold):
        # TODO: or new_value < safe_value
        new_Q = new_Q + alpha * (safe_Q - new_Q)
        new_policy = policy_from_q(new_Q, tensor=True)
        new_value, _ = calculate_expected_reward(factored_mdp['Ti'], factored_mdp['Tp'], factored_mdp['R'],
                                                 new_policy)
        new_value = new_value[init_state].item()
        new_instantiated_model, _ = instantiate_pmc(pmc, params, new_policy)
        new_check = stormpy.model_checking(new_instantiated_model, property[0])
        new_prob = new_check.at(init_state)
        counter += 1
        if counter > limit:
            new_Q = safe_Q
            break
    return new_Q


def double_plot(vals, probs, optimumval, optimumprob):
    # Create a new figure and a subplot
    fig, ax1 = plt.subplots()

    # Plot vals on ax1
    ax1.plot(vals, color='tab:red')
    # ax1.plot([optimumval]*len(vals), color='tab:blue', linestyle='dashed', label='Optimum Performance')
    ax1.set_ylabel('Expected Utility', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xlabel('Updates')
    # ax1.legend()

    # Create a second y-axis for the same plot
    ax2 = ax1.twinx()

    # Plot probs on ax2
    ax2.plot(probs, color='tab:blue')
    # ax2.plot([optimumprob]*len(probs), color='tab:red', linestyle='dashed', label='Optimum Conformance')
    ax2.set_ylabel('Probability of Satisfaction', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    # ax2.legend()
    plt.show()


def mat_plot(matrix, alpha_values, derivatives):
    fig, ax = plt.subplots()

    # Displaying the matrix
    cax = ax.imshow(matrix, cmap='viridis')

    # Labelling the axes
    ax.set_xlabel("number of derivatives")
    ax.set_xticks(np.arange(len(derivatives)))
    ax.set_xticklabels([int(x) for x in derivatives])

    ax.set_ylabel("alpha")
    ax.set_yticks(np.arange(len(alpha_values)))
    ax.set_yticklabels(['{:.3f}'.format(x) for x in alpha_values])

    # Adding a colorbar
    fig.colorbar(cax)

    # Annotating the values of the matrix
    for i in range(len(alpha_values)):
        for j in range(len(derivatives)):
            text = ax.text(j, i, round(matrix[i, j], 2),
                           ha="center", va="center", color="black", fontsize='small')

    plt.savefig('matrix.png')
    plt.show()


def sub_plots(three_d_data):
    x = len(three_d_data)
    y = len(three_d_data[0])
    fig, axs = plt.subplots(x, y, figsize=(3*x, 3*y))
    plt.tight_layout(pad=3.0)
    for i, two_d in enumerate(three_d_data):
        for j, one_d in enumerate(two_d):
            axs[i, j].plot(one_d)
    plt.show()


def hyper_param_experiment(auto_mdp):
    chime.theme('sonic')
    from examples import setupCliffworld
    formula = 'G ! ( "x2" & "y3" )'
    n = 5
    val_mat = np.zeros((n, n))
    prob_mat = np.zeros((n, n))
    lr_diffs = []
    for i, a in tqdm(enumerate(np.logspace(-3, 0, n)), total=n):
        gn_diffs = []
        for j, g in enumerate(np.linspace(1, 64, int(n))):
            out = optimize_policy(auto_mdp, formula,
                                  threshold=0.75,
                                  maximize_prop=True,
                                  alpha=a, grad_num=int(g))
            vals = out['vals']
            probs = out['probs']
            val_mat[i, j] = vals[-1]
            prob_mat[i, j] = probs[-1]
            grad_diffs = [out['prop_grads'][k] - out['perf_grads'][k] for k in range(1000)]
            grad_diffs = [grad_diffs[m].norm(p='fro') for m in range(1000)]
            gn_diffs.append(grad_diffs)
        lr_diffs.append(gn_diffs)

    chime.success()
    return lr_diffs, val_mat, prob_mat


def random_mdp_experiment(grad_q=True, shield=True, n=4):
    from examples import randomGridworld
    chime.theme('sonic')
    formula = 'G ! ( "gold" )'
    # n = 12
    x_len = 12
    y_len = 12
    vals_list = []
    probs_list = []
    lists = Parallel(n_jobs=3)(delayed(random_mdp_function)(grad_q, shield, x_len, y_len, formula) for _ in trange(n))
    vals_list, probs_list = zip(*lists)
    return vals_list, probs_list


def random_mdp_function(grad_q, shield, x_len, y_len, formula):
    threshold = 0.75
    optimum_probability = 0
    # check mdp if it meets the threshold
    while optimum_probability < threshold:
        mdp = randomGridworld(x_len, y_len, 10, 10, 10, 1)
        objective_prop = format_property(formula)
        optimum_probability, _, _ = get_property_maximizing_scheduler(mdp, objective_prop)
        print("Optimum Probability: ", optimum_probability)

    output = safe_q_learning(mdp, formula, threshold=0.75, grad_alpha=0.01, grad_num=None, eps=0.05, q_alpha=0.05,
                             episodes=5000, steps=100, update_every=5, grad_q=grad_q, shield=shield)
    return output['vals'], output['probs']


def data_to_dataframe(runs, epochs, probs_list, vals_list):
    d = {"Epoch": [], "Run": [], "Probability of Satisfaction": [], "Expected Utility": []}
    for epoch in range(epochs):
        for run in range(runs):
            d["Epoch"].append(epoch)
            d["Run"].append(run)
            d["Probability of Satisfaction"].append(probs_list[run][epoch])
            d["Expected Utility"].append(vals_list[run][epoch])
    return pd.DataFrame(data=d)


if __name__ == "__main__":
    chime.theme('sonic')
    # auto_mdp = load_dac_mdp(f"data/for_colin_cartpole_minimal_mdp_50k.pk")
    # auto_mdp = load_dac_mdp(f"/home/colin/Documents/GitHub/autonomous-learning/data/DACMDP/CartPole_Deontic_DAC_bs-500_ttyp-5_ttar-3.pk")
    from examples import setupCliffworld, setupGridworldSmaller, setupWindyDrone, setupHallTrap, randomGridworld

    # auto_mdp = setupCliffworld()
    # auto_mdp = setupGridworldSmaller()
    auto_mdp = setupWindyDrone()
    # auto_mdp = setupHallTrap()
    # auto_mdp = randomGridworld(12, 12, 10, 10, 10, 1)
    formula = 'G ! ( "x2" & "y3" )'
    # formula = 'G ! ( "trap" )'
    # formula = 'G ! ( "gold" )'
    # formula = 'G ! ( "pit" )'
    # formula = 'F ( "x0" & "y1" )'
    # grad = optimize_policy(auto_mdp, 'F "aq2"')
    # formula = 'F ( "aq0" | "aq4" )'
    # formula = 'G ! ( "x2" & "y2" )'
    # formula = 'F ( "x2" & "y2" )'
    # formula = 'F ( "x4" & "y3" )'
    # pmc, params = auto_mdp.convertToStormPDTMC(spec=format_property(formula))

    # diffs_data, val_data, prob_data = hyper_param_experiment(auto_mdp)
    # n = 5
    # sub_plots(diffs_data)
    # alphas = np.logspace(-3, 0, n)
    # n_grad = np.linspace(1, 64, n)
    # mat_plot(val_data, alphas, n_grad)
    # mat_plot(prob_data, alphas, n_grad)

    # output = optimize_policy(auto_mdp, formula, threshold=0.9, maximize_prop=True, alpha=1.0, grad_num=int(64))
    output = safe_q_learning(auto_mdp, formula, threshold=0.75, grad_alpha=0.01, grad_num=None, eps=0.25, q_alpha=0.1,
                             episodes=1000, update_every=2, grad_q=True, shield=False)
    # output = policy_interpolation(auto_mdp, formula, threshold=0.75)
    vals = output['vals']
    probs = output['probs']
    opt_val = output['opt_val']
    opt_prob = output['opt_prob']
    double_plot(vals, probs, opt_val, opt_prob)
    # print(vals, probs)
    # grad_diffs = [output['prop_grads'][i] - output['perf_grads'][i] for i in range(1000)]
    # grad_diffs = [grad_diffs[i].norm(p='fro') for i in range(1000)]
    # n = 24
    # vals, probs = random_mdp_experiment(False, False, n)
    # data = data_to_dataframe(n, 1000, probs, vals)
    # import seaborn as sns
    # sns.relplot(data=data, x="Epoch", y="Probability of Satisfaction", kind="line", errorbar=("ci", 80))
    # sns.relplot(data=data, x="Epoch", y="Expected Utility", kind="line", errorbar=("ci", 80))

    chime.success()
