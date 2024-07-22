"""
Created March 2023

@author: Colin
"""

import pickle
import chime
import time
import torch
import io
import matplotlib.pyplot as plt
import numpy as np
from igraph import Graph
from tqdm import tqdm
from model_check import Automaton, Obligation, checkObligation, checkStrategicObligation, getStrategicAutomaton

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def to_auto(dac_mdp):
    num_states = dac_mdp['S'].shape[0]
    quantiles = np.quantile(dac_mdp['S'], (0.2, 0.4, 0.6, 0.8), axis=0)
    num_actions = dac_mdp['Ti'].shape[1]
    ap_preps = ["xq", "dxq", "aq", "daq"]

    transitions = []
    actions = {a: [] for a in range(num_actions)}
    transition_probs = []
    transition_weights = []
    labels = []

    for s, state in tqdm(enumerate(dac_mdp['S'])):
        quant_belonging = state < torch.tensor(quantiles)
        quant_belonging = 4 - torch.sum(quant_belonging, 0)
        label = [ap_preps[q]+str(quant_belonging[q].item()) for q in range(len(ap_preps))]
        label = ', '.join(label)
        labels.append(label)

        for a in range(num_actions):
            prob_sum = 0
            probs = dac_mdp['Tp'][s, a]
            big_probs = probs > 1e-10
            not_need_check = big_probs.all()
            if not not_need_check:
                probs = probs * big_probs
                probs = probs/probs.sum()
            dac_mdp['Tp'][s, a] = probs

            for ns_index, next_state in enumerate(dac_mdp['Ti'][s, a]):
                prob = dac_mdp['Tp'][s, a, ns_index].item()
                if prob >= 0:
                    transitions.append((s, next_state.item()))
                    prob_sum += prob
                    transition_probs.append(prob)
                    transition_weights.append(dac_mdp['R'][s, a, ns_index].item())
                    actions[a].append(len(transitions)-1)
            # check to see that the sum of probabilities across this action sum to 1
            if prob_sum == 0:
                # it's an absorbing state, make a transition to itself with no reward
                transitions.append((s, s))
                transition_probs.append(1.0)
                transition_weights.append(0.0)
                actions[a].append(len(transitions)-1)


    graph = Graph.TupleList(transitions, directed=True)
    # graph.es["prob"] = transition_probs
    graph.es["weight"] = transition_weights
    graph.vs["label"] = labels

    auto = Automaton.with_actions(graph, actions, probs=transition_probs, labels=True)
    return auto


def solve_mdp(dac_mdp, softmax=False):
    Pi = None
    Q = None
    V = torch.rand(dac_mdp['Ti'].size()[0])
    epsilon = np.inf
    gamma = torch.tensor([0.95])
    P = torch.tensor(0)
    i_max = 10000
    while epsilon >= 0.05 and i_max > 0:
        if softmax:
            epsilon, Q, Pi, V = bellman_backup_operator_softmax(dac_mdp['Ti'], dac_mdp['Tp'], dac_mdp['R'], P, V, gamma)
        else:
            epsilon, Q, Pi, V = bellman_backup_operator(dac_mdp['Ti'], dac_mdp['Tp'], dac_mdp['R'], P, V, gamma)
        i_max -= 1
    return Q, Pi, V


def evaluate_mdp(dac_mdp, Pi):
    Q = None
    V = torch.rand(dac_mdp['Ti'].size()[0])
    epsilon = np.inf
    gamma = torch.tensor([0.95])
    P = torch.tensor(0)
    i_max = 100000
    while epsilon >= 0.05 and i_max > 0:
        epsilon, Q, Pi, V = evaluation_operator(dac_mdp['Ti'], dac_mdp['Tp'], dac_mdp['R'], P, V, gamma, Pi)
        i_max -= 1
    return V, Q


def calculate_expected_reward(Ti, Tp, R, Pi, gamma=0.95, epsilon=1e-6):
    """
    Function to calculate the expected reward for a given state using Value Iteration.
    Parameters:
        Ti: tensor of transition states [50000, 15, 5]
        Tp: tensor of transition probabilities [50000, 15, 5]
        R: tensor of rewards [50000, 15, 5]
        Pi: policy tensor [50000, 15]
        gamma: discount factor
        epsilon: threshold for convergence
    Returns:
        V: value function tensor [50000]
    """

    # Number of states
    n_states = Ti.shape[0]

    # Initialize the value function tensor
    V = torch.zeros(n_states)
    i = 0
    while True:
        V_old = V.clone()

        # Calculate the expected immediate reward and transition probabilities for each state-action pair
        expected_rewards = torch.einsum('ijk,ijk->ij', Tp, R)  # Tensor [50000, 15]
        transition_values = torch.einsum('ijk,ijk->ij', Tp, V[Ti])  # Tensor [50000, 15]

        # Compute Q values
        Q = expected_rewards + gamma * transition_values

        # Compute the expected value for each state given the policy
        V = torch.sum(Pi * Q, dim=-1)

        # Check for convergence
        if torch.max(torch.abs(V - V_old)) < epsilon or i >= 10000:
            break
        i += 1
    return V, Q



@torch.jit.script
def bellman_backup_operator(Ti, Tp, R, P, V, gamma):
    # R = R-P
    Q = torch.sum(torch.multiply(R, Tp), dim=2) + gamma[0]*torch.sum(torch.multiply(Tp, V[Ti]), dim=2)
    max_obj = torch.max(Q, dim=1)
    V_prime, Pi = max_obj.values, max_obj.indices
    epsilon = torch.max(V_prime - V)
    return epsilon, Q, Pi, V_prime


@torch.jit.script
def bellman_backup_operator_softmax(Ti, Tp, R, P, V, gamma):
    Q = torch.sum(torch.multiply(R, Tp), dim=2) + gamma[0] * torch.sum(torch.multiply(Tp, V[Ti]), dim=2)
    Pi = torch.functional.F.softmax(Q, dim=1)
    # Pi = Q+Q.min() / torch.sum(Q+Q.min(), dim=-1, keepdim=True)
    V_prime = torch.sum(torch.multiply(Q, Pi), dim=1)
    epsilon = torch.max(V_prime - V)
    return epsilon, Q, Pi, V_prime

@torch.jit.script
def evaluation_operator(Ti, Tp, R, P, V, gamma, Pi):
    Q = torch.sum(torch.multiply(R, Tp), dim=2) + gamma[0] * torch.sum(torch.multiply(Tp, V[Ti]), dim=2)
    V_prime = torch.sum(torch.multiply(Q, Pi), dim=1)
    epsilon = torch.max(V_prime - V)
    return epsilon, Q, Pi, V_prime


def time_model_checking(auto, obligation):
    # obligation = Obligation.fromPCTL('Pmax = ? [ (F "aq4") ]')
    # obligation = Obligation.fromPCTL('P <= 0.2 [ F "aq0" ]')
    # should the policies be part of checking the obligation, or how the Automaton gets optimal actions?
    # I think the latter - the Automaton will be handed a policy/q-function
    start = time.time()
    print("Start time:", start)
    print("MDP result:", checkObligation(auto, obligation))
    checkpoint = time.time()
    print("MDP time:", checkpoint - start)
    # print(checkStrategicObligation(auto_mdp, prob_dont_fall))
    print("MC result:", checkStrategicObligation(auto, obligation))
    finish = time.time()
    print("MC time:", finish - checkpoint)

    chime.theme('sonic')
    chime.success()


def load_dac_mdp(fname):
    with open(fname, 'rb') as f:
        dac_mdp = contents = CPU_Unpickler(f).load()
    if 'DACMDP' in dac_mdp:
        dac_mdp = dac_mdp['DACMDP']
    quality, policy, value = solve_mdp(dac_mdp)
    auto_mdp = to_auto(dac_mdp)
    auto_mdp.setPolicy(policy.numpy())
    return auto_mdp

if __name__ == "__main__":
    mdp_f = f"data/for_colin_cartpole_minimal_mdp_50k.pk"
    with open(mdp_f, 'rb') as f:
        raw_mdp = contents = CPU_Unpickler(f).load()
    quality, policy, value = solve_mdp(raw_mdp)
    auto_mdp = to_auto(raw_mdp)
    auto_mdp.setPolicy(policy.numpy())
    storm_mdp = auto_mdp.convertToStormMDP()
    storm_dtmc = getStrategicAutomaton(auto_mdp).convertToStormDTMC()

    time_model_checking(auto_mdp, Obligation.fromPCTL('Pmax = ? [ (F "aq4") ]'))
