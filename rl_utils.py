"""
Created June 2023

@author: Colin
"""

import pickle
import random
import torch
import numpy as np
from igraph import Graph


def epsilon_greed(graph, state, Q, eps):
    actions = list(set(graph.es['action']))
    actions.sort()
    if np.random.uniform(0, 1) < eps:
        action = np.random.randint(0, len(actions))
    else:
        action = np.argmax(Q[state, :])
    return action


def safe_epsilon_greed(graph, state, Q, eps, check_result, threshold, maximize_prop):
    # TODO: this function does not consider that future states may have unsafe actions as well, or that future states
    #  might act randomly and not greedily. Probabilities are based only on the transition probability and the
    #  likelihood of satisfaction using the given policy.
    actions = list(set(graph.es['action']))
    actions.sort()
    action_sat_probs = []
    no_safe_actions = True
    safe_actions = []
    for i, action in enumerate(actions):
        out_edges_of_action = graph.es.select(_source=state, action=action)
        action_sat_prob = 0
        for out_edge in out_edges_of_action:
            action_sat_prob += out_edge['prob'] * check_result.at(out_edge.target)
        if (action_sat_prob > threshold and maximize_prop) or (action_sat_prob < threshold and not maximize_prop):
            no_safe_actions = False
            safe_actions.append(i)
        action_sat_probs.append(action_sat_prob)
    if no_safe_actions:
        action = np.argmax(action_sat_probs)
    elif np.random.uniform(0, 1) < eps:
        action = random.choice(safe_actions)
    else:
        action = np.argmax(Q[state, safe_actions])
    return action


def act(graph, state, action):
    actions = list(set(graph.es['action']))
    actions.sort()
    reward = graph.vs[state]['reward']
    action_id = actions[action]
    out_edges_of_action = graph.es.select(_source=state, action=action_id)
    transition_probabilities = out_edges_of_action['prob']
    transition_taken = random.choices(out_edges_of_action,
                                      weights=transition_probabilities, k=1)[0]
    next_state = transition_taken.target
    return reward, next_state


def q_learning(graph, eps=0.65, alpha=0.5, gamma=0.95, episodes=100, steps=100):
    state_space = len(graph.vs)
    actions = list(set(graph.es['action']))
    actions.sort()
    Q = np.zeros((state_space, len(actions)))

    for i in range(episodes):
        current_state = 0
        t = 0
        finished = False
        while t < steps and not finished:
            if graph.vs[current_state]['absorbing']:
                finished = True
            action = epsilon_greed(graph, current_state, Q, eps)
            reward, next_state = act(graph, current_state, action)
            Q[current_state][action] = Q[current_state][action] + alpha * (
                    reward + gamma * np.max(
                        Q[next_state]) - Q[current_state][action])
            current_state = next_state
            t += 1

    return Q, actions


def policy_from_q(Q, softmax=True, tensor=False, temperature=1):
    if softmax:
        policy = np.exp(Q/temperature)/np.sum(np.exp(Q/temperature), axis=1)[:, None]
    else:
        policy = np.argmax(Q, axis=1)

    if not tensor:
        return policy
    else:
        return torch.from_numpy(policy).float()



def q_learning_lambda(graph, eps=0.3, alpha=0.1, gamma=0.95, lambd=0.9, episodes=100, steps=100):
    state_space = len(graph.vs)
    actions = list(set(graph.es['action']))
    actions.sort()
    Q = np.zeros((state_space, len(actions)))
    E = np.zeros((state_space, len(actions)))
    for i in range(episodes):
        current_state = 0
        t = 0
        finished = False
        action = epsilon_greed(graph, current_state, Q, eps)
        while t < steps and not finished:
            if graph.vs[current_state]['absorbing']:
                finished = True
            reward, next_state = act(graph, current_state, action)
            next_action = epsilon_greed(graph, next_state, Q, eps)
            action_star = np.argmax(Q[next_state, :])

            delta = reward + gamma * Q[next_state, action_star] - Q[current_state][action]
            E[current_state][action] += 1

            for s in range(state_space):
                for a in range(len(actions)):
                    Q[s][a] += alpha * delta * E[s][a]
                    if next_action == action_star:
                        E[s][a] *= gamma * lambd
                    else:
                        E[s][a] = 0

            current_state = next_state
            action = next_action
            t += 1

    return Q, actions


def sarsa_learning(graph, eps=0.65, alpha=0.5, gamma=0.95, episodes=100, steps=100):
    state_space = len(graph.vs)
    actions = list(set(graph.es['action']))
    actions.sort()
    Q = np.zeros((state_space, len(actions)))

    for i in range(episodes):
        current_state = 0
        action = epsilon_greed(graph, current_state, Q, eps)
        t = 0
        finished = False
        while t < steps and not finished:
            if graph.vs[current_state]['absorbing']:
                finished = True
            reward, next_state = act(graph, current_state, action)
            next_action = epsilon_greed(graph, next_state, Q, eps)
            Q[current_state][action] = Q[current_state][action] + alpha * (
                    reward + gamma * Q[next_state][next_action] - Q[current_state][action])
            current_state = next_state
            action = next_action
            t += 1

    return Q, actions


def sarsa_learning_lambda(graph, eps=0.3, alpha=0.1, gamma=0.95, lambd=0.9, episodes=100, steps=100):
    state_space = len(graph.vs)
    actions = list(set(graph.es['action']))
    actions.sort()
    Q = np.zeros((state_space, len(actions)))
    E = np.zeros((state_space, len(actions)))
    for i in range(episodes):
        current_state = 0
        t = 0
        finished = False
        action = epsilon_greed(graph, current_state, Q, eps)
        while t < steps and not finished:
            if graph.vs[current_state]['absorbing']:
                finished = True
            reward, next_state = act(graph, current_state, action)
            next_action = epsilon_greed(graph, next_state, Q, eps)

            delta = reward + gamma * Q[next_state, next_action] - Q[current_state][action]
            E[current_state][action] += 1

            for s in range(state_space):
                for a in range(len(actions)):
                    Q[s][a] += alpha * delta * E[s][a]
                    E[s][a] *= gamma * lambd

            current_state = next_state
            action = next_action
            t += 1

    return Q, actions


def v_approx(vertex, attributes, weights):
    sum = 0
    for i, attribute in enumerate(attributes):
        sum += weights[i] * vertex[attribute]
    return sum


def h_approx(vertex, attributes, action_coords, weights):
    sum = 0
    i = 0
    for i, attribute in enumerate(attributes):
        sum += weights[i] * vertex[attribute]
    sum += weights[i + 1] * action_coords[0]
    sum += weights[i + 2] * action_coords[1]
    return sum


def pi_theta(vertex, attributes, weights, actions):
    pi = []
    denominator = 0
    for i, action in enumerate(actions):
        coords = get_action_coords(actions, i)
        term = np.exp(h_approx(vertex, attributes, coords, weights))
        denominator += term
    for i, action in enumerate(actions):
        coords = get_action_coords(actions, i)
        numerator = np.exp(h_approx(vertex, attributes, coords, weights))
        pi.append(numerator / denominator)
    return pi


def get_action_coords(actions, action):
    act_dict = {'down': [0, -1], 'left': [-1, 0], 'right': [1, 0], 'up': [0, 1]}
    act_name = actions[action]
    return act_dict[act_name]


def actor_critic_l(graph, alpha=0.1, beta=0.1, gamma=0.95, lambd=0.9, episodes=100, steps=100, size_w=2):
    actions = list(set(graph.es['action']))
    actions.sort()
    w = np.random.random(size_w)
    theta = np.random.random(size_w + 2)
    attributes = ['x', 'y']
    for i in range(episodes):
        current_state = 0
        t = 0
        e_v = np.zeros_like(w)
        e_theta = np.zeros_like(theta)
        finished = False
        while t < steps and not finished:
            if graph.vs[current_state]['absorbing']:
                finished = True
            action = np.argmax(pi_theta(graph.vs[current_state], attributes, theta,
                                        actions))
            reward, next_state = act(graph, current_state, action)
            current_vertex = graph.vs[current_state]
            next_vertex = graph.vs[next_state]
            e_v = gamma * lambd * e_v + np.array(current_vertex['pos'])
            delta = reward + gamma * v_approx(next_vertex, attributes, w) - v_approx(current_vertex, attributes, w)
            w += alpha * delta * e_v

            action_coords = get_action_coords(actions, action)
            grad_pi = list(current_vertex['pos']) + action_coords
            grad_pi_denom = 1 + np.exp(h_approx(current_vertex, attributes, action_coords, theta))
            e_theta = gamma * lambd * e_theta + np.array(grad_pi) / grad_pi_denom
            theta += beta * delta * e_theta

            current_state = next_state
            t += 1

    return theta, actions


def evaluate_q(graph, Q, steps=100, episodes=100):
    actions = list(set(graph.es['action']))
    actions.sort()
    episode_rewards = []
    for i in range(episodes):
        t = 0
        finished = False
        episode_reward = 0
        current_state = 0
        while t < steps and not finished:
            if graph.vs[current_state]['absorbing']:
                finished = True
            action = epsilon_greed(graph, current_state, Q, 0)
            reward, next_state = act(graph, current_state, action)
            episode_reward += reward
            current_state = next_state
            t += 1
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)


def evaluate_theta(graph, theta, steps=100, episodes=100):
    actions = list(set(graph.es['action']))
    actions.sort()
    episode_rewards = []
    attributes = ['x', 'y']
    for i in range(episodes):
        t = 0
        finished = False
        episode_reward = 0
        current_state = 0
        while t < steps and not finished:
            if graph.vs[current_state]['absorbing']:
                finished = True
            action = np.argmax(pi_theta(graph.vs[current_state], attributes, theta,
                                        actions))
            reward, next_state = act(graph, current_state, action)
            episode_reward += reward
            current_state = next_state
            t += 1
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)


if __name__ == "__main__":
    with open("E:\\User Data\\Work\\Computer Science\\Intelligent Agents GTA\\gridworld_graph.pkl", 'rb') as f:
        graph = pickle.load(f)

    Q, actions = q_learning(graph)
    # print(Q, actions)
    print("Q_learning:", evaluate_q(graph, Q))

    Q, actions = sarsa_learning(graph)
    # print(Q, actions)
    print("SARSA_learning:", evaluate_q(graph, Q))

    Q, actions = q_learning_lambda(graph)
    # print(Q, actions)
    print("Q_learning-l:", evaluate_q(graph, Q))

    Q, actions = sarsa_learning_lambda(graph)
    # print(Q, actions)
    print("SARSA_learning-l:", evaluate_q(graph, Q))

    theta, actions = actor_critic_l(graph)
    print("actor-critic-l:", evaluate_theta(graph, theta))
