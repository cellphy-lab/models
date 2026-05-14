# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import sem

#models

class UCBRLAgent:

    def __init__(self, alpha=0.5, phi=0.5, beta=0.5, decay_lambda=0.5, bias=0.5, num_actions=2):

        self.alpha = alpha
        self.phi = phi
        self.beta = beta
        self.decay_lambda = decay_lambda
        self.beta_bias = bias

        self.ph1 = 0.5
        self.ph2 = 0.5

        self.num_actions = num_actions
        self.Q = np.full(num_actions, 0.5)
        self.N = np.zeros(num_actions)
        self.t = 0

    def update(self, action, reward, k, t):

        #k = the index of the last trial where current action was chosen

        Phigh = 0.8
        Plow = 0.1
        poh1 = np.array([[1-Plow, Plow], [1-Phigh, Phigh]])
        poh2 = np.array([[1-Phigh, Phigh], [1-Plow, Plow]])

        c = action
        o = reward

        ph1neu = poh1[c, o] * self.ph1 / (poh1[c, o] * self.ph1 + poh2[c, o] * self.ph2)
        self.ph1 += 1 * (ph1neu - self.ph1)
        self.ph2 = 1 - self.ph1
        prev = 0.25 #fixed value
        self.ph1 = (1 - prev) * self.ph1 + prev * self.ph2
        self.ph2 = 1 - self.ph1

        if abs(self.ph1 - self.ph2) < 0.2:
            self.N = np.zeros(2)
            self.t = 0

        N_stable = np.where(self.N == 0, 1, self.N)
        t_stable = max(1, self.t)

        if self.t > 0.0:
            uncertainty_bonus = np.sqrt(np.log(t_stable) / N_stable)
        else:
            uncertainty_bonus = np.zeros(self.num_actions)

        dQ = self.Q[1] - self.Q[0]
        dUB = uncertainty_bonus[1] - uncertainty_bonus[0]


        probabilities = 1 / (1 + np.exp(-(self.beta * (dQ + self.phi * dUB + self.beta_bias))))
        action_probs = [1 - probabilities, probabilities]


        prediction_error = reward - self.decay_lambda**(t-k) * self.Q[action]
        self.Q[action] += self.alpha * prediction_error

        self.N[action] += 1
        self.t += 1

        return self.Q, uncertainty_bonus, [1-probabilities, probabilities], prediction_error

class RLAgent:

    def __init__(self, alpha=0.5, beta=0.5, decay_lambda=0.5, bias=0.5, num_actions=2):

        self.alpha = alpha
        self.beta = beta
        self.decay_lambda = decay_lambda
        self.beta_bias = bias

        self.ph1 = 0.5
        self.ph2 = 0.5

        self.num_actions = num_actions
        self.Q = np.full(num_actions, 0.5)
        self.N = np.zeros(num_actions)
        self.t = 0

    def update(self, action, reward, k, t):
        # k = the index of the last trial where current action was chosen

        Phigh = 0.8
        Plow = 0.1
        poh1 = np.array([[1-Plow, Plow], [1-Phigh, Phigh]])
        poh2 = np.array([[1-Phigh, Phigh], [1-Plow, Plow]])

        c = action
        o = reward

        ph1neu = poh1[c, o] * self.ph1 / (poh1[c, o] * self.ph1 + poh2[c, o] * self.ph2)
        self.ph1 += 1 * (ph1neu - self.ph1)
        self.ph2 = 1 - self.ph1
        prev = 0.25
        self.ph1 = (1 - prev) * self.ph1 + prev * self.ph2
        self.ph2 = 1 - self.ph1

        if abs(self.ph1 - self.ph2) < 0.2:
            self.N = np.zeros(2)
            self.t = 0

        dQ = self.Q[1] - self.Q[0]
        probabilities = 1 / (1 + np.exp(-(self.beta * (dQ + self.beta_bias))))
        action_probs = [1 - probabilities, probabilities]

        prediction_error = reward - self.decay_lambda**(t-k) * self.Q[action]
        self.Q[action] += self.alpha * prediction_error

        self.N[action] += 1
        self.t += 1

        return self.Q, action_probs, prediction_error

class BayesAgent:

    def __init__(self, prev=0.5, beta=0.5, delta=0.5, bias=0.5, num_actions=2,):

        self.prev = prev
        self.beta = beta
        self.delta = delta
        self.beta_bias = bias


        self.b = 0.5


        self.num_actions = num_actions
        self.Q = np.full(num_actions, 0.5)
        self.t = 0
    def update(self, action, reward):

        dQ = self.Q[1] - self.Q[0]
        probabilities = 1 / (1 + np.exp(-(self.beta * (dQ + self.beta_bias))))
        action_probs = [1 - probabilities, probabilities]

        prediction_error = reward - self.Q[action]

        Phigh = 0.8
        Plow = 0.1
        poh1 = np.array([[1-Plow, Plow], [1-Phigh, Phigh]])
        poh2 = np.array([[1-Phigh, Phigh], [1-Plow, Plow]])

        c = action
        o = reward

        self.b = poh1[c, o] * self.b / (poh1[c, o] * self.b + poh2[c, o] * (1 - self.b))

        self.b = (1 - self.prev) * self.b + self.prev * (1 - self.b)

        self.Q[c] = Phigh * self.b + Plow * (1 - self.b)
        self.Q[1-c] *= self.delta

        return self.Q, self.b, action_probs, prediction_error

class BRLAgent:

    def __init__(self, alpha=0.5, beta=0.5, q=0.5, bias=0.5):

        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.beta_bias = bias

        self.rho1 = 0.8
        self.rho2 = 0.1

        self.w1 = self.rho1
        self.w2 = self.rho2
        self.b = np.array([0.5, 0.5])
        self.Q = np.array([0.5, 0.5])

    def update(self, action, reward):


        self.Q[1] = self.w1 * self.b[1] + self.w2 * self.b[0]
        self.Q[0] = self.w1 * self.b[0] + self.w2 * self.b[1]

        D_t = self.Q[1] - self.Q[0] + self.beta_bias
        pR = 1 / (1 + np.exp(-(self.beta * D_t )))


        b_chosen = self.b[action]
        b_unchosen = self.b[1 - action]

        Q_chosen = self.w1 * b_chosen + self.w2 * b_unchosen
        RPE = reward - Q_chosen

        self.w1 += self.alpha * RPE * b_chosen
        self.w2 += self.alpha * RPE * b_unchosen #w2 is fixed in BRLfwr

        L = np.zeros(2)
        for z in [0, 1]:
            p_reward = self.rho1 if z == action else self.rho2
            L[z] = p_reward if reward == 1 else (1 - p_reward)

        b_post = L * self.b
        b_post /= np.sum(b_post)

        b_next = np.zeros(2)
        b_next[0] = (1 - self.q) * b_post[0] + self.q * b_post[1]
        b_next[1] = (1 - self.q) * b_post[1] + self.q * b_post[0]

        self.b = b_next
        return self.Q, self.b[1], [1-pR, pR], RPE

class PHAgent:

    def __init__(self, alpha_p=0.5, alpha_n=0.5, beta=0.5, psi=0.5, delta=0.5, bias=0.5):

        self.alpha_p = alpha_p
        self.alpha_n = alpha_n
        self.beta = beta
        self.psi = psi
        self.delta = delta
        self.beta_bias = bias

        self.q0 = 0.5
        self.alpha_v = 1
        self.Q = np.array([0.5, 0.5])

    def update(self, action, reward):


        D_t = self.Q[1] - self.Q[0] + self.beta_bias
        pR = 1 / (1 + np.exp(-(self.beta * D_t )))



        Q_chosen = self.Q[action]
        Q_unchosen = self.Q[1-action]

        RPE = reward - Q_chosen

        alpha_base = self.alpha_p if RPE >= 0 else self.alpha_n

        Q_chosen_next = Q_chosen + (alpha_base * self.alpha_v * RPE)
        Q_unchosen_next = self.q0 + self.delta * (Q_unchosen - self.q0)

        self.alpha_v = self.alpha_v + self.psi * (abs(RPE) - self.alpha_v)

        self.Q[action] = Q_chosen_next
        self.Q[1-action] = Q_unchosen_next

        return self.Q, self.alpha_v, [1-pR, pR], RPE

#optimization algorithm for each model

def UCBRLAgent_optz (prms, df, optimization=True):
    alpha, phi, beta, decay_lambda, bias = prms
    Qs = []
    UBs = []
    prbs = []
    RPEs = []

    last_seen_indices = np.ones(2, dtype=int)

    agent = UCBRLAgent(
            alpha=alpha,
            phi=phi,
            beta=beta,
            decay_lambda=decay_lambda,
            bias = bias
        )

    actions = df['poke']
    rewards = df['reward']

    for t in range(len(actions)):
        observed_action = int(actions[t])
        observed_reward = int(rewards[t])

        k = last_seen_indices[observed_action] + 1

        Q, uncertainty_bonus, probabilities, RPE = agent.update(observed_action, observed_reward, k, t+1)

        last_seen_indices[observed_action] = t+1

        Qs.append(Q.copy())
        UBs.append(uncertainty_bonus)
        prbs.append(probabilities)
        RPEs.append(RPE)

    epsilon = 1e-15

    pL = np.clip(np.array(prbs)[:, 0], epsilon, 1 - epsilon)
    pR = np.clip(np.array(prbs)[:, 1], epsilon, 1 - epsilon)
    MLL = -np.sum(np.log(pR[actions == 1])) - np.sum(np.log(pL[actions == 0]))

    if optimization:
        return MLL
    else:
        return Qs, UBs, prbs, RPEs

def RLAgent_optz (prms, df, optimization=True):
    alpha, beta, decay_lambda, bias = prms
    Qs = []
    prbs = []
    RPEs = []

    last_seen_indices = np.ones(2, dtype=int)

    agent = RLAgent(
            alpha=alpha,
            beta=beta,
            decay_lambda=decay_lambda,
            bias = bias
        )

    actions = df['poke']
    rewards = df['reward']

    for t in range(len(actions)):
        observed_action = int(actions[t])
        observed_reward = int(rewards[t])

        k = last_seen_indices[observed_action] + 1

        Q, probabilities, RPE = agent.update(observed_action, observed_reward, k, t+1)

        last_seen_indices[observed_action] = t+1

        Qs.append(Q.copy())
        prbs.append(probabilities)
        RPEs.append(RPE)

    epsilon = 1e-15

    pL = np.clip(np.array(prbs)[:, 0], epsilon, 1 - epsilon)
    pR = np.clip(np.array(prbs)[:, 1], epsilon, 1 - epsilon)
    MLL = -np.sum(np.log(pR[actions == 1])) - np.sum(np.log(pL[actions == 0]))

    if optimization:
        return MLL
    else:
        return Qs, prbs, RPEs

def BayesAgent_optz (prms, df, optimization=True):

    prev, beta, delta, bias = prms

    Qs = []
    phs = []
    prbs = []
    RPEs = []

    agent = BayesAgent(
            prev=prev,
            beta=beta,
            bias=bias,
            delta=delta
        )

    actions = df['poke']
    rewards = df['reward']

    for t in range(len(actions)):
        observed_action = int(actions[t])
        observed_reward = int(rewards[t])

        Q, ph, probabilities, RPE = agent.update(observed_action, observed_reward)

        Qs.append(Q.copy())
        phs.append(ph.copy())
        prbs.append(probabilities)
        RPEs.append(RPE)


    epsilon = 1e-15

    pL = np.clip(np.array(prbs)[:, 0], epsilon, 1 - epsilon)
    pR = np.clip(np.array(prbs)[:, 1], epsilon, 1 - epsilon)
    MLL = -np.sum(np.log(pR[actions == 1])) - np.sum(np.log(pL[actions == 0]))

    if optimization:
        return MLL
    else:
        return Qs, phs, prbs, RPEs

def BRLAgent_optz (prms, df, optimization=True):

    alpha, beta, q, bias = prms

    Qs = []
    phs = []
    prbs = []
    RPEs = []


    agent = BRLAgent(
            alpha=alpha,
            beta=beta,
            q=q,
            bias=bias
        )

    actions = df['poke']
    rewards = df['reward']

    for t in range(len(actions)):
        observed_action = int(actions[t])
        observed_reward = int(rewards[t])

        Q, ph, probabilities, RPE = agent.update(observed_action, observed_reward)

        Qs.append(Q.copy())
        phs.append(ph.copy())
        prbs.append(probabilities)
        RPEs.append(RPE)


    epsilon = 1e-15

    pL = np.clip(np.array(prbs)[:, 0], epsilon, 1 - epsilon)
    pR = np.clip(np.array(prbs)[:, 1], epsilon, 1 - epsilon)
    MLL = -np.sum(np.log(pR[actions == 1])) - np.sum(np.log(pL[actions == 0]))

    if optimization:
        return MLL
    else:
        return Qs, phs, prbs, RPEs

def PHAgent_optz (prms, df, optimization=True):

    alpha_p, alpha_n, beta, psi, delta, bias = prms

    Qs = []
    alpha_vs = []
    prbs = []
    RPEs = []


    agent = PHAgent(
            alpha_p=alpha_p,
            alpha_n=alpha_n,
            beta=beta,
            psi=psi,
            delta=delta,
            bias=bias
        )

    actions = df['poke']
    rewards = df['reward']

    for t in range(len(actions)):
        observed_action = int(actions[t])
        observed_reward = int(rewards[t])

        Q, alpha_v, probabilities, RPE = agent.update(observed_action, observed_reward)

        Qs.append(Q.copy())
        alpha_vs.append(alpha_v.copy())
        prbs.append(probabilities)
        RPEs.append(RPE)


    epsilon = 1e-15

    pL = np.clip(np.array(prbs)[:, 0], epsilon, 1 - epsilon)
    pR = np.clip(np.array(prbs)[:, 1], epsilon, 1 - epsilon)
    MLL = -np.sum(np.log(pR[actions == 1])) - np.sum(np.log(pL[actions == 0]))

    if optimization:
        return MLL
    else:
        return Qs, alpha_vs, prbs, RPEs



#optimization algorithm


def optz(df, agent_optz, params, bounds):
    result = minimize(
        agent_optz,
        params,
        args=(df, True),
        method='L-BFGS-B',
        bounds=bounds
        )

    return result



###intial parameters & upper/lower bounds

#UCBRL = alpha, phi(c), beta, decay, bias
#RL = alpha, beta, decay, bias
#Bayes = prev, beta, delta, bias
#BRL = alpha, beta, q(switch_param), bias
#PH = alpha_p, alpha_n, beta, psi(uncertainty adj.), delta, bias

UCBRL_initial_params = np.array([0.5, 0.5, 0.15, 0.9, 0])
RL_initial_params = np.array([0.5, 0.15, 0.9, 0])
Bayes_initial_params = np.array([0.5, 0.15, 0.1, 0])
BRL_initial_params = np.array([0.5, 0.15, 0.1, 0.5])
PH_initial_params = np.array([0.5, 0.5, 0.15, 0.5, 0.1, 0.5])

UCBRL_bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (-1,1)]
RL_bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (-1,1)]
Bayes_bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (-1,1)]
BRL_bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]
PH_bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]



#logistic regression

def logit(data, nh=10):

    poke = data['poke'].values
    reward = data['reward'].values

    xw_raw = sliding_window_view(poke[:-1], window_shape=nh)
    rw_raw = sliding_window_view(reward[:-1], window_shape=nh)

    y = poke[nh:]

    xw = np.where(xw_raw == 0, -1, 1)
    y = np.where(y == 0, -1, 1)

    xrw = xw * rw_raw
    xuw = xw * (1 - rw_raw)

    X = np.hstack((xrw, xuw))

    model = LogisticRegression(
        solver='saga',
        penalty='l1',
        max_iter=2000,
        random_state=42
    )
    model.fit(X, y)

    return model.coef_[0]

#plot logit regression

def plot_logit_coeffs(coefficients, name, nh=10): #individual
    # Split the returned coefficients back into rewarded vs unrewarded
    # Your model stacked them: [xrw_1...xrw_10, xuw_1...xuw_10]
    reward_coeffs = coefficients[:nh]
    unrewarded_coeffs = coefficients[nh:]

    trials_back = range(1, nh + 1)

    plt.figure(figsize=(8, 5))

    # Plotting Choice * Reward (Success influence)
    plt.plot(trials_back, reward_coeffs, marker='o', label='Rewarded Trials', color='green')

    # Plotting Choice * Unrewarded (Failure influence)
    plt.plot(trials_back, unrewarded_coeffs, marker='s', label='Unrewarded Trials', color='red')


    # Formatting
    plt.axhline(0, color='black', lw=1, ls='--') # Baseline
    plt.title(f"logistic regression: {name}")
    plt.xlabel('Trials Back (t - n)')
    plt.ylabel('Coefficient Weight')
    plt.xticks(trials_back)
    plt.legend()
    plt.grid(True, alpha=0.3)


    plt.show()

def plot_mean_logit_coeffs(all_coeffs, name, nh=10): #group

    mean_weights = np.mean(all_coeffs, axis=0)
    sem_weights = sem(all_coeffs, axis=0)


    mean_rwd = mean_weights[:nh]
    sem_rwd = sem_weights[:nh]

    mean_urw = mean_weights[nh:]
    sem_urw = sem_weights[nh:]

    trials_back = np.arange(1, nh + 1)

    plt.figure(figsize=(9, 6), dpi=100)


    plt.plot(trials_back, mean_rwd, marker='o', label='Rewarded Trials', color='green', lw=2)
    plt.fill_between(trials_back, mean_rwd - sem_rwd, mean_rwd + sem_rwd, color='green', alpha=0.2)


    plt.plot(trials_back, mean_urw, marker='s', label='Unrewarded Trials', color='red', lw=2)
    plt.fill_between(trials_back, mean_urw - sem_urw, mean_urw + sem_urw, color='red', alpha=0.2)


    plt.axhline(0, color='black', lw=1, ls='--')
    plt.title(f"logistic regression: {name}")
    plt.xlabel('Trials Back (t - n)')
    plt.ylabel('Coefficient Weight (Mean ± SEM)')
    plt.xticks(trials_back)
    plt.ylim(-0.35,0.35)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.5)


    plt.show()

