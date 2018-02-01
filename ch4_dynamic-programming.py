# Chapter 4. Dynamic programming
import numpy as np
import math
from itertools import product
from collections import namedtuple
import matplotlib.pyplot as plt
from IPython import get_ipython

get_ipython().magic(u'matplotlib inline')


# %% Figure 4.1: Gridworld exmaple
# actions:
#           0 (up)
# 3 (left) [agent] 1 (right)
#           2 (down)

# rewards = rewards(state, action)
def gen_rewards():
    rewards = np.zeros((16, 4)) - 1
    rewards[0, :] = 0
    rewards[15, :] = 0
    return rewards


# p(s'|s,a):
#   state_probs[s, a, s'] = p(s'|s,a)
def gen_state_probs():
    state_probs = np.zeros((16, 4, 16))
    for s in range(16):
        if s - 4 >= 0:
            state_probs[s, 0, s - 4] = 1
        else:
            state_probs[s, 0, s] = 1
        if (s - 3) % 4 != 0:
            state_probs[s, 1, s + 1] = 1
        else:
            state_probs[s, 1, s] = 1
        if s + 4 <= 15:
            state_probs[s, 2, s + 4] = 1
        else:
            state_probs[s, 2, s] = 1
        if s % 4 != 0:
            state_probs[s, 3, s - 1] = 1
        else:
            state_probs[s, 3, s] = 1
    state_probs[0, :, :] = 0
    state_probs[0, :, 0] = 1
    state_probs[15, :, :] = 0
    state_probs[15, :, 15] = 1
    return state_probs


def update_values(rewards, gamma, state_probs, values, pi):
    q = rewards + gamma * np.tensordot(state_probs, values, axes=1)
    return np.diagonal(np.dot(q, pi))


rewards = gen_rewards()
state_probs = gen_state_probs()
values = np.zeros(16)
pi = np.zeros((4, 16)) + 0.25
gamma = 1
values.reshape((4, 4)).round(1)
# array([[ 0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.]])
values = update_values(rewards, gamma, state_probs, values, pi)
values.reshape((4, 4)).round(1)
# array([[ 0., -1., -1., -1.],
#        [-1., -1., -1., -1.],
#        [-1., -1., -1., -1.],
#        [-1., -1., -1.,  0.]])
values = update_values(rewards, gamma, state_probs, values, pi)
values.reshape((4, 4)).round(1)
# array([[ 0. , -1.8, -2. , -2. ],
#        [-1.8, -2. , -2. , -2. ],
#        [-2. , -2. , -2. , -1.8],
#        [-2. , -2. , -1.8,  0. ]])
values = update_values(rewards, gamma, state_probs, values, pi)
values.reshape((4, 4)).round(1)
# array([[ 0. , -2.4, -2.9, -3. ],
#        [-2.4, -2.9, -3. , -2.9],
#        [-2.9, -3. , -2.9, -2.4],
#        [-3. , -2.9, -2.4,  0. ]])
for _ in range(7):
    values = update_values(rewards, gamma, state_probs, values, pi)
values.reshape((4, 4)).round(1)
# array([[ 0. , -6.1, -8.4, -9. ],
#        [-6.1, -7.7, -8.4, -8.4],
#        [-8.4, -8.4, -7.7, -6.1],
#        [-9. , -8.4, -6.1,  0. ]])

delta = 1
while True:
    v = values.copy()
    q = rewards + gamma * np.tensordot(state_probs, values, axes=1)
    values = np.diagonal(np.dot(q, pi))
    delta = min(delta, abs(v - values).max())
    if delta < 0.001:
        break

values.reshape((4, 4)).round()
# array([[  0., -14., -20., -22.],
#        [-14., -18., -20., -20.],
#        [-20., -20., -18., -14.],
#        [-22., -20., -14.,   0.]])


# %% Figure 4.2
# Jack's car rental
some_ideas = """
    q(s,a) = sum_{s',r} p(s',r|s,a)(r+gamma*v(s'))
    is equivalent to:
    q(s,a) = sum_{r} (r * p(r|s,a)) + gamma * sum_{s'} (p(s'|s,a) * v(s'))
    then:
    r * p(r|s,a) = 10 * total_requests * p(total_requests|s,a) - 2 * abs(a)
    so we could precompute:
        p(total_requests|s,a) == p(total_requests|s + (-a, a)) == p_requests
            total requests depend only on (s + (-a, a))
            total_requests = 0, 1, 2, ..., 40 == max_cars * 2
    and
        p(s'|s,a) == p(s'|s + (-a, a)) == p_state
    then in the update_value function we can iterate only through states

    We need to remember that there may be states where the business is closed,
    because there are not enough cars to rent. They should also acknowledged.

    Hmm, after look ing at the v(s) = sum... equation it seems we don't have to
    care because the value and reward in those states is 0.
"""


def poisson_prob(k, l):
    return (l ** k) * np.exp(-l) / math.factorial(k)


max_cars = 21  # should be 20 but we want range(max_cars) to include 20
PoissonParams = namedtuple('PoissonParams', ['req', 'ret'])
poiss1 = PoissonParams(3, 3)
poiss2 = PoissonParams(4, 2)


def precompute_probs():
    p_state = np.zeros((max_cars,) * 4)
    p_requests = np.zeros((max_cars * 2 - 1, max_cars, max_cars))

    # let s = (s1, s2) -> sp = (sp1, sp2)
    # location 1: req1, ret1
    # location 2: req2, ret2

    for req1, ret1 in product(range(max_cars), range(max_cars)):
        for req2, ret2 in product(range(max_cars), range(max_cars)):
            for s1, s2 in product(range(req1, max_cars),
                                  range(req2, max_cars)):
                prob = (poisson_prob(req1, poiss1.req) *
                        poisson_prob(ret1, poiss1.ret) *
                        poisson_prob(req2, poiss2.req) *
                        poisson_prob(ret2, poiss2.ret))
                sp1 = min(s1 - req1 + ret1, 20)
                sp2 = min(s2 - req2 + ret2, 20)
                p_state[s1, s2, sp1, sp2] += prob
                p_requests[req1 + req2, s1, s2] += prob

    return p_state, p_requests


def load_probs():
    try:
        probs = np.load('ch4_car_rental_probs.npz')
        return probs['p_state'], probs['p_requests']
    except:
        p_state, p_requests = precompute_probs()
        np.savez('ch4_car_rental_probs.npz',
                 p_state=p_state, p_requests=p_requests)
        return p_state, p_requests


p_state, p_requests = load_probs()
# # Sanity checks:
# p_state[0, 0, 0, 0]
# # should equal:
# zeros = (poisson_prob(0, 3) * poisson_prob(0, 4) * poisson_prob(0, 3)
#          * poisson_prob(0, 2))
# zeros
# p_requests[0, 0, 0]
# # should be tiny bit less then:
# zeros = (poisson_prob(0, 3) * poisson_prob(0, 4))
# zeros


def update_value(s1, s2, a, v, gamma=0.9):
    """
    implements:
        sum_{s',r} p(s',r|s,a)(r+gamma*v(s'))
    but breaks the sum into:
        q(s,a) = sum_{r} (r * p(r|s,a)) + gamma * sum_{s'} (p(s'|s,a) * v(s'))
    see variable some_ideas above.
    """
    reward_update = 0
    for total_requests in range(2 * max_cars - 1):
        reward_update += ((10 * total_requests - 2 * np.abs(a)) *
                          p_requests[total_requests, s1 - a, s2 + a])
    value_update = 0
    for sp1, sp2 in product(range(max_cars), range(max_cars)):
        value_update += p_state[s1 - a, s2 + a, sp1, sp2] * v[sp1, sp2]
    return reward_update + gamma * value_update

# init:
#   policy <- 0 for all s
#   v <- 0 for all s
pi = np.zeros((max_cars, max_cars), dtype=int)
v = np.zeros((max_cars, max_cars))
np.set_printoptions(precision=2, suppress=True, linewidth=150)

value_functions = []
policies = []
# policy evaluation:
while True:
    while True:
        delta = 0
        for s1 in range(max_cars):
            for s2 in range(max_cars):
                v_ = v[s1, s2]
                action = pi[s1, s2]
                v[s1, s2] = update_value(s1, s2, action, v)
                delta = max(delta, np.abs(v_ - v[s1, s2]))
        print("delta: ", np.round(delta, 2))
        if delta < 0.01:
            break
    value_functions.append(v.copy())

    # policy improvement:
    policy_stable = True
    for s1 in range(max_cars):
        for s2 in range(max_cars):
            old_action = pi[s1, s2]
            actions = [a for a in range(-5, 6)
                       if s1 - a >= 0 and s1 - a <= 20
                       and s2 + a >= 0 and s2 + a <= 20]
            qs = np.array([update_value(s1, s2, a, v) for a in actions])
            pi[s1, s2] = actions[qs.argmax()]
            if old_action != pi[s1, s2]:
                policy_stable = False
    policies.append(pi.copy())
    if policy_stable:
        break


for policy in policies[:-1]:
    plt.subplots(figsize=(9, 7))
    pp = plt.imshow(policy, origin='lower', interpolation='none')
    plt.colorbar(pp, orientation='vertical')
    plt.show()

# It looks reasonably similar but not exactly the same.
# There might be some subtle differences in how I interpret the rules.
# I see no point in spending additional hours on looking for them.
# It is very suspicious that lower right corner in Sutton's plots is -4
# rather then -5..

# %% Figure 4.3
idea = """
We will use similar approach as before:
    q(s,a) = sum_{s',r} p(s',r|s,a)(r+gamma*v(s'))
    is equivalent to:
    q(s,a) = sum_{r} (r * p(r|s,a)) + gamma * sum_{s'} (p(s'|s,a) * v(s'))
and do it with matrx multiplication.
Where:
    sum_{r} (r * p(r|s,a)) = 1 * prob(winning|s,a)
and:
    sum_{s'} (p(s'|s,a) * v(s')) =
        p_h * v(s + a) +
        (1 - p_h) * v(s - a)
"""


def figure_4_3(p_h, acc=4):
    # rewards = rewards(state, action)
    def gen_rewards(p):
        rewards = np.zeros((101, 51))
        for state in range(50, 100):
            rewards[state, 100 - state] = p
        return rewards

    # p(s'|s,a):
    #   state_probs[s, a, s'] = p(s'|s,a)
    def gen_state_probs(p):
        state_probs = np.zeros((101, 51, 101))
        # terminal states:
        state_probs[0, :, 0] = 1
        state_probs[100, :, 100] = 1
        # non-terminal states
        for state in range(1, 100):
            for action in range(51):
                if state + action < 101 and state >= action:
                    state_probs[state, action, state + action] = p
                    state_probs[state, action, state - action] = 1 - p
        return state_probs

    rewards = gen_rewards(p_h)
    state_probs = gen_state_probs(p_h)
    values = np.zeros(101)
    gamma = 1
    remember = [1, 2, 3, 32]
    i = 1
    history = []
    policy = 0  # 0 is a placeholder
    q = 0  # a placeholder as well

    while True:
        v_ = values
        q = rewards + gamma * np.tensordot(state_probs, values, axes=1)
        values = q.max(axis=1)
        policy = q.round(acc).argmax(-1)
        if i in remember:
            history.append(values)
        i += 1
        if abs(v_ - values).max() < 0.001 and i > 100:
            print("V inc:", abs(v_ - values).max())
            print("i: ", i)
            history.append(values)
            break

    plt.subplots(figsize=(8, 6))
    for i in range(len(history)):
        plt.plot(range(1, 100), history[i][1:100])
    plt.show()

    plt.plot(range(1, 100), policy[1:100])
    plt.subplots(figsize=(14, 11))
    pp = plt.imshow(q.round(acc), origin='lower', interpolation='none')
    plt.colorbar(pp, orientation='vertical')
    plt.show()

# it's funny how much role accuracy plays here:
figure_4_3(0.4, acc=100)
figure_4_3(0.4, acc=4)
figure_4_3(0.25)
figure_4_3(0.55)
figure_4_3(0.5)
