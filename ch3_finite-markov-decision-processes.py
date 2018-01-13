# Chapter 3. Finite Markov Decision Processes
import numpy as np


# %% Figure 3.5: Gridworld exmaple
# actions:
#           0 (up)
# 3 (left) [agent] 1 (right)
#           2 (down)

# rewards = rewards(state, action)
def gen_rewards():
    rewards = np.zeros((25, 4))
    rewards[0:5, 0] = -1
    rewards[4:25:5, 1] = -1
    rewards[20:25, 2] = -1
    rewards[0:25:5, 3] = -1
    rewards[1, :] = 10
    rewards[3, :] = 5
    return rewards


# p(s'|s,a):
#   state_probs[s, a, s'] = p(s'|s,a)
def gen_state_probs():
    state_probs = np.zeros((25, 4, 25))
    for s in range(25):
        if s == 1:  # state A
            state_probs[s, :, 21] = 1
        elif s == 3:  # state B
            state_probs[s, :, 13] = 1
        else:
            if s - 5 >= 0:
                state_probs[s, 0, s - 5] = 1
            else:
                state_probs[s, 0, s] = 1
            if (s - 4) % 5 != 0:
                state_probs[s, 1, s + 1] = 1
            else:
                state_probs[s, 1, s] = 1
            if s + 5 <= 24:
                state_probs[s, 2, s + 5] = 1
            else:
                state_probs[s, 2, s] = 1
            if s % 5 != 0:
                state_probs[s, 3, s - 1] = 1
            else:
                state_probs[s, 3, s] = 1
    return state_probs
# np.sum(state_probs, axis=2)
# np.tensordot(state_probs, np.arange(25), axes=1)

# lets compute the values:
values = np.zeros(25)
rewards = gen_rewards()
state_probs = gen_state_probs()
pi = np.zeros((4, 25)) + 0.25
gamma = 0.9

for _ in range(1000):
    q = rewards + gamma * np.tensordot(state_probs, values, axes=1)
    values = np.diagonal(np.dot(q, pi))
values.reshape((5, 5)).round(1)
# array([[ 3.3,  8.8,  4.4,  5.3,  1.5],
#        [ 1.5,  3. ,  2.3,  1.9,  0.5],
#        [ 0.1,  0.7,  0.7,  0.4, -0.4],
#        [-1. , -0.4, -0.4, -0.6, -1.2],
#        [-1.9, -1.3, -1.2, -1.4, -2. ]])

# %% Figure 3.8
# we can use previous code, but add value adjustment step to final loop:
# lets compute the values:
values = np.zeros(25)
rewards = gen_rewards()
state_probs = gen_state_probs()
gamma = 0.9

for _ in range(1000):
    q = rewards + gamma * np.tensordot(state_probs, values, axes=1)
    # now values are just max_a q(s,a)
    values = q.max(-1)
values.reshape((5, 5)).round(1)
# array([[ 22. ,  24.4,  22. ,  19.4,  17.5],
#        [ 19.8,  22. ,  19.8,  17.8,  16. ],
#        [ 17.8,  19.8,  17.8,  16. ,  14.4],
#        [ 16. ,  17.8,  16. ,  14.4,  13. ],
#        [ 14.4,  16. ,  14.4,  13. ,  11.7]])
