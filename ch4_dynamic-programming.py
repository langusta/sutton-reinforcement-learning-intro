# Chapter 4. Dynamic programming
import numpy as np


# %% Figure 4.2: Gridworld exmaple
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
