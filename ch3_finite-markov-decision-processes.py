# Chapter 3. Finite Markov Decision Processes
import numpy as np


# %% Figure 3.5: Gridworld exmaple
# gw - Gridworld
gw = np.zeros((5, 5))
value = np.zeros((5, 5))

# actions:
#           0 (up)
# 3 (left) [agent] 1 (right)
#           2 (down)


# value(s) = \sum_a (\pi(a|s)  *
#                    \sum_{s',r}(p(s',r|s,a) * (r+\gamma*value(s')) ) )
def uniform_policy(action, state):
    return 0.25


def reward(s1, a, s2):  # (s1,a) -> s2
    if s1 == (0, 1):
        return 10
    elif s1 == (0, 3):
        return 5
    elif (s1[0] == 0 and a == 0) or
    (s1[1] == 4 and a == 1) or
    (s1[0] == 4 and a == 2) or
    (s1[1] == 0 and a == 3):
        return -1
    else:
        return 0

# check numpy dot and numpy tensordot
