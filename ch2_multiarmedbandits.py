# Multi armed bandits
import numpy as np
import math
# from multiprocessing import Pool
import matplotlib.pyplot as plt
%matplotlib inline

# TODO:
# refactor

# %% Classes:
# %% Arms:


class Arm(object):
    """ Multi armed bandit arm template"""
    def __init__(self):
        pass

    def reward(self):
        pass

    def modify(self):
        pass

    def __call__(self):
        pass


class NormalArm(Arm):
    """Stationary arm sampled from normal distribution"""
    def __init__(self, mu=0, sd=1):
        Arm.__init__(self)
        self.mu = mu
        self.sd = sd

    def reward(self):
        return np.random.normal(self.mu, self.sd)

    def modify(self, delta_mu, delta_sd):
        self.mu += delta_mu
        self.sd += delta_sd

    def __call__(self):
        return self.mu


# %% Actions:


class ActionValue:
    """ Estimate value """
    def __init__(self, estimate=0):
        self.estimate = estimate
        self.n = 0

    def step_size(self):
        pass

    def use(self):
        self.n += 1

    def update(self, target):
        self.use()
        self.estimate = self.estimate + \
            self.step_size()*(target - self.estimate)

    def __call__(self):
        return self.estimate


class SampleAverageActionValue(ActionValue):
    """ Value as the average of past rewards """
    def __init__(self, estimate=0):
        ActionValue.__init__(self, estimate)

    def step_size(self):
        return 1/self.n


class ConstantStepSizeActionValue(ActionValue):
    """ Value as the average of past rewards """
    def __init__(self, estimate=0, step_size=0.1):
        ActionValue.__init__(self, estimate)
        self.constant_step_size = step_size

    def step_size(self):
        return self.constant_step_size


class PreferenceActionValue(ConstantStepSizeActionValue):
    def update(self, target):
        self.use()
        self.estimate = self.estimate + self.step_size()*target


# %% Bandit


class Bandit:
    """ choose an arm and win the prize """
    def __init__(self, arms, action_values, delta_mu=lambda: 0,
                 delta_sd=lambda: 0):
        self.arms = arms
        self.action_values = action_values
        self.delta_mu = delta_mu
        self.delta_sd = delta_sd

    def choose_arm(self, iteration):
        pass

    def one_step(self, iteration):
        index = self.choose_arm(iteration)
        reward = self.arms[index].reward()
        self.action_values[index].update(reward)
        # line below makes most sense in case of nonstationary arms
        if_best = self.arms[index]() == max([arm() for arm in self.arms])
        # line below is for the case of nonstationary arms
        # self.arms[index].mu += self.delta_mu()
        # self.arms[index].sd += self.delta_sd()
        self.arms[index].modify(self.delta_mu(), self.delta_sd())
        return reward, if_best

    def simulate(self, iterations, **kwargs):
        out = [self.one_step(it) for it in range(iterations)]
        return [r for r, i in out], [i for r, i in out]


class BanditEpsGreedy(Bandit):
    def __init__(self, arms, action_values, delta_mu=lambda: 0,
                 delta_sd=lambda: 0, eps=0.1):
        Bandit.__init__(self, arms, action_values, delta_mu, delta_sd)
        self.eps = eps

    def choose_arm(self, iteration):
        if np.random.uniform() < self.eps:
            return np.random.choice(len(self.arms))
        else:
            v = [av() for av in self.action_values]
            return v.index(max(v))


class UCB(Bandit):
    def __init__(self, arms, action_values, delta_mu=lambda: 0,
                 delta_sd=lambda: 0, c=2.0):
        Bandit.__init__(self, arms, action_values, delta_mu, delta_sd)
        self.c = c

    def choose_arm(self, iteration):
        ns = [av.n for av in self.action_values]
        if min(ns) == 0:
            return ns.index(0)
        else:
            ucb = [av()+self.c*math.sqrt(math.log(iteration+1)/av.n)
                   for av in self.action_values]
            return ucb.index(max(ucb))


class GradientBandit(Bandit):
    """It assumes that you will use PreferenceActionValue-s"""
    def __init__(self, arms, action_values, delta_mu=lambda: 0,
                 delta_sd=lambda: 0, baseline=SampleAverageActionValue()):
        Bandit.__init__(self, arms, action_values, delta_mu, delta_sd)
        self.baseline = baseline

    def policy_probs(self):
        pref = np.exp([av() for av in self.action_values])
        return pref/sum(pref)

    def choose_arm(self, iteration):
        pref = self.policy_probs()
        return np.random.choice(range(len(pref)), p=pref)

    def one_step(self, iteration):
        index = self.choose_arm(iteration)
        reward = self.arms[index].reward()
        self.baseline.update(reward)
        prefs = (-1)*self.policy_probs()
        prefs[index] += 1
        for i in range(len(self.action_values)):
            self.action_values[i].update((reward - self.baseline())*prefs[i])

        # line below makes most sense in case of nonstationary arms
        if_best = self.arms[index]() == max([arm() for arm in self.arms])
        # line below is for the case of nonstationary arms
        self.arms[index].modify(self.delta_mu(), self.delta_sd())
        return reward, if_best


# %% tests:

def experiment(bandit_constr, arms, action_values, iterations, sessions):
    cum_rewards = np.array([0.0]*iterations)
    cum_proc_best = np.array([0.0]*iterations)
    for s in range(sessions):
        bandit = bandit_constr(arms(), action_values())
        rewards, proc_best = bandit.simulate(iterations)
        cum_rewards += rewards
        cum_proc_best += proc_best
    return cum_rewards/sessions, cum_proc_best/sessions


def plot_multiple(x, *args):
    plt.subplots(figsize=(8, 6))
    for y, label in args:
        plt.plot(x, y, label=label)
    leg = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                     ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.7)
    plt.show()


# %%

def get_normal_arms(number=10):
    arms_init = np.random.normal(0, 1, number)
    return [NormalArm(mu, 1) for mu in arms_init]


def get_sample_actions_values(number=10, estimate=0.0):
    return [SampleAverageActionValue(estimate) for _ in range(number)]


iterations = 1000
samples = 500
rewards_eps01, proc_best_eps01 = \
                     experiment(lambda a, b: BanditEpsGreedy(a, b, eps=0.1),
                                get_normal_arms,
                                get_sample_actions_values,
                                iterations, samples)

rewards_eps001, proc_best_eps001 = \
                     experiment(lambda a, b: BanditEpsGreedy(a, b, eps=0.01),
                                get_normal_arms,
                                get_sample_actions_values,
                                iterations, samples)

rewards_eps0, proc_best_eps0 = \
                     experiment(lambda a, b: BanditEpsGreedy(a, b, eps=0.0),
                                get_normal_arms,
                                get_sample_actions_values,
                                iterations, samples)


# %% Eps vs greedy

plot_multiple(range(iterations),
              (rewards_eps01, "Eps01"),
              (rewards_eps001, "Eps001"),
              (rewards_eps0, "Eps0 (Greedy)"))

plot_multiple(range(iterations),
              (proc_best_eps01, "Eps01"),
              (proc_best_eps001, "Eps001"),
              (proc_best_eps0, "Eps0 (Greedy)"))


# %% Optimistic Greedy vs Eps Greedy 0.1


def get_css_actions_value(number=10, estimate=0.0, step_size=0.1):
    return [ConstantStepSizeActionValue(estimate=estimate, step_size=step_size)
            for _ in range(number)]


def get_css_actions_values_optimistic(number=10):
    return get_css_actions_value(number, estimate=5.0)


iterations = 1000
samples = 500
rewards_eps01_css, proc_best_eps01_css = \
                     experiment(lambda a, b: BanditEpsGreedy(a, b, eps=0.1),
                                get_normal_arms,
                                get_css_actions_value,
                                iterations, samples)

rewards_greedy_opt, proc_best_greedy_opt = \
                     experiment(lambda a, b: BanditEpsGreedy(a, b, eps=0.0),
                                get_normal_arms,
                                get_css_actions_values_optimistic,
                                iterations, samples)

plot_multiple(range(iterations),
              (proc_best_greedy_opt,
              "Greedy, Optimistic, Q1=5, step size=0.1"),
              (proc_best_eps01_css, "Eps01, step size=0.1"))

# %% Ex 2.3 Nonstationary problems:


def get_equal_normal_arms(number=10):
    return [NormalArm(0, 1) for _ in range(number)]


def binary_delta_mu():
    return 0.1*(1 if np.random.rand() < 0.5 else -1)


def normal_delta_mu():
    return np.random.normal(0, 0.1)


iterations = 1500
samples = 200
# non_stationary, sample_mean
rewards_nons_smean, proc_best_nons_smean = \
                     experiment(lambda a, b:
                                BanditEpsGreedy(a, b, eps=0.1,
                                                delta_mu=binary_delta_mu),
                                get_equal_normal_arms,
                                get_sample_actions_values,
                                iterations, samples)
# non_stationary, constant step size
rewards_nons_css, proc_best_nons_css = \
                     experiment(lambda a, b:
                                BanditEpsGreedy(a, b, eps=0.1,
                                                delta_mu=binary_delta_mu),
                                get_equal_normal_arms,
                                get_css_actions_value,
                                iterations, samples)

print("Nonstationary arms - step size is either 0.1 or -0.1")
print("Eps-greedy algorithm with eps=0.1 and either:")
print("- sample mean arm value estimation")
print("- constant step size arm value estimation")

plot_multiple(range(iterations),
              (rewards_nons_smean, "NonS,Eps01,SampleMean"),
              (rewards_nons_css, "NonS,Eps01,CSS01"))

plot_multiple(range(iterations),
              (proc_best_nons_smean, "NonS,Eps01,SampleMean"),
              (proc_best_nons_css, "NonS,Eps01,CSS01"))

# Random walk given by normal distributed steps:

iterations = 1500
samples = 200
# non_stationary, sample_mean
rewards_nons_smean, proc_best_nons_smean = \
                     experiment(lambda a, b:
                                BanditEpsGreedy(a, b, eps=0.1,
                                                delta_mu=normal_delta_mu),
                                get_equal_normal_arms,
                                get_sample_actions_values,
                                iterations, samples)

# non_stattionary, constant step size
rewards_nons_css, proc_best_nons_css = \
                     experiment(lambda a, b:
                                BanditEpsGreedy(a, b, eps=0.1,
                                                delta_mu=normal_delta_mu),
                                get_equal_normal_arms,
                                get_css_actions_value,
                                iterations, samples)

print("Nonstationary arms - step size is normally distributed.")
print("Eps-greedy algorithm with eps=0.1 and either:")
print("- sample mean arm value estimation")
print("- constant step size arm value estimation")

plot_multiple(range(iterations),
              (rewards_nons_smean, "SampleMean"),
              (rewards_nons_css, "CSS01"))

plot_multiple(range(iterations),
              (proc_best_nons_smean, "SampleMean"),
              (proc_best_nons_css, "CSS01"))


# %% UCB

iterations = 1000
samples = 2000
rewards_eps01_sam, proc_best_eps01_sam = \
                     experiment(lambda a, b: BanditEpsGreedy(a, b, eps=0.1),
                                get_normal_arms,
                                get_sample_actions_values,
                                iterations, samples)

rewards_eps01_css, proc_best_eps01_css = \
                     experiment(lambda a, b: BanditEpsGreedy(a, b, eps=0.1),
                                get_normal_arms,
                                get_css_actions_value,
                                iterations, samples)

rewards_ucb_c2_sam, proc_best_ucb_c2_sam = \
                     experiment(lambda a, b: UCB(a, b, c=2),
                                get_normal_arms,
                                get_sample_actions_values,
                                iterations, samples)

rewards_ucb_c2_css, proc_best_ucb_c2_css = \
                     experiment(lambda a, b: UCB(a, b, c=2),
                                get_normal_arms,
                                get_css_actions_value,
                                iterations, samples)

rewards_ucb_c1_css, proc_best_ucb_c1_css = \
                     experiment(lambda a, b: UCB(a, b, c=1),
                                get_normal_arms,
                                get_css_actions_value,
                                iterations, samples)

plot_multiple(range(iterations),
              (rewards_eps01_sam, "Eps01, sample"),
              # (rewards_eps01_css, "Eps01, css"),
              (rewards_ucb_c2_sam, "UCB c=2, sample")
              # (rewards_ucb_c2_css, "UCB c=2, css"))
              )

plot_multiple(range(iterations),
              (proc_best_eps01_sam, "Eps01, sample avg"),
              (proc_best_ucb_c2_sam, "UCB c=2, sample avg"))

# %% Gradient Bandit Algorithms


def get_normal_arms_mu4(number=10):
    arms_init = np.random.normal(4, 1, number)
    return [NormalArm(mu, 1) for mu in arms_init]


def get_pref01_actions_values(number=10, estimate=0.0):
    return [PreferenceActionValue(estimate) for _ in range(number)]


def get_pref04_actions_values(number=10, estimate=0.0):
    return [PreferenceActionValue(estimate, step_size=0.4)
            for _ in range(number)]


def zero_base():
    return ConstantStepSizeActionValue(step_size=0)


iterations = 1000
samples = 500
rewards_grad_alf01_b, proc_best_grad_alf01_b = \
                     experiment(lambda a, b: GradientBandit(a, b),
                                get_normal_arms_mu4,
                                get_pref01_actions_values,
                                iterations, samples)

rewards_grad_alf04_b, proc_best_grad_alf04_b = \
                     experiment(lambda a, b: GradientBandit(a, b),
                                get_normal_arms_mu4,
                                get_pref04_actions_values,
                                iterations, samples)
rewards_grad_alf01_nob, proc_best_grad_alf01_nob = \
                     experiment(lambda a, b: GradientBandit(a, b,
                                baseline=zero_base()),
                                get_normal_arms_mu4,
                                get_pref01_actions_values,
                                iterations, samples)
rewards_grad_alf04_nob, proc_best_grad_alf04_nob = \
                     experiment(lambda a, b: GradientBandit(a, b,
                                baseline=zero_base()),
                                get_normal_arms_mu4,
                                get_pref04_actions_values,
                                iterations, samples)


plot_multiple(range(iterations),
              (proc_best_grad_alf01_b, "Gradient, alfa=0.1, baseline"),
              (proc_best_grad_alf04_b, "Gradient, alfa=0.4, baseline"),
              (proc_best_grad_alf01_nob, "Gradient, alfa=0.1, no baseline"),
              (proc_best_grad_alf04_nob, "Gradient, alfa=0.1, no baseline"))

# %% Bandits comparison
# Bandit to compare - its parameter:
#   Eps-greedy - eps
#   greedy with optimist initialization, alfa = 0.1 - Q0 (initialization value)
#   UCB - c
#   gradient bandit - alfa

# Figure 2.6


def experiment_bandits(bandit_f, bandit_args, arms, action_values, iterations,
                       sessions):
    def experiment_b(b_args):
        # bb = bandit_f(arms(), action_values(), **b_args)
        # print(bb.eps)
        return experiment(lambda a, b: bandit_f(a, b, **b_args),
                          arms, action_values, iterations, sessions)

    return [sum(rews)/len(rews)
            for rews, proc_best in map(experiment_b, bandit_args)]


xs = [2**e for e in range(-7, 3)]
iterations = 1000
samples = 200
bandits_args = [{'eps': ep} for ep in xs[:6]]
rewards_epsg = experiment_bandits(BanditEpsGreedy,
                                  bandits_args,
                                  get_normal_arms,
                                  get_sample_actions_values,
                                  iterations, samples)
# plot_multiple(xs[:6], (rewards_epsg, "Eps-greedy, eps"))

bandits_args = [{'c': c} for c in xs[3:]]
rewards_ucb = experiment_bandits(UCB,
                                 bandits_args,
                                 get_normal_arms,
                                 get_sample_actions_values,
                                 iterations, samples)
# plot_multiple(xs[3:], (rewards_ucb, "UCB, c"))

# gradient


def get_pref_actions_values_al(alfa, number=10, estimate=0.0):
    return [PreferenceActionValue(estimate, step_size=alfa)
            for _ in range(number)]


rewards_grad = [experiment(GradientBandit, get_normal_arms,
                           lambda: get_pref_actions_values_al(alfa),
                           iterations, samples)[0] for alfa in xs[2:]]
rewards_grad = [sum(rews)/len(rews) for rews in rewards_grad]
# plot_multiple(xs[2:], (rewards_grad, "Gradient, alfa"))
# optimistic:


def get_css_actions_values_est(estimate, number=10):
    return get_css_actions_value(number, estimate=estimate)


rewards_greed_opt = [experiment(lambda a, b: BanditEpsGreedy(a, b, eps=0.0),
                                get_normal_arms,
                                lambda: get_css_actions_values_est(q0),
                                iterations, samples)[0] for q0 in xs[5:]]
rewards_greed_opt = [sum(rews)/len(rews) for rews in rewards_greed_opt]
# plot_multiple(xs[5:], (rewards_greed_opt, "Greedy optimistic, Q0"))


plt.subplots(figsize=(8, 6))
plt.semilogx(xs[5:], rewards_greed_opt, label="Greedy optimistic, Q0", basex=2)
plt.plot(xs[2:], rewards_grad, label="Gradient, alfa")
plt.plot(xs[3:], rewards_ucb, label="UCB, c")
plt.plot(xs[:6], rewards_epsg, label="Eps-greedy, eps")
leg = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.7)
plt.show()
