# sutton-reinforcement-learning-intro
Excersises and figures from "Reinforcement Learning, An Introduction" by Sutton and Barto

Some of the figures:

![Figure 2.6](/images/figure_2_6.png)

Figure 2.6

It doesn't look exactly like in Sutton (unlike all the other figures from chapter 2 which I got almost ideally the same). It turns out that the average return from those bandits over 1000 iterations have pretty big variance. I did average (the average reward) over 200 runs for each bandit and each parameter (I bet Sutton did it over 2000 runs, but I didn't want to wait this long..).

![Figure 4.2](/images/figure_4_2.png)

Figure 4.2

It looks reasonably similar but not exactly the same. There might be some subtle differences in how I interpret the rules. I see no point in spending additional hours on looking for them. (Of course there could also be some bugs in my code.) It is very suspicious that lower right corner in Sutton's plots is -4 rather then -5..

![Figure 4.3](/images/fig_4_3_acc_diff.png)

Figure 4.3 On the left is the policy without paying attention to precision and on the right after rounding q-function to 4 decimal places.

![Figure 4.3*](/images/fig_4_3_q-function.png)

More interesting for figure 4.3 is to see the values of q function..
