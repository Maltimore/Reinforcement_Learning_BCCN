import numpy as np
import matplotlib.pyplot as plt
import our_functions as RL


def exponential_epsilon_decline(episode_number):
    return 1.5**(-episode_number) + .1
    

###############################################################################
# Point 1: Averaging over 10 rats and plot the learning curve
N_rats = 2
N_episodes = 25
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline, lambda_=0)

plt.figure()
plt.plot(np.arange(1,N_episodes+1), steps_needed)
plt.title("Number of steps needed averaged over " + str(N_rats) + " rats")
plt.xlim([1,N_episodes])
plt.xlabel("Number of episode")
plt.ylabel("Number of steps needed")

###############################################################################
# Point 2: Plotting the navigation map of the animal after different numbers of
# trials
N_rats = 1
N_episodes = 11
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline,
                             [2, 6, 10])