import numpy as np
import matplotlib.pyplot as plt
import our_functions as RL


def exponential_epsilon_decline(episode_number):
    return 1.4**(-episode_number) + .1
    

N_rats = 5
N_episodes = 25
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline,
                             [5, 15, 29])

plt.figure()
plt.plot(np.arange(1,N_episodes+1), steps_needed)
plt.title("Number of steps needed averaged over " + str(N_rats) + " rats")
plt.xlim([1,N_episodes])