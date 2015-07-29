import numpy as np
import matplotlib.pyplot as plt
import our_functions as RL
plt.style.use('ggplot')

def plot_learning_curve(steps_needed, N_rats, N_episodes, plottitle):
    plt.figure()
    plt.plot(np.arange(1,N_episodes+1), steps_needed)
    plt.title(plottitle)
    plt.xlim([1,N_episodes])
    plt.xlabel("Number of episode")
    plt.ylabel("Number of steps needed")
    
def exponential_epsilon_decline(episode_number):
    return 1.5**(-episode_number) + .1
    
def reverse_exponential_epsilon(episode_number):
    return 1 - 1.5**(-episode_number) + .1
    

###############################################################################
# Point 1: Averaging over 10 rats and plot the learning curve
N_rats = 10
N_episodes = 25
plottitle = "Needed steps averaged over " + str(N_rats) + " rats" \
            " using standard parameters"
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline)
plot_learning_curve(steps_needed, N_rats, N_episodes, plottitle)

###############################################################################
# Point 2: Plotting the navigation map of the animal after different numbers of
# trials
N_rats = 1
N_episodes = 10
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline,
                             [2, 6, 10])
                             
###############################################################################
# Point 3: Comparing for different values of eligibility trace
N_rats = 5
N_episodes = 30
lambda_ = 0
plottitle = "Needed steps averaged over " + str(N_rats) + " rats" \
            " with lambda = " + str(lambda_)
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline, \
                             lambda_=lambda_)
plot_learning_curve(steps_needed, N_rats, N_episodes, plottitle)

lambda_ = .95
plottitle = "Needed steps averaged over " + str(N_rats) + " rats" \
            " with lambda = " + str(lambda_)
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline, \
                             lambda_=lambda_)
plot_learning_curve(steps_needed, N_rats, N_episodes, plottitle)

###############################################################################
# Point 4: Varying the time course of the exploration / exploitation parameter
steps_needed = RL.run_trials(N_rats, N_episodes, reverse_exponential_epsilon)
plottitle = "Needed steps averaged over " + str(N_rats) + " rats" \
            " with decreasing epsilon"
plot_learning_curve(steps_needed, N_rats, N_episodes, plottitle)

###############################################################################
# Point 5: Varying the number of directions
N_rats = 1
N_episodes = 30
N_a_vec = [4, 5, 6, 7, 8]
steps_needed_mat = np.zeros((len(N_a_vec), N_episodes))
for idx, N_a in enumerate(N_a_vec):
    steps_needed_mat[idx,:] = RL.run_trials(N_rats, N_episodes, \
                                            exponential_epsilon_decline, \
                                            N_a=N_a)
