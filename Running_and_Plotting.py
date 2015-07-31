# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import our_functions as RL
plt.style.use('ggplot')

import matplotlib as mpl
mpl.rcParams['font.size'] = 20
mpl.rcParams['lines.linewidth']= 3

def plot_learning_curve(steps_needed, N_rats, N_episodes, plottitle):
    plt.figure(figsize=(12,8))
    plt.plot(np.arange(1,N_episodes+1), steps_needed)
    #plt.title(plottitle)
    plt.xlim([1,N_episodes])
    plt.xlabel("Number of episode")
    plt.ylabel("Number of steps needed")
    
def plot_multiple_learning_curve(steps_needed_list, labels, N_rats, N_episodes, plottitle):
    plt.figure(figsize=(16,12))
    for i in range(len(steps_needed_list)):
        plt.plot(np.arange(1,N_episodes+1), steps_needed_list[i],label=labels[i])
    plt.legend()
    #plt.title(plottitle)
    plt.xlim([1,N_episodes])
    plt.xlabel("Number of episode")
    plt.ylabel("Number of steps needed")
    
def exponential_epsilon_decline(episode_number):
    return 1.5**(-episode_number) + .1
    
def reverse_exponential_epsilon(episode_number):
    return 1 - 1.5**(-episode_number) + .1
    
def linear_epsilon(episode_number):
    return 0.85 - 0.1*episode_number + .1
    
def static_epsilon(value):
    def func(episode):
        return value
    return func
    

################################################################################
## Point 1: Averaging over 10 rats and plot the learning curve
N_rats = 10
N_episodes = 25
plottitle = "Needed steps averaged over " + str(N_rats) + " rats" \
            " using standard parameters"
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline)
plot_learning_curve(steps_needed, N_rats, N_episodes, plottitle)
plt.savefig('plots/learning_curve_standard.png',format='png')

###############################################################################
# Point 2: Plotting the navigation map of the animal after different numbers of 
# trials
N_rats = 1
N_episodes = 25
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline,
                             [2, 6, 25], weight_decay=False)
                             
################################################################################
## Point 3: Comparing for different values of eligibility trace
N_rats = 5
N_episodes = 30
lambda_ = 0

steps_needed_list = []
labels = []
plottitle = "Needed steps averaged over " + str(N_rats) + " rats" \
            " with lambda = " + str(lambda_)
label = '$\lambda$ = ' +str(lambda_)
labels.append(label)
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline, \
                             lambda_=lambda_)
steps_needed_list.append(steps_needed)


lambda_ = .95
plottitle = "Needed steps averaged over " + str(N_rats) + " rats" \
            " with lambda = " + str(lambda_)
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline, \
                             lambda_=lambda_)
label = '$\lambda$ = ' +str(lambda_)
labels.append(label)
steps_needed_list.append(steps_needed)
plot_multiple_learning_curve(steps_needed_list,labels, N_rats, N_episodes, plottitle)
plt.savefig('plots/learning_curves_lambda.png',format='png')
################################################################################
## Point 4: Varying the time course of the exploration / exploitation parameter
N_rats = 10
N_episodes = 10
steps_needed_list = []
labels = []

plottitle = "Needed steps averaged over " + str(N_rats) + " rats" \
            " with increasing epsilon"
            
# Compare dynamic time courses for epsilon
steps_needed = RL.run_trials(N_rats, N_episodes, exponential_epsilon_decline)
steps_needed_list.append(steps_needed)
label = 'exp. decrease of $\epsilon$'
labels.append(label)
steps_needed = RL.run_trials(N_rats, N_episodes, reverse_exponential_epsilon)
steps_needed_list.append(steps_needed)
label = 'exp. increase of $\epsilon$'
labels.append(label)
steps_needed = RL.run_trials(N_rats, N_episodes, linear_epsilon)
steps_needed_list.append(steps_needed)
label = 'linear decrease of $\epsilon$'
labels.append(label)

plot_multiple_learning_curve(steps_needed_list, labels, N_rats, N_episodes, plottitle)
plt.savefig('plots/learning_curve_epsilonDynamic.png',format='png')

# compare static values for epsilon
steps_needed_list = []
labels = []
steps_needed = RL.run_trials(N_rats, N_episodes, static_epsilon(0.1))
steps_needed_list.append(steps_needed)
label = 'static $\epsilon$=0.1'
labels.append(label)
steps_needed = RL.run_trials(N_rats, N_episodes, static_epsilon(0.5))
steps_needed_list.append(steps_needed)
label = 'static $\epsilon$=0.5'
labels.append(label)
steps_needed = RL.run_trials(N_rats, N_episodes, static_epsilon(0.75))
steps_needed_list.append(steps_needed)
label = 'static $\epsilon$=0.75'
labels.append(label)

plot_multiple_learning_curve(steps_needed_list, labels, N_rats, N_episodes, plottitle)
plt.savefig('plots/learning_curve_epsilonStatic.png',format='png')

################################################################################
## Point 5: Varying the number of directions
#N_rats = 1
#N_episodes = 30
#N_a_vec = [4, 5, 6, 7, 8]
#steps_needed_mat = np.zeros((len(N_a_vec), N_episodes))
#for idx, N_a in enumerate(N_a_vec):
#    steps_needed_mat[idx,:] = RL.run_trials(N_rats, N_episodes, \
#                                            exponential_epsilon_decline, \
#                                            N_a=N_a)
#plot_learning_curve(steps_needed_mat.T, N_rats, N_episodes, plottitle)
#labels = ['$N_a$ = ' +str(N_a_vec[i]) for i in range(len(N_a_vec))]
#plt.legend(labels)
#plt.savefig('plots/learning_curves_Na.png',format='png')