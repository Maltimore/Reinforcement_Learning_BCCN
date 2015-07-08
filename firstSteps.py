import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv

def gen_place_centers():
    """
    - generates center points for a T-maze of the following form:
        - 50 cm long and 10 cm wide vertical arm with
          two horizontal arms of the same shape, all connected by 10cm x 10 cm
          junction: (1 '=' : 10cm x 10 cm block)
          ===== = =====
                =
                =
                =
                =
                =
    - the origin is defined as the bottom left corner outside of the maze
    - most upper left center in the maze is then at x = 2.5, y = 60-2.5 = 57.5
    - spacing is equidistant with distance 5 cm
    """
    # x coordinates of horizontal arms
    horArmsX = np.arange(2.5,110,step=5)
    # x coordinates of vertical arm
    verArmX = np.repeat([52.5,57.5],10)
    # collect all x coordinates    
    xCoords = np.reshape(np.concatenate((horArmsX,horArmsX,verArmX),axis=0),(64,1))
    
    # y coordinates of rows in horizontal arms
    firstRowY = np.tile([57.5],22)
    secRowY = np.tile([52.5],22)
    # y coordinates of vertical arm
    verArmY = np.tile(np.arange(47.5,0,step=-5),2)
    # collect all y coordinates
    yCoords = np.reshape(np.concatenate((firstRowY,secRowY,verArmY),axis=0),(64,1))
    
    return np.hstack((xCoords,yCoords))
    
#centers = gen_place_centers()
#plt.plot(centers[:,0],centers[:,1],'ok')
#plt.title('Place field centers in the T-maze')

def in_maze(x, y):
    """Returns True if given coordinates are inside of the maze, else False"""
    # vertical arm up to the top
    if x>=50 and x<=60 and y>=0 and y<=60:
        return True
    elif  x>=0 and x<=50 and y>=50 and y<=60:
        return True
    elif  x>=60 and x<=110 and y>=50 and y<=60:
        return True
    else:
        return False


def in_pickup(x, y):
    """Returns True if given coordinates are inside the pickup area"""
    if x>=90 and x<=110 and y>=50 and y<=60:
        return True
    else:
        return False


def in_target(x, y):
    """Returns True if given coordinates are inside the target area"""
    if x>=0 and x<=20 and y>=50 and y<=60:
        return True
    else:
        return False
   
     
def input_layer(centers, state):
    """
    Computes the activity of the neurons in the input layer.
    
    This function copmutes the activity of all neurons in the input 
    layer. Sigma is assumed to be 5. This function does not perform
    a check whether the state is within the maze, this is assumed
    to be the case.
    
    Parameters:
    centers: array-like
             shape: N_neurons x 2 (x and y coordinates)
             The centers of the input neurons
    state:   array-like
             shape: 3 elements
             Containing the following elements: [x, y, alpha]
             
    Returns:
    R:       array-like
             shape: N_neurons x 2 (beta indices)
             An array containing the activity of all neurons,
             with element at position [j,beta] corresponding to 
             activity for the neuron j with population index beta.
    """
    sigma = 5    
    x = state[0]
    y = state[1]
    alpha = state[2]
    R = np.zeros((centers.shape[0],2))
    if alpha == 0:
        R[:,0] = np.exp(- ((centers[:,0] - x)**2 + (centers[:,1] - y)**2)
                        / (2 * sigma**2))
        R[:,1] = 0
    elif alpha == 1:
        R[:,0] = 0 
        R[:,1] = np.exp(- ((centers[:,0] - x)**2 + (centers[:,1] - y)**2)
                        / (2 * sigma**2))
    else:
        raise(Exception("input_layer function was called with alpha that " \
                         "is not 0 or 1"))
    return R


def output_layer(R, W):
    """
    Computes the activity of the neurons in the output layer.
    
    This function computes the activity of the neurons in the output 
    layer.
    
    Parameters:
    R:       array-like
             shape: N_input_neurons x 2 (beta indices)
             The activity of the neurons in the input layer.
    W:       array-like
             shape: N_output_neurons x N_input_neurons x beta_indices
             Connectivity matrix, follows format [input_neuron,
             output_neuron, beta] (corresponds to [a,j,beta] from
             the problem sheet)
    
    Returns:
    Q:       array-like
             shape: N_output_neurons
             The activity of the nreurons in the output layer.
    dirs:    array-like
             shape: N_output_neurons
             The directions that the output neurons correspond to.
             Unit is radians.
    """
    N_a = W.shape[0]
    
    # Compute Q
    # The following admittedly looks cryptic, but I tested it and it works    
    sum_over_outputs = np.sum(W[:,:,:] * R[np.newaxis,:,:], axis=1)
    Q = np.sum(sum_over_outputs, axis=1)
    
    # Compute directions
    dirs = 2*np.pi*np.arange(1,N_a+1) / N_a    
    return Q, dirs


def choose_action(Q, directions, epsilon):
    """
    Choose an action.
    
    Function returns an action that the mouse takes. The action
    is the x and y value of the movement that the rat will perform.
    
    Parameters:
    Q:       array-like
             shape: N_output_neurons
             The activity of the nreurons in the output layer.
    dirs:    array-like
             shape: N_output_neurons
             The directions that the output neurons correspond to.
             Unit is radians.
    epsilon: float
             The epsilon value determines how many times (relatively)
             the mouse chooses an exploratory action instead of the
             reward maximizing action.
             
    Returns:
    a:       int
             the action as index of Q values
    step:    [int, int]
             The step [x value, y value] that the mouse will take.
    """
    # set mean and standard deviation values for step
    mean = 3
    sd = 1.5
    # determine stepsize
    stepsize = np.random.normal(loc=mean, scale=sd)
    
    # choose action
    if np.random.uniform() < epsilon:
        # take exploratory action
        a = np.random.randint(Q.shape[0])
        angle = directions[a]
    else:
        # take exploitative action
        a = np.argmax(Q)
        angle = directions[a]
    
    step = stepsize * np.array([np.cos(angle), np.sin(angle)])
    
    return a, step


def get_reward(state):
    """ Compute reward for given state."""    
    
    x = state[0]
    y = state[1]
    alpha = state[2]
    
    if in_target(x, y) and alpha == 1:
        return 20
    elif not in_maze(x, y):
        return -1
    else:
        return 0


def update_state(state, step):
    """
    Does action step.

    Does action step and returns new state. Also checks whether
    a reward is received. In this exercise, if a reward is 
    received, this means that a terminal state has been reached.
    
    Parameters:
    state:     array-like
               shape: 3 elements
               Containing the following elements: [x, y, alpha]
    step:      [int, int]
               The step [x value, y value] that the mouse will take.  
               shape: 3 elements
               Containing the following elements: [x, y, alpha]    
               
    Returns:
    new_state: array-like
               shape: 3 elements
               Containing the following elements: [x, y, alpha]
    reward:    int
               reward that was received for performing last step.
    """
    alpha = state[2]    
    # calculate new position    
    x = state[0] + step[0]
    y = state[1] + step[1]
    new_state = [x, y, alpha]
    
    # update alpha if necessary
    if alpha == 0:
        if in_pickup(x, y):
            new_state[2] = 1
    
    # check if final goal is reached
    if alpha == 1 and in_target(x, y):
        return new_state, 20
    
    # check if the rat ran into a wall
    if not in_maze(x, y):
        return new_state, -1
    else:
        # if the rat didn't run into a wall and nothing else happened
        # return 0 reward
        return new_state, 0

def update_weights(R_t, Q_t, action_t, reward, Q_tp,
                   W, eta=.01, gamma=.95, lambda_=1):
    """
    Update weights according to SARSA.
    
    Update the weights of the neuron from input layer to output layer
    according to the SARSA rule.
    
    Parameters:
    R_t:        activity of inputs
    Q_t:        Q value of state_t, action_t pair
    action_t:   int
                action taken at timestep t
    reward:     int
                reward received after choosing action_t at timestep
                t and being in state_t
    Q_tp:       Q value of state_tp, action_tp pair
    W:          array-like
                shape: N_output_neurons x N_input_neurons x beta_indices
                Connectivity matrix, follows format [input_neuron,
                output_neuron, beta] (corresponds to [a,j,beta] from
                the problem sheet)
    eta:        float
                Learning rate
    gamma:      float
                Discount factor
    lambda_:    float
                Decay rate
    """
    delta_Q = eta * (reward + gamma*Q_tp - Q_t)
    delta_W = delta_Q * pinv(R_t)
    W[action_t,:,:] = W[action_t,:,:] + delta_W.T

    return W
    
    
# test in_maze()
#print("Testing in_maze()")
#print("The following should be True : " + str(in_maze(50,50)))
#print("The following should be False: " + str(in_maze(49,49)))
#print("The following should be True : " + str(in_maze(50,49)))
#print("The following should be False: " + str(in_maze(50,61)))
#print("The following should be False: " + str(in_maze(-1,50)))
#print("The following should be True : " + str(in_maze(55,30)))

# test input_layer()
#state = [55,55,0]
#centers = gen_place_centers()
#R = 1/np.abs(np.log(input_layer(centers, state)[:,0])) * 100
#plt.scatter(centers[:,0],centers[:,1], s=R, color="blue")
#plt.scatter(state[0],state[1], s = np.amax(R)/10, color="black")
#plt.title("Activity of neurons shown as size of dots (with logarithmic scale)")
    
# test output_layer()
#W = np.ones((4,centers.shape[0],2))
#W[0,:,:] = 2
#R = input_layer(centers,state)
#Q, directions = output_layer(R, W)

#position = np.array([55.,0.])
#for i in range(20):
#    position += choose_action(Q, directions, .4)
#    plt.scatter(position[0], position[1])


# implement SARSA
N_a = 4
centers = gen_place_centers()
W = np.random.normal(size=(N_a, centers.shape[0], 2))
W = np.zeros((N_a, centers.shape[0], 2))
W_before = W.copy()
epsilon = 1
obencounter = 0
untencounter = 0
for episode in np.arange(2000):

    # initialize s    
    state_t = [55,0,0]
    
    # choose a from s using policy
    R_t = input_layer(centers, state_t)
    Q_t, directions = output_layer(R_t, W)
    a_t, step_t = choose_action(Q_t, directions, epsilon)
    
    
    # repeat (for each step of episode)
    non_terminal = True
    steps_needed = 0
    states = [state_t]



    while non_terminal:
        if steps_needed > 10000:
            print("needed more than 10.000 steps")
            if state_t[2] == 1:
                print("pickup area had been reached")
            elif state_t[2] == 0:
                print("pickup area had not been reached")
            break
        steps_needed += 1
        state_tp, r = update_state(state_t, step_t)
        states.append(state_tp)
        # choose a_tp from state_tp using policy
        R_tp = input_layer(centers, state_tp)
        Q_tp, directions = output_layer(R_tp, W)
        a_tp, step_tp = choose_action(Q_tp, directions, epsilon)
        
        if r != 0:
            # set flag to end loop
            if r == 20:
                non_terminal = False
            # update weights according to SARSA
            W = update_weights(R_t, Q_t[a_t], a_t, r, Q_tp[a_tp], W)

            # if the animal broke through the wall, set it back
            # to where it was
            if r == -1:
                state_tp = state_t
            
        # set a_tp, step_tp, Q_tp, R_tp to currenct values
        state_t = state_tp
        step_t = step_tp
        Q_t = Q_tp
        a_t = a_tp
        R_t = R_tp
        
    if r == 20 or steps_needed > 10000:
        print("steps needed to reach goal: " + str(steps_needed))
        states = np.array(states)
        plt.figure()
        plt.plot(centers[:,0],centers[:,1],'ok')
        plt.plot(states[:,0], states[:,1])
        plt.title("Steps to reach goal: " +str(steps_needed))
    
    epsilon = 1.1**(-episode) + .1


# for debugging
states = np.array(states)
plt.plot(centers[:,0],centers[:,1],'ok')
plt.plot(states[:,0], states[:,1])









