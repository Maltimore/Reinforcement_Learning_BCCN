import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from mpl_toolkits.mplot3d import axes3d
#from _functions import learn_rat

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


#def newfun(state1, state2):
    


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
    if int(alpha) == 0:
        R[:,0] = np.exp(- ((centers[:,0] - x)**2 + (centers[:,1] - y)**2)
                        / (2 * sigma**2))
        R[:,1] = 0
    elif int(alpha) == 1:
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
    dirs = 2*np.pi*np.arange(1, N_a+1) / N_a    
    return Q, dirs


def choose_action(Q, directions, epsilon, mean=3, sd=1.5):
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

    # determine stepsize
    if sd == 0:
        stepsize = mean
    else:
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
    
    if in_target(x, y) and int(alpha) == 1:
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
    if int(alpha) == 0:
        if in_pickup(x, y):
            new_state[2] = 1
    
    # check if final goal is reached
    if int(alpha) == 1 and in_target(x, y):
        return new_state, 20
    
    # check if the rat ran into a wall
    if not in_maze(x, y):
        return new_state, -1
    else:
        # if the rat didn't run into a wall and nothing else happened
        # return 0 reward
        return new_state, 0

    
def update_weights_eligibility(W, E, Q_t, a_t, r, Q_t1,
                               eta=.05, gamma=.95, lambda_=.9):
    """
    Update weights according to SARSA(Lambda).
    
    Update the weights of the neuron from input layer to output layer
    according to the SARSA(Lambda) rule.
    
    Parameters:
    eligibility_history: list of lists
                The "higher" dimension holds the history of the chosen
                actions and states, and every list of the list holds 
                the five parameters [R_t, Q_t, action_t, reward, Q_t1]
                (see the "normal" update_weights function).
    W:          array-like
                shape: N_output_neurons x N_input_neurons x beta_indices
                Connectivity matrix, follows format [input_neuron,
                output_neuron, beta] (corresponds to [a,j,beta] from
                the problem sheet)
    eta:        float, optional
                Learning rate
    gamma:      float, optional
                Discount factor
    lambda_:    float, optional
                memory discount
    """    
    
    delta_Q = r + gamma*Q_t1 - Q_t
    delta_W = eta * delta_Q * E
    W = W + delta_W
    return W


def reset_mouse(old_state, new_state):    
    def is_between(a, b, c):
        a[0], a[1] = round(a[0], 3), round(a[1], 3)
        b[0], b[1] = round(b[0], 3), round(b[1], 3)
        c[0], c[1] = round(c[0], 3), round(c[1], 3)
        return (np.isclose((b[0] - a[0]) * (c[1] - a[1]), (c[0] - a[0]) * (b[1] - a[1]), .001, .001) and
            (((a[0] <= c[0]) and (b[0] >= c[0])) or ((a[0] >= c[0]) and (b[0] <= c[0]))) and
            (((a[1] <= c[1]) and (b[1] >= c[1])) or ((a[1] >= c[1]) and (b[1] <= c[1]))))
    
    def intersection(q0, q1, p0, p1):
        dy = q0[1] - p0[1]
        dx = q0[0] - p0[0]
        lhs0 = [-dy, dx]
        rhs0 = p0[1] * dx - dy * p0[0]
        
        dy = q1[1] - p1[1]
        dx = q1[0] - p1[0]
        lhs1 = [-dy, dx]
        rhs1 = p1[1] * dx - dy * p1[0]
        
        a = np.array([lhs0, 
                      lhs1])
        
        b = np.array([rhs0, 
                      rhs1])
        try:
            px = np.linalg.solve(a, b)
        except:
            px = np.array([np.nan, np.nan])
        return px
    old_state = np.asarray(old_state)
    new_state = np.asarray(new_state)
    old_pos = old_state[:2]
    new_pos = new_state[:2]

    startpoints = np.array([[0, 60],
                            [0, 50],
                            [60, 50],
                            [50, 0],
                            [0, 50],
                            [50, 0],
                            [60, 0],
                            [110, 50]])
    endpoints = np.array([[110, 60],
                          [50, 50],
                          [110, 50],
                          [60, 0],
                          [0, 60],
                          [50, 50],
                          [60, 50],
                          [110, 60]])
    for i in np.arange(8):   
        px = intersection(startpoints[i,:], old_pos, endpoints[i,:], new_pos)
        if is_between(startpoints[i,:], endpoints[i,:], px) and \
           is_between(old_pos, new_pos, px):
            reset_to = px + .1 * (old_pos - px)
            return np.hstack((reset_to, old_state[2])), np.hstack((px, old_state[2]))
#    print("no line was sected apparently")
#    print("Estimated intersection point: " + str(px))
#    print("Old position was: " + str(old_pos))
#    print("New position is:  " + str(new_pos))
    return old_state, np.hstack(([0,0], old_state[2]))





N_a = 4
centers = gen_place_centers()
W = np.random.normal(size=(N_a, centers.shape[0], 2))
W = np.zeros((N_a, centers.shape[0], 2))
E = np.zeros(W.shape)
gamma = .95
lambda_ = .90
epsilon = 1
break_after_steps = 20000
total_steps = []

learned_flag = False
ratIsLost_flag = False

for episode in np.arange(50):

    # initialize s    
    state_t = [55,0,0]
    # initialize variables
    non_terminal = True
    steps_needed = 0
    states = []    
    bumps = []
    E = np.zeros(W.shape)

    # choose a from s using policy
    R_t = input_layer(centers, state_t)
    Q_t, directions = output_layer(R_t, W)
    a_t, step_t = choose_action(Q_t, directions, epsilon)
    
    # save all the steps, Q-values, weights and e-values
    step_history = []
    q_history = []
    w_history = []
    e_history = []
    
    # repeat (steps of the episode)
    while non_terminal:

        if steps_needed >= break_after_steps:
            # if more than break_after_steps steps were needed, break (because the mouse
            # most likely got stuck)
            break
        
        steps_needed += 1
        
        if learned_flag and steps_needed>200:
            ratIsLost_flag = True
            break
        # take action a (defined by step_t)        
        state_t1, r = update_state(state_t, step_t)
           

        # choose a_t1 from state_t1 using policy
        R_t1 = input_layer(centers, state_t1)
        Q_t1, directions = output_layer(R_t1, W)
        a_t1, step_t1 = choose_action(Q_t1, directions, epsilon)


        # Distinguish the three cases of reward (-1, 0 and 20)
        if r == -1 :
            # if the reward was -1, the mouse crashed into the wall. In this
            # case, Q_t1 is zero. Also, do not append the state to the history
            # of states (needed for plotting later)
            eligibility = [np.nan, Q_t[a_t], a_t, r, 0]
            # Reset mouse
            state_t1, bump = reset_mouse(state_t, state_t1)
            bumps.append(bump)            

        elif r == 0:         
           eligibility = [np.nan, Q_t[a_t], a_t, r, Q_t1[a_t1]]
        elif r == 20:
            # In the case that the reward is 20, the trial is over and Q_t1 is
            # therefore zero. Also set the flag to end the loop.
            eligibility = [np.nan, Q_t[a_t], a_t, r, 0]
            non_terminal = False

        # it seems weird that update_weight_eligibility returns the eligibility
        # history (although it gets it as an argument), but that is because
        # the eligibility history is "trimmed" to a useful length (see docstring
        # of the function)
        E = E * gamma * lambda_
        E[a_t,:,:] += R_t
        eligibility[0] = E
        W = update_weights_eligibility(W, *eligibility)
        

        if r == -1:        
            # choose a new step if it has bummed into a wall
            R_t1 = input_layer(centers, state_t1)
            Q_t1, directions = output_layer(R_t1, W)
            a_t1, step_t1 = choose_action(Q_t1, directions, epsilon)
        
        # save values in history
        step_history.append(step_t)
        q_history.append(Q_t)
        w_history.append(W)
        e_history.append(E)        
        
        # set a_t1, step_t1, Q_t1, R_t1 to currenct values
        state_t = state_t1
        step_t = step_t1
        Q_t = Q_t1
        a_t = a_t1
        R_t = R_t1
    
        
        # save state to plot later
        states.append(state_t)
            
    if ratIsLost_flag:
        break
    
    if r == 20 or steps_needed >= break_after_steps:
        print("steps needed: " + str(steps_needed))
        states = np.array(states)
        plt.figure()
        plt.plot(centers[:,0],centers[:,1],'ok')
        plt.plot(states[:,0], states[:,1])
        plt.title("Steps: " +str(steps_needed) + " epsilon: " + str(epsilon))
        bumps = np.array(bumps)
        if len(bumps) > 0:
            plt.scatter(bumps[:,0], bumps[:,1], s = 100, c = 100 * bumps[:,2], edgecolor="")


#        for alpha in [0,1]:
#            plt.figure()
#            arrowvec = np.zeros(centers.shape)
#            for idx, coordinate in enumerate(centers):
#                state = np.array([coordinate[0], coordinate[1], alpha])
#                R = input_layer(centers, state)
#                Q, direction = output_layer(R, W)
#                _, arrowvec[idx,:] = choose_action(Q, directions, 0, mean=.6, sd=0)
#            plt.figure()
#            plt.quiver(centers[:,0], centers[:,1], arrowvec[:,0], arrowvec[:,1])
#            plt.title("Arrows represent choices for greedy policy and alpha = " + str(alpha))

        if steps_needed < 100:
            learned_flag = True
        
        if steps_needed < 40 or steps_needed >= break_after_steps:
           break
    
    if episode == 30:
        break
    
    epsilon = 1.5**(-episode-1) + .1
    
    total_steps.append(steps_needed)






#rat_steps = np.zeros((10,50))
#for rat in np.arange(10):
#    rat_steps[rat,:] = learn_rat()
#
#plt.plot(np.mean(rat_steps,axis=0))
#plt.plot(rat_steps)
#plt.ylim(0,1000)




# vector field
#for alpha in [0,1]:
#    plt.figure()
#    arrowvec = np.zeros(centers.shape)
#    for idx, coordinate in enumerate(centers):
#        state = np.array([coordinate[0], coordinate[1], alpha])
#        R = input_layer(centers, state)
#        Q, direction = output_layer(R, W)
#        _, arrowvec[idx,:] = choose_action(Q, directions, 0, mean=.6, sd=0)
#    plt.figure()
#    plt.quiver(centers[:,0], centers[:,1], arrowvec[:,0], arrowvec[:,1])
#    plt.title("Arrows represent choices for greedy policy and alpha = " + str(alpha))

#weights_pop1_2D = {'dir0':np.zeros((60,110)),'dir1':np.zeros((60,110)),\
#              'dir2':np.zeros((60,110)),'dir3':np.zeros((60,110))}
#weights_pop2_2D = {'dir0':np.zeros((60,110)),'dir1':np.zeros((60,110)),\
#              'dir2':np.zeros((60,110)),'dir3':np.zeros((60,110))}
#weights_2D = {'pop1':weights_pop1_2D, 'pop2':weights_pop2_2D}
#
#for idx in np.arange(centers.shape[0]):
#    xCoo = int(np.floor(centers[idx,0]))
#    yCoo = int(np.floor(centers[idx,1]))
#    for i in range(4):
#        for j in range(2):
#            weights_2D['pop1']['dir'+str(i)][yCoo,xCoo] = W[i,idx,0]
#            weights_2D['pop2']['dir'+str(i)][yCoo,xCoo] = W[i,idx,1]
#
#direction_strings = ['up','left','down','right']
#
#for pop in weights_2D.keys():
#    plt.figure(figsize=(16,12))
#    for i in range(4):
#        plt.subplot(2,2,i+1)    
#        plt.imshow(weights_2D[pop]['dir'+str(i)]*10,origin='lower',clim=[-1,1])
#        plt.title('Direction: ' + direction_strings[i])
#        plt.colorbar()
#    plt.suptitle(pop)
#
## i played around a bit with 3D plots but the result is not that satisfying
#from matplotlib import cm
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#xgrid = np.arange(60)
#ygrid = np.arange(110)
#x,y = np.meshgrid(ygrid,xgrid)
#ax.plot_wireframe(x,y,weights_2D['pop1']['dir0'],alpha=0.3)
#cset = ax.contour(x, y, weights_2D['pop1']['dir0'], zdir='z', offset=-2, cmap=cm.coolwarm)
#ax.set_zlabel('Z')
#ax.set_zlim(-2, 2)






