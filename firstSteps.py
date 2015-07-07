import numpy as np
import matplotlib.pyplot as plt

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
    """
    N_a = W.shape[0]
    
    # Compute Q
    # The following admittedly looks cryptic, but I tested it and it works    
    sum_over_outputs = np.sum(W[:,:,:] * R[np.newaxis,:,:], axis=1)
    Q = np.sum(sum_over_outputs, axis=1)
    
    # Compute directions
    dirs = 2*np.pi*np.arange(1,N_a+1) / N_a    
    
    return Q, dirs
    


# test in_maze()
print("Testing in_maze()")
print("The following should be True : " + str(in_maze(50,50)))
print("The following should be False: " + str(in_maze(49,49)))
print("The following should be True : " + str(in_maze(50,49)))
print("The following should be False: " + str(in_maze(50,61)))
print("The following should be False: " + str(in_maze(-1,50)))
print("The following should be True : " + str(in_maze(55,30)))

# test input_layer()
state = [55,55,0]
centers = gen_place_centers()
R = 1/np.abs(np.log(input_layer(centers, state)[:,0])) * 100
plt.scatter(centers[:,0],centers[:,1], s=R, color="blue")
plt.scatter(state[0],state[1], s = np.amax(R)/10, color="black")
plt.title("Activity of neurons shown as size of dots (with logarithmic scale)")
    
# test output_layer()
W = np.ones((4,centers.shape[0],2))
W[2,:,:] = 2
R = input_layer(centers,state)
Q, directions = output_layer(R, W)



