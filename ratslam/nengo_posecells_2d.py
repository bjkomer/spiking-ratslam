# Prototype of Nengo posecell network, but in 2D only
#NOTE: nodes shouldn't be allowed to go negative, this might be breaking things
import nengo
import numpy as np

PC_DIM_XY=5#11
PC_DIM_TH=5#36
PC_W_E_DIM=7
PC_W_I_DIM=5

N_NEURONS=6

WEIGHT_DIM=3
WEIGHT_OFFSET = np.floor(WEIGHT_DIM/2.)

# Function used to ensure that the sum of the activity in the network is constant
NORMALIZATION_THRESHOLD = 1
def pseudonormalize(x):
    return NORMALIZATION_THRESHOLD - x#x[0]

# A kernel of weights for the connection strength between neurons of a given separation
weight_kernel = np.zeros((WEIGHT_DIM, WEIGHT_DIM, WEIGHT_DIM))

#TODO: use a mexican hat function to generate this
#FIXME: using something hard coded for now
weight_kernel = np.array([[-.1,.3,-.1],
                          [.3,.9,.3],
                          [-.1,.3,-.1]
                         ])
weight_kernel_left = np.array([[.3,-.1,-.1],
                               [.9,.3,.3],
                               [.3,-.1,-.1]
                              ])
weight_kernel_right = np.array([[-.1,-.1,.3],
                                [.3,.3,.9],
                                [-.1,-.1,.3]
                               ])
# Make the kernel have 0 mean so things don't explode
weight_kernel -= np.mean(weight_kernel)
weight_kernel -= .002 # global inhibition
print(weight_kernel)
model = nengo.Network('Posecells', seed=13)

with model:
    # Create a 2D array of ensembles for the posecells
    posecells = np.empty( (PC_DIM_XY, PC_DIM_XY), dtype=object)
    for i in range(PC_DIM_XY):
        for j in range(PC_DIM_XY):
            posecells[i,j] = nengo.Ensemble(n_neurons=N_NEURONS,
                                            dimensions=1,
                                            label="[%i,%i]"%(i,j)
                                           )
    
    # Connect all of the ensembles to the neighboring ones with specific weights
    for i in range(PC_DIM_XY):
        for j in range(PC_DIM_XY):
            # For each pose cell, connect it to its neighbors with a
            # specific weight based on the distance (and wrapping)
            for x in range(WEIGHT_DIM):
                for y in range(WEIGHT_DIM):

                    nengo.Connection(posecells[i, j], 
                                     posecells[(i + x - WEIGHT_OFFSET) % PC_DIM_XY,
                                               (j + y - WEIGHT_OFFSET) % PC_DIM_XY,
                                              ],
                                     transform = weight_kernel[x, y])

    # computes the sum of all activities, used for pseudonormalization
    # TODO: could also skip this population, and have this done directly through
    # connections, and possibly putting everything into one ensemble
    summation = nengo.Ensemble(n_neurons=N_NEURONS*3, dimensions=1)
    for i in range(PC_DIM_XY):
        for j in range(PC_DIM_XY):
            nengo.Connection(posecells[i, j], summation)
            nengo.Connection(summation, posecells[i, j],
                             function=pseudonormalize)

    # Computes the average position of the activity in the network 
    position = nengo.Ensemble(n_neurons=1, dimensions=2,
                              neuron_type=nengo.Direct())
    for i in range(PC_DIM_XY):
        for j in range(PC_DIM_XY):
            iw = (i-(PC_DIM_XY/2.0))/PC_DIM_XY
            jw = (j-(PC_DIM_XY/2.0))/PC_DIM_XY
            nengo.Connection(posecells[i, j], position, 
                             transform=[[iw],[jw]])
    #"""
    injector = nengo.Node([0,0,0])

    nengo.Connection(injector[0], posecells[0,0])
    nengo.Connection(injector[1], posecells[1,1])
    nengo.Connection(injector[2], posecells[2,2])
    #"""



