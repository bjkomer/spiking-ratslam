# Prototype of Nengo posecell network, but in 2D only
import nengo
import numpy as np

PC_DIM_XY=5#11
PC_DIM_TH=5#36
PC_W_E_DIM=7
PC_W_I_DIM=5

N_NEURONS=4

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
weight_kernel = np.array([[[.1,.3,.1],[.3,.5,.3],[.1,.3,.1]],
                          [[.3,.5,.3],[.5,.9,.5],[.3,.5,.3]],
                          [[.1,.3,.1],[.3,.5,.3],[.1,.3,.1]]],
                        )
print(np.mean(weight_kernel))
weight_kernel -= np.mean(weight_kernel)

model = nengo.Network('Posecells', seed=13)

with model:
    # Create a 3D array of ensembles for the posecells
    posecells = np.empty( (PC_DIM_XY, PC_DIM_XY, PC_DIM_TH), dtype=object)
    for i in range(PC_DIM_XY):
        for j in range(PC_DIM_XY):
            for k in range(PC_DIM_TH):
                posecells[i,j,k] = nengo.Ensemble(n_neurons=N_NEURONS,
                                                  dimensions=1)
    
    # Connect all of the ensembles to the neighboring ones with specific weights
    for i in range(PC_DIM_XY):
        for j in range(PC_DIM_XY):
            for k in range(PC_DIM_TH):
                # For each pose cell, connect it to its neighbors with a
                # specific weight based on the distance (and wrapping)
                for x in range(WEIGHT_DIM):
                    for y in range(WEIGHT_DIM):
                        for z in range(WEIGHT_DIM):

                            nengo.Connection(posecells[i, j, k], 
                                             posecells[(i + x - WEIGHT_OFFSET) % PC_DIM_XY,
                                                       (j + y - WEIGHT_OFFSET) % PC_DIM_XY,
                                                       (k + z - WEIGHT_OFFSET) % PC_DIM_TH],
                                             transform = weight_kernel[x, y, z])

    # computes the sum of all activities, used for pseudonormalization
    # TODO: could also skip this population, and have this done directly through
    # connections, and possibly putting everything into one ensemble
    summation = nengo.Ensemble(n_neurons=N_NEURONS*3, dimensions=1)
    for i in range(PC_DIM_XY):
        for j in range(PC_DIM_XY):
            for k in range(PC_DIM_TH):
                nengo.Connection(posecells[i, j, k], summation)
                nengo.Connection(summation, posecells[i, j, k],
                                 function=pseudonormalize)

    # Computes the average position of the activity in the network 
    position = nengo.Ensemble(n_neurons=1, dimensions=3,
                              neuron_type=nengo.Direct())
    for i in range(PC_DIM_XY):
        for j in range(PC_DIM_XY):
            for k in range(PC_DIM_TH):
                iw = (i-(PC_DIM_XY/2.0))/PC_DIM_XY
                jw = (j-(PC_DIM_XY/2.0))/PC_DIM_XY
                kw = (k-(PC_DIM_TH/2.0))/PC_DIM_TH
                nengo.Connection(posecells[i, j, k], position, 
                                 transform=[[iw],[jw],[kw]])

    injector = nengo.Node([0])

    nengo.Connection(injector, posecells[2,2,2])



