import nengo
import numpy as np
import time

DIM_XY=11
n_neurons = 2000#1000
tau = .01

def recurrent(x):

    #new_x = ((x[0] + .01*x[2]) + DIM_XY/2.0) % DIM_XY - DIM_XY/2.0
    #new_y = ((x[1] + .01*x[3]) + DIM_XY/2.0) % DIM_XY - DIM_XY/2.0
    
    new_x = (x[0] + .1*x[2] + 1) % 2 - 1
    new_y = (x[1] + .1*x[3] + 1) % 2 - 1
    
    #new_x = ((x[0] + x[2]) + DIM_XY/2.0) % DIM_XY - DIM_XY/2.0
    #new_y = ((x[1] + x[3]) + DIM_XY/2.0) % DIM_XY - DIM_XY/2.0

    return new_x, new_y, tau*x[2], tau*x[3]

model = nengo.Network(seed=13)
#model.config[nengo.Ensemble].neuron_type = nengo.Direct() #TODO: temp, just use direct for debugging
with model:
    velocity = nengo.Node([0,0]) # x-y velocity

    #posecells = nengo.Ensemble(n_neurons, dimensions=4, radius=DIM_XY/2.0)
    posecells = nengo.Ensemble(n_neurons, dimensions=4)

    nengo.Connection(velocity, posecells[2:])
    #nengo.Connection(posecells, posecells[:2], function=recurrent)
    nengo.Connection(posecells, posecells, function=recurrent, synapse=tau)


if __name__ == '__main__':

    print( "starting simulator..." )
    before = time.time()

    sim = nengo.Simulator(model)

    after = time.time()
    print( "time to build:" )
    print( after - before )

    print( "running simulator..." )
    before = time.time()

    while True:
        sim.step()
        time.sleep(0.0001)
