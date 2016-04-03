import nengo
import numpy as np
import time

DIM_XY=11
n_neurons = 5000#1000
tau = .1

def integrate(x):

    return [(x[0] + x[1]*x[4]*tau)*1.1, (x[1] - x[0]*x[4]*tau)*1.1, 
            (x[2] + x[3]*x[5]*tau)*1.1, (x[3] - x[2]*x[5]*tau)*1.1]

def decoding(x):

    return np.arctan2(x[0], x[1]), np.arctan2(x[2], x[3])

model = nengo.Network(seed=13)
#model.config[nengo.Ensemble].neuron_type = nengo.Direct() #TODO: temp, just use direct for debugging
with model:


    velocity = nengo.Node([0,0]) # x-y velocity
    position = nengo.Ensemble(1000, dimensions=2, neuron_type=nengo.Direct())
    posecells = nengo.Ensemble(n_neurons, dimensions=6, radius=1)

    #nengo.Connection(velocity, posecells, transform = [[tau,0],[0,tau]])
    nengo.Connection(velocity, posecells[-2:])
    nengo.Connection(posecells, posecells[:-2], function=integrate)
    nengo.Connection(posecells, position, function=decoding)
    
    initial_stim = nengo.Node(lambda t: [1,1] if t < 0.15 else [0,0])
    nengo.Connection(initial_stim, posecells[[0,2]])


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
