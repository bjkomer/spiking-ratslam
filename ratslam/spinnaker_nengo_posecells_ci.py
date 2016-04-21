# Posecell Network using function space representation
# TODO: inject some energy in the middle initially to get it started
# TODO: on the inject function, also output a magnitude, which decays over time

import nengo
import numpy as np
import time
import nengo_spinnaker

from nengo_posecell_network import NengoPosecellNetwork

n_neurons=500#1000#2000
tau = .1

def integrate(x):

    return (x[0] + x[1]*x[2]*tau)*1.1, (x[1] - x[0]*x[2]*tau)*1.1

def decoding(x):

    return np.arctan2(x[0], x[1]) / np.pi

def stim_input(x):

    #TODO: doublecheck that cos and sin are correct
    return np.cos(x[0])*x[1], np.sin(x[0])*x[1]

def x_component(x):
    # convert trans velocity into x component
    return x[0] * np.cos(x[1])

def y_component(x):
    # convert trans velocity into y component
    return x[0] * np.sin(x[1])

model = nengo.Network(seed=13)
#model.config[nengo.Ensemble].neuron_type = nengo.Direct() #TODO: temp, just use direct for debugging
ps_neuron_type = nengo.LIF() # neuron type for posecells
with model:

    # Node that handles ROS communication
    # in:  best_x, best_y, best_th
    # out: vtrans, vrot, stim_x, stim_y, stim_th, energy
    if __name__ == '__main__':
        posecell_node = nengo.Node(NengoPosecellNetwork(), size_in=3, size_out=6)
    else: # ROS can't register signals within nengo_gui
        posecell_node = nengo.Node(NengoPosecellNetwork(disable_signals=True), size_in=3, size_out=6)

    # Splitting posecell network into x, y, and th components to reduce
    # computational burden. Some representational power is lost, but it is
    # hopefully good enough to still work.
    posecells_x = nengo.Ensemble(n_neurons=n_neurons, dimensions=3, neuron_type=ps_neuron_type)
    posecells_y = nengo.Ensemble(n_neurons=n_neurons, dimensions=3, neuron_type=ps_neuron_type)
    posecells_th = nengo.Ensemble(n_neurons=n_neurons, dimensions=3, neuron_type=ps_neuron_type)

    # Handles injected current
    stim_x = nengo.Ensemble(100, 2)
    stim_y = nengo.Ensemble(100, 2)
    stim_th = nengo.Ensemble(100, 2)

    nengo.Connection(stim_x, posecells_x[:2], function=stim_input)
    nengo.Connection(stim_y, posecells_y[:2], function=stim_input)
    nengo.Connection(stim_th, posecells_th[:2], function=stim_input)


    # vtrans and current theta.
    velocity_input = nengo.Ensemble(n_neurons=500, dimensions=2) 
    
    nengo.Connection(velocity_input, posecells_x[2], function=x_component)
    nengo.Connection(velocity_input, posecells_y[2], function=y_component)

    max_x = nengo.Ensemble(n_neurons=100, dimensions=1)
    max_y = nengo.Ensemble(n_neurons=100, dimensions=1)
    max_th = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    # Posecell recurrent connections and decoding
    nengo.Connection(posecells_x, posecells_x[:2], function=integrate)
    nengo.Connection(posecells_x, max_x, function=decoding)

    nengo.Connection(posecells_y, posecells_y[:2], function=integrate)
    nengo.Connection(posecells_y, max_y, function=decoding)

    nengo.Connection(posecells_th, posecells_th[:2], function=integrate)
    nengo.Connection(posecells_th, max_th, function=decoding)


    nengo.Connection(max_th, velocity_input[1], transform=np.pi)

    nengo.Connection(max_x, posecell_node[0])
    nengo.Connection(max_y, posecell_node[1])
    nengo.Connection(max_th, posecell_node[2])

    nengo.Connection(posecell_node[0], velocity_input[0])
    nengo.Connection(posecell_node[1], posecells_th[2])

    nengo.Connection(posecell_node[2], stim_x[0])
    nengo.Connection(posecell_node[3], stim_y[0])
    nengo.Connection(posecell_node[4], stim_th[0])

    # Energy
    nengo.Connection(posecell_node[5], stim_x[1])
    nengo.Connection(posecell_node[5], stim_y[1])
    nengo.Connection(posecell_node[5], stim_th[1])

# replace Ensembles in nengo.Direct mode with Nodes
def direct_nodes(model):
    conns = list(model.all_connections)
    inputs, outputs = nengo.utils.builder.find_all_io(conns)
    
    nets = [model] + list(model.all_networks)
    
    for net in nets:
        for ens in list(net.ensembles):
            if isinstance(ens.neuron_type, nengo.Direct):
                print 'replacing', ens
            
                with net:
                    node = nengo.Node(output=lambda t, x: x,
                                      size_in=ens.dimensions,
                                      label=ens.label)
                    inputs[node] = []
                    outputs[node] = []
                
                if ens in inputs:
                    for c in inputs[ens]:
                        for cnet in nets:
                            if c in cnet.connections:
                                with cnet:
                                    new_c = nengo.Connection(c.pre, 
                                                             node[c.post_slice],
                                                             function=c.function, 
                                                             transform=c.transform,
                                                             synapse=c.synapse)
                                outputs[c.pre_obj].remove(c)
                                outputs[c.pre_obj].append(new_c)
                                inputs[node].append(new_c)
                                cnet.connections.remove(c)
                                break
                        else:
                            print('Error processing %s' % ens)
                        
                if ens in outputs:
                    for c in outputs[ens]:
                        for cnet in nets:
                            if c in cnet.connections:
                                with cnet:
                                    new_c = nengo.Connection(node[c.pre_slice], 
                                                             c.post,
                                                             function=c.function, 
                                                             transform=c.transform,
                                                             synapse=c.synapse)
                                inputs[c.post_obj].remove(c)
                                inputs[c.post_obj].append(new_c)
                                outputs[node].append(new_c)
                                cnet.connections.remove(c)
                                break
                        else:
                            print('Error processing %s' % ens)
                        
                net.ensembles.remove(ens)
    return model                

# replace passthrough Nodes with equivalent Nodes (so they're run on the PC, not SpiNNaker)
def replace_passthrough(model):                
    for node in model.all_nodes:
        if node.output is None:
            node.output = lambda t, x: x                
    return model

model = direct_nodes(model)
model = replace_passthrough(model)

nengo_spinnaker.add_spinnaker_params(model.config)

if __name__ == '__main__':

    print( "starting simulator..." )
    before = time.time()

    sim = nengo_spinnaker.Simulator(model)

    after = time.time()
    print( "time to build:" )
    print( after - before )

    print( "running simulator..." )
    before = time.time()
    
    with sim:
        sim.run(1000)

