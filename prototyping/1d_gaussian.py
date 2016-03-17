# 1D cyclic gaussian with velocity using Nengo function space stuff

import nengo
import numpy as np

import nengo.utils.function_space
nengo.dists.Function = nengo.utils.function_space.Function
nengo.dists.Combined = nengo.utils.function_space.Combined
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace

domain_min = -1
domain_max = 1
domain_range = domain_max - domain_min
domain_points = 2000
domain = np.linspace(domain_min, domain_max, domain_points)

def gaussian(mag, mean, std):
    if mean > domain_max:
        mean -= domain_range
    if mean < domain_min:
        mean += domain_range
    try:
        # Adding gaussians offset by the domain range to simulate cycling
        return mag * ( np.exp(-(domain - mean)**2 / (2 * std**2)) +\
                       np.exp(-(domain - mean - domain_range)**2 / (2 * std**2)) +\
                       np.exp(-(domain - mean + domain_range)**2 / (2 * std**2))
                     )

    except FloatingPointError:
        return domain * 0

fs = nengo.FunctionSpace(nengo.dists.Function(gaussian,
                                              mean=nengo.dists.Uniform(domain_min,
                                                                       domain_max),
                                              std=nengo.dists.Uniform(.1, .7),
                                              mag=1),
                         n_basis=10)

model = nengo.Network(seed=13)
model.config[nengo.Ensemble].neuron_type = nengo.Direct() #TODO: temp, just use direct for debugging
with model:
    posecells = nengo.Ensemble(n_neurons=3000, dimensions=fs.n_basis + 1)
    posecells.encoders = nengo.dists.Combined([fs.project(nengo.dists.Function(gaussian,
                                        mean=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=1)),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs.n_basis, 1], weights=[1,1],
                                              normalize_weights=True)
    
    posecells.eval_points = nengo.dists.Combined([fs.project(nengo.dists.Function(gaussian,
                                        mean=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=nengo.dists.Uniform(0,1))),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs.n_basis, 1], weights=[1,1],
                                              normalize_weights=True)


    stimulus = fs.make_stimulus_node(gaussian, 3)
    nengo.Connection(stimulus, posecells[:-1])

    plot = fs.make_plot_node(domain=domain, lines=2, n_pts=50)

    nengo.Connection(posecells[:-1], plot[:fs.n_basis], synapse=0.1)
    nengo.Connection(stimulus, plot[fs.n_basis:], synapse=0.1)

    def collapse(x):
        pts = fs.reconstruct(x[:-1])
        peak = np.argmax(pts)
        data = gaussian(mag=1, std=0.2, mean=domain[peak])

        shift = int(x[-1]*domain_points/4)

        data = fs.project(np.roll(data, shift))*1.1
        return data
    
    def collapse2(x):
        pts = fs.reconstruct(x[:-1])
        peak = np.argmax(pts)
        data = gaussian(mag=1, std=0.2, mean=domain[peak])

        shift = int(x[-1]*50)

        res = fs.project(np.roll(pts*.5 + data*.5, shift))*1.00#1.1
        return res

    nengo.Connection(posecells, posecells[:-1], synapse=0.1, function=collapse)
    #nengo.Connection(posecells, posecells[:-1], synapse=0.1, function=collapse2)

    velocity = nengo.Node([0])
    nengo.Connection(velocity, posecells[-1])
    
    stim_control = nengo.Node([1,0,0.2])
    nengo.Connection(stim_control, stimulus)
