# Posecell Network using function space representation

import nengo
import numpy as np

import nengo.utils.function_space
nengo.dists.Function = nengo.utils.function_space.Function
nengo.dists.Combined = nengo.utils.function_space.Combined
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace

from nengo_posecell_network import NengoPosecellNetwork

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

    # Node that handles ROS communication
    # in:  best_x, best_y, best_th
    # out: vtrans, vrot, stim_x, stim_y, stim_th
    posecell_node = nengo.Node(NengoPosecellNetwork(), size_in=3, size_out=5)

    # Splitting posecell network into x, y, and th components to reduce
    # computational burden. Some representational power is lost, but it is
    # hopefully good enough to still work.
    posecells_x = nengo.Ensemble(n_neurons=3000, dimensions=fs.n_basis + 1)
    posecells_x.encoders = nengo.dists.Combined([fs.project(nengo.dists.Function(gaussian,
                                        mean=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=1)),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs.n_basis, 1], weights=[1,1],
                                              normalize_weights=True)
    
    posecells_x.eval_points = nengo.dists.Combined([fs.project(nengo.dists.Function(gaussian,
                                        mean=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=nengo.dists.Uniform(0,1))),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs.n_basis, 1], weights=[1,1],
                                              normalize_weights=True)
    
    posecells_y = nengo.Ensemble(n_neurons=3000, dimensions=fs.n_basis + 1)
    posecells_y.encoders = nengo.dists.Combined([fs.project(nengo.dists.Function(gaussian,
                                        mean=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=1)),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs.n_basis, 1], weights=[1,1],
                                              normalize_weights=True)
    
    posecells_y.eval_points = nengo.dists.Combined([fs.project(nengo.dists.Function(gaussian,
                                        mean=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=nengo.dists.Uniform(0,1))),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs.n_basis, 1], weights=[1,1],
                                              normalize_weights=True)
    
    posecells_th = nengo.Ensemble(n_neurons=3000, dimensions=fs.n_basis + 1)
    posecells_th.encoders = nengo.dists.Combined([fs.project(nengo.dists.Function(gaussian,
                                        mean=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=1)),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs.n_basis, 1], weights=[1,1],
                                              normalize_weights=True)
    
    posecells_th.eval_points = nengo.dists.Combined([fs.project(nengo.dists.Function(gaussian,
                                        mean=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=nengo.dists.Uniform(0,1))),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs.n_basis, 1], weights=[1,1],
                                              normalize_weights=True)


    stimulus_x = fs.make_stimulus_node(gaussian, 3)
    stimulus_y = fs.make_stimulus_node(gaussian, 3)
    stimulus_th = fs.make_stimulus_node(gaussian, 3)
    nengo.Connection(stimulus_x, posecells_x[:-1])
    nengo.Connection(stimulus_y, posecells_y[:-1])
    nengo.Connection(stimulus_th, posecells_th[:-1])

    plot_x = fs.make_plot_node(domain=domain, lines=2, n_pts=50)
    plot_y = fs.make_plot_node(domain=domain, lines=2, n_pts=50)
    plot_th = fs.make_plot_node(domain=domain, lines=2, n_pts=50)

    nengo.Connection(posecells_x[:-1], plot_x[:fs.n_basis], synapse=0.1)
    nengo.Connection(stimulus_x, plot_x[fs.n_basis:], synapse=0.1)

    nengo.Connection(posecells_y[:-1], plot_y[:fs.n_basis], synapse=0.1)
    nengo.Connection(stimulus_y, plot_y[fs.n_basis:], synapse=0.1)

    nengo.Connection(posecells_th[:-1], plot_th[:fs.n_basis], synapse=0.1)
    nengo.Connection(stimulus_th, plot_th[fs.n_basis:], synapse=0.1)

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

    def peak_location(x):
        # gets the location of the max value in the gaussian
        pts = fs.reconstruct(x[:-1])
        peak = np.argmax(pts)
        return domain[peak]
    
    def peak_location_th(x):
        # gets the location of the max value in the gaussian for theta
        pts = fs.reconstruct(x[:-1])
        peak = np.argmax(pts)
        return domain[peak] * np.pi

    def velocity_components(x):
        # convert trans and rot velocity into x and y components
        return [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
    
    def x_component(x):
        # convert trans and rot velocity into x component
        return x[0] * np.cos(x[2])
    
    def y_component(x):
        # convert trans and rot velocity into x component
        return x[0] * np.sin(x[2])

    nengo.Connection(posecells_x, posecells_x[:-1], synapse=0.1, function=collapse)
    nengo.Connection(posecells_y, posecells_y[:-1], synapse=0.1, function=collapse)
    nengo.Connection(posecells_th, posecells_th[:-1], synapse=0.1, function=collapse)

    v_trans_rot = nengo.Node([0,0]) # linear (forward) and rotational velocity
    
    # velocity in x and y directions, and current theta.
    velocity_input = nengo.Ensemble(n_neurons=500, dimensions=3) 
    
    nengo.Connection(v_trans_rot, velocity_input[:2])
    nengo.Connection(velocity_input, posecells_x[-1], function=x_component)
    nengo.Connection(velocity_input, posecells_y[-1], function=y_component)
    nengo.Connection(v_trans_rot[1], posecells_th[-1])
    
    stim_control_x = nengo.Node([1,0,0.2])
    nengo.Connection(stim_control_x, stimulus_x)
    
    stim_control_y = nengo.Node([1,0,0.2])
    nengo.Connection(stim_control_y, stimulus_y)
    
    stim_control_th = nengo.Node([1,0,0.2])
    nengo.Connection(stim_control_th, stimulus_th)

    max_x = nengo.Ensemble(n_neurons=100, dimensions=1)
    max_y = nengo.Ensemble(n_neurons=100, dimensions=1)
    max_th = nengo.Ensemble(n_neurons=100, dimensions=1)

    nengo.Connection(max_th, velocity_input[2])

    nengo.Connection(posecells_x, max_x, function=peak_location)
    nengo.Connection(posecells_y, max_y, function=peak_location)
    nengo.Connection(posecells_th, max_th, function=peak_location_th)

    nengo.Connection(max_x, posecell_node[0])
    nengo.Connection(max_y, posecell_node[1])
    nengo.Connection(max_th, posecell_node[2])

    nengo.Connection(posecell_node[0], v_trans_rot[0])
    nengo.Connection(posecell_node[1], v_trans_rot[1])

    nengo.Connection(posecell_node[2], stim_control_x[1])
    nengo.Connection(posecell_node[3], stim_control_y[1])
    nengo.Connection(posecell_node[4], stim_control_th[1])
