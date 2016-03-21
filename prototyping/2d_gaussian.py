# 2D cyclic gaussian with velocity using Nengo function space stuff

import nengo
import numpy as np

import nengo.utils.function_space
nengo.dists.Function = nengo.utils.function_space.Function
nengo.dists.Combined = nengo.utils.function_space.Combined
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace

n_basis=20
n_samples=1000
domain_min = -1
domain_max = 1
domain_range = domain_max - domain_min
domain_points = 75#200
x_domain = np.linspace(domain_min, domain_max, domain_points)
y_domain = np.linspace(domain_min, domain_max, domain_points)
vx_domain, vy_domain = np.meshgrid(x_domain, y_domain)
def gaussian2d(mag, mean_x, mean_y, std):
    #y_offset = np.array([0, domain_range]).reshape((2,1,1))
    #x_offset = np.array([domain_range, 0]).reshape((2,1,1))
    y_offset = domain_range
    x_offset = domain_range
    #mean = np.array([mean_x, mean_y]).reshape((2,1,1))
    try:
        # Adding gaussians offset by the domain range to simulate cycling
        return (mag * ( np.exp(-((vx_domain - mean_x)**2 / (2 * std**2) +\
                                (vy_domain - mean_y)**2 / (2 * std**2))) +\

                       np.exp(-((vx_domain - mean_x - x_offset)**2 / (2 * std**2) +\
                               (vy_domain - mean_y)**2 / (2 * std**2))) +\

                       np.exp(-((vx_domain - mean_x + x_offset)**2 / (2 * std**2) +\
                               (vy_domain - mean_y)**2 / (2 * std**2))) +\

                       np.exp(-((vx_domain - mean_x)**2 / (2 * std**2) +\
                               (vy_domain - mean_y - y_offset)**2 / (2 * std**2))) +\

                       np.exp(-((vx_domain - mean_x)**2 / (2 * std**2) +\
                               (vy_domain - mean_y + y_offset)**2 / (2 * std**2))) +\
                       
                       np.exp(-((vx_domain - mean_x - x_offset)**2 / (2 * std**2) +\
                               (vy_domain - mean_y - y_offset)**2 / (2 * std**2))) +\
                       
                       np.exp(-((vx_domain - mean_x - x_offset)**2 / (2 * std**2) +\
                               (vy_domain - mean_y + y_offset)**2 / (2 * std**2))) +\
                       
                       np.exp(-((vx_domain - mean_x + x_offset)**2 / (2 * std**2) +\
                               (vy_domain - mean_y - y_offset)**2 / (2 * std**2))) +\
                       
                       np.exp(-((vx_domain - mean_x + x_offset)**2 / (2 * std**2) +\
                               (vy_domain - mean_y + y_offset)**2 / (2 * std**2)))

                     )).flatten()
    except FloatingPointError:
        return (vx_domain * 0).flatten()

fs = nengo.FunctionSpace(nengo.dists.Function(gaussian2d,
                                              mean_x=nengo.dists.Uniform(domain_min,
                                                                         domain_max),
                                              mean_y=nengo.dists.Uniform(domain_min,
                                                                         domain_max),
                                              std=nengo.dists.Uniform(.1, .7),
                                              mag=1),
                         n_basis=n_basis, n_samples=n_samples)

model = nengo.Network(seed=13)
model.config[nengo.Ensemble].neuron_type = nengo.Direct() #TODO: temp, just use direct for debugging
with model:
    posecells = nengo.Ensemble(n_neurons=2000, dimensions=fs.n_basis + 2)
    posecells.encoders = nengo.dists.Combined([fs.project(nengo.dists.Function(gaussian2d,
                                        mean_x=nengo.dists.Uniform(domain_min,
                                                                   domain_max),
                                        mean_y=nengo.dists.Uniform(domain_min,
                                                                   domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=1)),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs.n_basis, 2], weights=[1,1],
                                              normalize_weights=True)
    
    posecells.eval_points = nengo.dists.Combined([fs.project(nengo.dists.Function(gaussian2d,
                                        mean_x=nengo.dists.Uniform(domain_min,
                                                                   domain_max),
                                        mean_y=nengo.dists.Uniform(domain_min,
                                                                   domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=nengo.dists.Uniform(0,1))),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs.n_basis, 2], weights=[1,1],
                                              normalize_weights=True)


    stimulus = fs.make_stimulus_node(gaussian2d, 4)
    nengo.Connection(stimulus, posecells[:-2])

    # plot node uses only one linear domain, and x and y are the same
    plot = fs.make_2Dplot_node(domain=x_domain)

    nengo.Connection(posecells[:-2], plot[:fs.n_basis], synapse=0.1)
    #nengo.Connection(stimulus, plot[fs.n_basis:], synapse=0.1)

    def collapse(x):
        pts = fs.reconstruct(x[:-2])
        peak = np.argmax(pts)
        data = gaussian2d(mag=1, std=0.2, 
                          mean_x=vx_domain.flatten()[peak],
                          mean_y=vy_domain.flatten()[peak])

        shift_x = int(x[-2]*domain_points/4)
        shift_y = int(x[-1]*domain_points/4)

        data2d = data.reshape((domain_points, domain_points))

        data2d = np.roll(data2d, shift_x, axis=0)
        data2d = np.roll(data2d, shift_y, axis=1)
        data = fs.project(data2d.flatten())*1.1

        return data
    
    def collapse2(x):
        pts = fs.reconstruct(x[:-2])
        peak = np.argmax(pts)
        data = gaussian2d(mag=1, std=0.2, 
                          mean_x=domain[peak][0],
                          mean_y=domain[peak][1])

        shift = int(x[-1]*50)

        res = fs.project(np.roll(pts*.5 + data*.5, shift))*1.00#1.1
        return res

    nengo.Connection(posecells, posecells[:-2], synapse=0.1, function=collapse)
    #nengo.Connection(posecells, posecells[:-1], synapse=0.1, function=collapse2)

    velocity = nengo.Node([0,0])
    nengo.Connection(velocity, posecells[-2:])
    
    stim_control = nengo.Node([1,0,0,0.2])
    nengo.Connection(stim_control, stimulus)
