# Posecell Network using function space representation
# TODO: inject some energy in the middle initially to get it started
# TODO: on the inject function, also output a magnitude, which decays over time

import nengo
import numpy as np
import time

import nengo.utils.function_space
nengo.dists.Function = nengo.utils.function_space.Function
nengo.dists.Combined = nengo.utils.function_space.Combined
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace

from nengo_posecell_network import NengoPosecellNetwork

n_basis_1d=10
n_basis_2d=100#30
n_samples=10000
domain_min = -1
domain_max = 1
domain_range = domain_max - domain_min
domain_points = 80#75#2000
domain = np.linspace(domain_min, domain_max, domain_points)

x_domain = np.linspace(domain_min, domain_max, domain_points)
y_domain = np.linspace(domain_min, domain_max, domain_points)
vx_domain, vy_domain = np.meshgrid(x_domain, y_domain)
def gaussian2d(mag, mean_x, mean_y, std):
    y_offset = domain_range
    x_offset = domain_range
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
                         n_basis=n_basis_1d)
fs2d = nengo.FunctionSpace(nengo.dists.Function(gaussian2d,
                                              mean_x=nengo.dists.Uniform(domain_min,
                                                                         domain_max),
                                              mean_y=nengo.dists.Uniform(domain_min,
                                                                         domain_max),
                                              std=nengo.dists.Uniform(.1, .7),
                                              mag=1),
                         n_basis=n_basis_2d, n_samples=n_samples)

model = nengo.Network(seed=13)
model.config[nengo.Ensemble].neuron_type = nengo.Direct() #TODO: temp, just use direct for debugging
ps_neuron_type = nengo.Direct()#nengo.LIF() # neuron type for posecells
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
    posecells_xy = nengo.Ensemble(n_neurons=5000, dimensions=fs2d.n_basis + 2,
                                 neuron_type=ps_neuron_type)
    posecells_xy.encoders = nengo.dists.Combined([fs2d.project(nengo.dists.Function(gaussian2d,
                                        mean_x=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        mean_y=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=1)),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs2d.n_basis, 2], weights=[1,1],
                                              normalize_weights=True)
    
    posecells_xy.eval_points = nengo.dists.Combined([fs2d.project(nengo.dists.Function(gaussian2d,
                                        mean_x=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        mean_y=nengo.dists.Uniform(domain_min,
                                                                 domain_max),
                                        std=nengo.dists.Uniform(.2,.2),
                                        mag=nengo.dists.Uniform(0,1))),
                                               nengo.dists.UniformHypersphere(surface=False)
                                              ],
                                              [fs2d.n_basis, 2], weights=[1,1],
                                              normalize_weights=True)
    
    posecells_th = nengo.Ensemble(n_neurons=3000, dimensions=fs.n_basis + 1,
                                  neuron_type=ps_neuron_type)
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


    stimulus_xy = fs2d.make_stimulus_node(gaussian2d, 4)
    stimulus_th = fs.make_stimulus_node(gaussian, 3)
    nengo.Connection(stimulus_xy, posecells_xy[:-2])
    nengo.Connection(stimulus_th, posecells_th[:-1])

    plot_xy = fs2d.make_2Dplot_node(domain=x_domain)
    plot_th = fs.make_plot_node(domain=domain, lines=2, n_pts=50)

    nengo.Connection(posecells_xy[:-2], plot_xy[:fs2d.n_basis], synapse=0.1)
    #nengo.Connection(stimulus_xy, plot_xy[fs2d.n_basis:], synapse=0.1)

    nengo.Connection(posecells_th[:-1], plot_th[:fs.n_basis], synapse=0.1)
    nengo.Connection(stimulus_th, plot_th[fs.n_basis:], synapse=0.1)

    def collapse(x):
        pts = fs.reconstruct(x[:-1])
        peak = np.argmax(pts)
        data = gaussian(mag=1, std=0.2, mean=domain[peak])

        shift = int(x[-1]*domain_points/4)

        data = fs.project(np.roll(data, shift))*1.1
        return data
    
    def collapse2d(x):
        pts = fs2d.reconstruct(x[:-2])
        peak = np.argmax(pts)
        data = gaussian2d(mag=1, std=0.2, 
                          mean_x=vx_domain.flatten()[peak],
                          mean_y=vy_domain.flatten()[peak])

        shift_x = int(x[-2]*domain_points/4)
        shift_y = int(x[-1]*domain_points/4)

        data2d = data.reshape((domain_points, domain_points))

        data2d = np.roll(data2d, shift_x, axis=0)
        data2d = np.roll(data2d, shift_y, axis=1)
        data = fs2d.project(data2d.flatten())*1.1

        return data
    
    def collapse_other(x):
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
    
    def peak_location_xy(x):
        # gets the location of the max value in the gaussian
        pts = fs2d.reconstruct(x[:-2])
        peak = np.argmax(pts)
        return domain[peak[0]], domain[peak[1]]
    
    def peak_location_x(x):
        # gets the location of the max value in the gaussian
        pts = fs2d.reconstruct(x[:-2]).reshape(domain_points, domain_points)
        peak = np.unravel_index(np.argmax(pts), pts.shape)
        return domain[peak[0]]
    
    def peak_location_y(x):
        # gets the location of the max value in the gaussian
        pts = fs2d.reconstruct(x[:-2]).reshape(domain_points, domain_points)
        peak = np.unravel_index(np.argmax(pts), pts.shape)
        return domain[peak[1]]
    
    def peak_location_th(x):
        # gets the location of the max value in the gaussian for theta
        pts = fs.reconstruct(x[:-1])
        peak = np.argmax(pts)
        return domain[peak] * np.pi

    def velocity_components(x):
        # convert vtrans into x and y components
        return [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]
    
    def xy_component(x):
        # convert vtrans into x and y components
        return [x[0] * np.cos(x[1]), x[0] * np.sin(x[1])]

    nengo.Connection(posecells_xy, posecells_xy[:-2], synapse=0.1, function=collapse2d)
    nengo.Connection(posecells_th, posecells_th[:-1], synapse=0.1, function=collapse)

    # vtrans and current theta.
    velocity_input = nengo.Ensemble(n_neurons=500, dimensions=2) 
    
    nengo.Connection(velocity_input, posecells_xy[-2:], function=xy_component)
    #nengo.Connection(velocity_input, posecells_yy[-1], function=y_component)
    
    stim_control_xy = nengo.Node([1,0,0,0.2])
    stim_xy = nengo.Ensemble(n_neurons=400, dimensions=4)
    nengo.Connection(stim_control_xy[[2,3]], stim_xy[[2,3]])
    nengo.Connection(stim_xy, stimulus_xy)
    
    stim_control_th = nengo.Node([1,0,0.2])
    stim_th = nengo.Ensemble(n_neurons=300, dimensions=3)
    nengo.Connection(stim_control_th[2], stim_th[2])
    nengo.Connection(stim_th, stimulus_th)

    max_x = nengo.Ensemble(n_neurons=100, dimensions=1)
    max_y = nengo.Ensemble(n_neurons=100, dimensions=1)
    max_th = nengo.Ensemble(n_neurons=100, dimensions=1)

    nengo.Connection(max_th, velocity_input[1])

    nengo.Connection(posecells_xy, max_x, function=peak_location_x)
    nengo.Connection(posecells_xy, max_y, function=peak_location_y)
    nengo.Connection(posecells_th, max_th, function=peak_location_th)

    nengo.Connection(max_x, posecell_node[0])
    nengo.Connection(max_y, posecell_node[1])
    nengo.Connection(max_th, posecell_node[2])

    nengo.Connection(posecell_node[0], velocity_input[0])
    nengo.Connection(posecell_node[1], posecells_th[-1])

    nengo.Connection(posecell_node[2], stim_xy[1])
    nengo.Connection(posecell_node[3], stim_xy[2])
    nengo.Connection(posecell_node[4], stim_th[1])

    nengo.Connection(posecell_node[5], stim_xy[0])
    nengo.Connection(posecell_node[5], stim_th[0])

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
