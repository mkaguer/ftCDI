import numpy as np
import openpnm as op

# load network
data = np.load('../networks/scale_network_1a.npz')
data = {key: np.array(data[key]) for key in data.files}

# take out trained weights
Dp = data["pore.diameter"]
Dt = data["throat.diameter"]

# create network
spacing = 1e-5
shape = [70, 70, 70]
net = op.network.Cubic(shape=shape, spacing=spacing)

# get coords and conns
coords = net["pore.coords"]
conns = net["throat.conns"]

# add pore and throat diameters to network
net["pore.diameter"] = Dp
net["throat.diameter"] = Dt

# add geometry models 
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
del geo_mods["pore.seed"], geo_mods["pore.max_size"], geo_mods["pore.diameter"]
del geo_mods["throat.max_size"], geo_mods["throat.diameter"]
net.add_model_collection(models=geo_mods)
net.regenerate_models()

# properties of mercurcy invasion
sigma = 0.4791  # N/m
theta = 140  # degrees

# create phase object
phase = op.phase.Phase(network=net)
phase["throat.contact_angle"] = theta
phase["throat.surface_tension"] = sigma
phase["throat.viscosity"] = 1e-3 

# add physics models
phys_mods = op.models.collections.physics.basic.copy()
del phys_mods["throat.diffusive_conductance"]
phase.add_model_collection(models=phys_mods)
phase.regenerate_models()

# run drainage
alg = op.algorithms.Drainage(phase=phase, network=net)
alg.set_inlet_BC(pores=net.pores("surface"))
alg.run()

# add pore and throat invasion sequences to 
net["throat.invasion_sequence"] = alg["throat.invasion_sequence"]
net["pore.invasion_sequence"] = alg["pore.invasion_sequence"]

# save as xdmf for visualization
net["throat.radius"] = net["throat.diameter"]/2
project = net.project
op.io.project_to_xdmf(project,
                      filename="../paraview/" + "scale_network_1a" + ".xdmf")