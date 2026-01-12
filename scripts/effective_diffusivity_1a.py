"""
This script imports a fitted network and calculates the effective diffusivity
in each direction, written for micro network! 

Created by: Mike McKague
January 12, 2026
"""

import numpy as np
import openpnm as op

# import network
data = np.load('../networks/fit_network_micro.npz')
data = {key: np.array(data[key]) for key in data.files}

# take out trained weights
D = data["pore.diameter"]
tsf = data["throat.tsf"]

# take out initial weights
D0 = data["pore.initial_diameters"]
tsf0 = data["throat.initial_tsf"]

spacing = 5e-7
shape = [10, 10, 10]
net = op.network.Cubic(shape=shape, spacing=spacing)

# get coords and conns
coords = net["pore.coords"]
conns = net["throat.conns"]

# add pore and throat diameters to network
net["pore.diameter"] = D * spacing
net["throat.diameter"] = tsf * np.min(np.abs(D[conns]), axis=1) * spacing

# add geometry models 
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
del geo_mods["pore.seed"], geo_mods["pore.max_size"], geo_mods["pore.diameter"]
del geo_mods["throat.max_size"], geo_mods["throat.diameter"]
net.add_model_collection(models=geo_mods)
net.regenerate_models()

# create phase object
phase = op.phase.Phase(network=net)
phase["throat.diffusivity"] = 1e-9

# add physics models
phys_mods = op.models.collections.physics.basic.copy()
del phys_mods["throat.hydraulic_conductance"]
del phys_mods["throat.entry_pressure"]
phase.add_model_collection(models=phys_mods)
phase.regenerate_models()

# stokes flow simulation to estimate permeability
cin, cout = 1, 0
Ax = shape[1] * shape[2] * spacing ** 2
Ay = shape[0] * shape[2] * spacing ** 2
Az = shape[0] * shape[1] * spacing ** 2
Lx, Ly, Lz = np.array(shape) * spacing - spacing

# measure K in x-direction
diff_x = op.algorithms.FickianDiffusion(network=net, phase=phase)
diff_x.set_value_BC(pores=net.pores("xmin"), values=cin)
diff_x.set_value_BC(pores=net.pores("xmax"), values=cout)
diff_x.run()

J_x = diff_x.rate(pores=net.pores("xmin"), mode='group')[0]
D_eff_x = J_x * Lx / (Ax * (cin - cout))
print(f'D_eff_x is: {D_eff_x} m2/s')

# measure K in y-direction
diff_y = op.algorithms.FickianDiffusion(network=net, phase=phase)
diff_y.set_value_BC(pores=net.pores("ymin"), values=cin)
diff_y.set_value_BC(pores=net.pores("ymax"), values=cout)
diff_y.run()

J_y = diff_y.rate(pores=net.pores("ymin"), mode='group')[0]
D_eff_y = J_y * Ly / (Ay * (cin - cout))
print(f'D_eff_y is: {D_eff_y} m2/s')

# measure K in z-direction
diff_z = op.algorithms.FickianDiffusion(network=net, phase=phase)
diff_z.set_value_BC(pores=net.pores("zmin"), values=cin)
diff_z.set_value_BC(pores=net.pores("zmax"), values=cout)
diff_z.run()

J_z = diff_z.rate(pores=net.pores("zmin"), mode='group')[0]
D_eff_z = J_z * Lz / (Az * (cin - cout))
print(f'D_eff_z is: {D_eff_z} m2/s')

D_eff_avg = np.average([D_eff_x, D_eff_y, D_eff_z])
print(f'D_eff is: {D_eff_avg} m2/s')
