"""
This script produces figures showing fitted propertes for macro and micro
networks!

Created by: Mike McKague
Date: January 12, 2026
"""

import numpy as np
import openpnm as op
import matplotlib.pyplot as plt

op.visualization.set_mpl_style()

# properties of mercurcy invasion
sigma = 0.4791  # N/m
theta = 140  # degrees

#%% MACRO Network
data = np.load('../networks/fit_network_macro.npz')
data = {key: np.array(data[key]) for key in data.files}

# take out trained weights
D = data["pore.diameter"]
tsf = data["throat.tsf"]

# take out initial weights
D0 = data["pore.initial_diameters"]
tsf0 = data["throat.initial_tsf"]

spacing = 1e-5
shape = [10, 10, 10]
net_ma = op.network.Cubic(shape=shape, spacing=spacing)

# get coords and conns
coords = net_ma["pore.coords"]
conns = net_ma["throat.conns"]

# add pore and throat diameters to network
net_ma["pore.diameter"] = D * spacing
net_ma["throat.diameter"] = tsf * np.min(np.abs(D[conns]), axis=1) * spacing

# add geometry models 
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
del geo_mods["pore.seed"], geo_mods["pore.max_size"], geo_mods["pore.diameter"]
del geo_mods["throat.max_size"], geo_mods["throat.diameter"]
net_ma.add_model_collection(models=geo_mods)
net_ma.regenerate_models()

# create phase object
phase = op.phase.Phase(network=net_ma)
phase["throat.contact_angle"] = theta
phase["throat.surface_tension"] = sigma
phase["throat.viscosity"] = 1e-3 

# add physics models
phys_mods = op.models.collections.physics.basic.copy()
del phys_mods["throat.diffusive_conductance"]
phase.add_model_collection(models=phys_mods)
phase.regenerate_models()

# stokes flow simulation to estimate permeability
Pin, Pout = 1, 0
Ax = shape[1] * shape[2] * spacing ** 2
Ay = shape[0] * shape[2] * spacing ** 2
Az = shape[0] * shape[1] * spacing ** 2
Lx, Ly, Lz = np.array(shape) * spacing - spacing
mu = phase['pore.viscosity'].max()

# measure K in x-direction
flow_x = op.algorithms.StokesFlow(network=net_ma, phase=phase)
flow_x.set_value_BC(pores=net_ma.pores("xmin"), values=Pin)
flow_x.set_value_BC(pores=net_ma.pores("xmax"), values=Pout)
flow_x.run()

Q_x = flow_x.rate(pores=net_ma.pores("xmin"), mode='group')[0]
K_x = Q_x * Lx * mu / (Ax * (Pin - Pout))
print(f'K_x is: {K_x} m2')

# measure K in y-direction
flow_y = op.algorithms.StokesFlow(network=net_ma, phase=phase)
flow_y.set_value_BC(pores=net_ma.pores("ymin"), values=Pin)
flow_y.set_value_BC(pores=net_ma.pores("ymax"), values=Pout)
flow_y.run()

Q_y = flow_y.rate(pores=net_ma.pores("ymin"), mode='group')[0]
K_y = Q_y * Ly * mu / (Ay * (Pin - Pout))
print(f'K_y is: {K_y} m2')

# measure K in z-direction
flow_z = op.algorithms.StokesFlow(network=net_ma, phase=phase)
flow_z.set_value_BC(pores=net_ma.pores("zmin"), values=Pin)
flow_z.set_value_BC(pores=net_ma.pores("zmax"), values=Pout)
flow_z.run()

Q_z = flow_z.rate(pores=net_ma.pores("zmin"), mode='group')[0]
K_z = Q_z * Lz * mu / (Az * (Pin - Pout))
print(f'K_z is: {K_z} m2')

K_avg = np.average([K_x, K_y, K_z])
print(f'K is: {K_avg} m2')

# run drainage
alg_ma = op.algorithms.Drainage(phase=phase, network=net_ma)
alg_ma.set_inlet_BC(pores=net_ma.pores("surface"))
alg_ma.run()

# get pc curve data
data = alg_ma.pc_curve()
sat_ma = data.snwp
pc_ma = data.pc

#%% MICRO Network
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
net_mi = op.network.Cubic(shape=shape, spacing=spacing)

# get coords and conns
coords = net_mi["pore.coords"]
conns = net_mi["throat.conns"]

# add pore and throat diameters to network
net_mi["pore.diameter"] = D * spacing
net_mi["throat.diameter"] = tsf * np.min(np.abs(D[conns]), axis=1) * spacing

# add geometry models 
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
del geo_mods["pore.seed"], geo_mods["pore.max_size"], geo_mods["pore.diameter"]
del geo_mods["throat.max_size"], geo_mods["throat.diameter"]
net_mi.add_model_collection(models=geo_mods)
net_mi.regenerate_models()

# create phase object
phase = op.phase.Phase(network=net_mi)
phase["throat.contact_angle"] = theta
phase["throat.surface_tension"] = sigma

# add physics models
phys_mods = op.models.collections.physics.basic.copy()
del phys_mods["throat.diffusive_conductance"]
del phys_mods["throat.hydraulic_conductance"]
phase.add_model_collection(models=phys_mods)
phase.regenerate_models()

# calculate porosity
V_total = np.prod(np.array(shape)*spacing)
V_pore = np.sum(net_mi["pore.volume"]) + np.sum(net_mi["throat.volume"])
porosity_mi = V_pore/V_total
print(f"porosity={porosity_mi*100}%")

# run drainage
alg_mi = op.algorithms.Drainage(phase=phase, network=net_mi)
alg_mi.set_inlet_BC(pores=net_mi.pores("surface"))
alg_mi.run()

# get pc curve data
# pressures = np.logspace(np.log10(2829802.8), np.log10(27112410), num=250)
# data = alg_ma.pc_curve(pressures=pressures)
data = alg_mi.pc_curve()
sat_mi = data.snwp
pc_mi = data.pc

#%% load experiment data
data = np.loadtxt("../figures/guyes2017b/mip-data.csv",
                  delimiter=",", skiprows=1)

# retrieve data
x = data[:, 0]  # um
y = data[:, 1]  # sat

# transform to sat and Pc data
sat_exp = y/np.max(y)
pc_exp = -4 * sigma * np.cos(theta*np.pi/180) / (x/1e6)  # assume cylinderical tubes

#%% plot porosimetry curves

# FIXME: Should we add these points
pc_mi = np.concatenate((np.array([2829802.8]), pc_mi, np.array([27112410])))
sat_mi = np.concatenate((np.array([0.0]), sat_mi, np.array([1.0])))

# first scale data
sat_ma *= y[26]/y[0]
sat_mi *= (y[0]-y[26])/y[0] # (sat_mi - np.minsat_mi)/(np.min(sat_mi))
sat_mi += y[26]/y[0] 

plt.figure(1, dpi=500)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3)
# Plot experimental data
plt.semilogx(pc_exp, sat_exp, label='Experiment',
             color='k', linestyle='solid', linewidth=3,
             marker='o', markersize=4,
             markerfacecolor='k',          # transparent center
             markeredgecolor='k', markeredgewidth=2)
# Plot macro data
plt.semilogx(pc_ma, sat_ma, label='Macro PNM',
             color='g', linestyle='solid', linewidth=3,
             marker='o', markersize=10,
             markerfacecolor='none',          # transparent center
             markeredgecolor='g', markeredgewidth=2)
# Plot micro data
plt.semilogx(pc_mi, sat_mi, label='Micro PNM',
             color='purple', linestyle='solid', linewidth=3,
             marker='o', markersize=8,
             markerfacecolor='none',          # transparent center
             markeredgecolor='purple', markeredgewidth=2)
plt.xlabel('Pressure (Pa)', fontsize=18)
plt.ylabel('Saturation', fontsize=18)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(frameon=True, fontsize=16)

# save
plt.savefig('../figures/fitted_networks_1a_porosimetry.png')
plt.show()

#%% plot permeabilities
plt.figure(2, dpi=500)
ax = plt.gca()  # Get current axes

# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3) 

K_fitted = np.array([K_x, K_y, K_z, K_avg])
K_target = np.array([2e-16, 2e-16, 2e-16, 2e-16])

x = np.arange(len(K_fitted))
bar_width = 0.25

plt.bar(x + bar_width/2, K_target, width=bar_width, label='Target', color='k')
plt.bar(x + bar_width*3/2, K_fitted, width=bar_width, label='Macro PNM', color='g')
plt.ylabel('Permeability (m$^2$)', fontsize=18, fontweight='normal')
plt.xticks(x + bar_width, ['X', 'Y', 'Z', 'Avg'], fontsize=18, fontweight='normal')
plt.yticks(fontsize=18, fontweight='normal')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=16, frameon=True)
plt.tight_layout()
plt.savefig('../figures/fitted_networks_1a_K.png')
plt.show()

#%% plot porosity
plt.figure(3, dpi=500)
ax = plt.gca()  # Get current axes

# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3) 
ax.tick_params(axis='x', labelbottom=False)

rho_fitted = np.array([porosity_mi])
rho_target = np.array([0.187])

x = np.arange(len(rho_fitted))
bar_width = 0.25

plt.bar(x + bar_width/2, rho_target, width=bar_width, label='Target', color='k')
plt.bar(x + bar_width*3/2, rho_fitted, width=bar_width, label='Micro PNM', color='purple')
plt.ylabel('Porosity (%)', fontsize=18, fontweight='normal')
plt.xticks(x + bar_width)
plt.yticks(fontsize=18, fontweight='normal')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=16, frameon=True)
plt.tight_layout()
plt.savefig('../figures/fitted_networks_1a_porosity.png')
plt.show()

#%% export networks

# micro
net_mi["pore.invasion_sequence"] = alg_mi["pore.invasion_sequence"]
net_mi["throat.invasion_sequence"] = alg_mi["throat.invasion_sequence"]
net_mi["throat.radius"] = net_mi["throat.diameter"]/2
op.io.project_to_xdmf(project=net_mi.project,
                      filename="../paraview/fitted_networks_1a_micro")

# macro
net_ma["pore.invasion_sequence"] = alg_ma["pore.invasion_sequence"]
net_ma["throat.invasion_sequence"] = alg_ma["throat.invasion_sequence"]
net_ma["throat.radius"] = net_ma["throat.diameter"]/2
op.io.project_to_xdmf(project=net_ma.project,
                      filename="../paraview/fitted_networks_1a_macro")