"""
Create a large unperforated network from fitted one!

Created by: Mike McKague
Date: December 6, 2025
"""
import numpy as np
import openpnm as op
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial import cKDTree
import copy
import models.numpy as models

op.visualization.set_mpl_style()

np.random.seed(3)

#%% load network
data = np.load('../networks/fit_network_macro.npz')
data = {key: np.array(data[key]) for key in data.files}

# take out trained weights
D = data["pore.diameter"]
tsf = data["throat.tsf"]

# take out initial weights
D0 = data["pore.initial_diameters"]
tsf0 = data["throat.initial_tsf"]

#%% create a new network of trained data in openpnm
spacing = 1e-5
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

# stokes flow simulation to estimate permeability
Pin, Pout = 1, 0
Ax = shape[1] * shape[2] * spacing ** 2
Ay = shape[0] * shape[2] * spacing ** 2
Az = shape[0] * shape[1] * spacing ** 2
Lx, Ly, Lz = np.array(shape) * spacing - spacing
mu = phase['pore.viscosity'].max()

# measure K in x-direction
flow_x = op.algorithms.StokesFlow(network=net, phase=phase)
flow_x.set_value_BC(pores=net.pores("xmin"), values=Pin)
flow_x.set_value_BC(pores=net.pores("xmax"), values=Pout)
flow_x.run()

Q_x = flow_x.rate(pores=net.pores("xmin"), mode='group')[0]
K_x = Q_x * Lx * mu / (Ax * (Pin - Pout))
print(f'K_x is: {K_x} m2')

# measure K in y-direction
flow_y = op.algorithms.StokesFlow(network=net, phase=phase)
flow_y.set_value_BC(pores=net.pores("ymin"), values=Pin)
flow_y.set_value_BC(pores=net.pores("ymax"), values=Pout)
flow_y.run()

Q_y = flow_y.rate(pores=net.pores("ymin"), mode='group')[0]
K_y = Q_y * Ly * mu / (Ay * (Pin - Pout))
print(f'K_y is: {K_y} m2')

# measure K in z-direction
flow_z = op.algorithms.StokesFlow(network=net, phase=phase)
flow_z.set_value_BC(pores=net.pores("zmin"), values=Pin)
flow_z.set_value_BC(pores=net.pores("zmax"), values=Pout)
flow_z.run()

Q_z = flow_z.rate(pores=net.pores("zmin"), mode='group')[0]
K_z = Q_z * Lz * mu / (Az * (Pin - Pout))
print(f'K_z is: {K_z} m2')

K_avg = np.average([K_x, K_y, K_z])
print(f'K is: {K_avg} m2')

# run drainage
alg = op.algorithms.Drainage(phase=phase, network=net)
alg.set_inlet_BC(pores=net.pores("surface"))
alg.run()

# get pc curve data
data = alg.pc_curve()
sat_tra = data.snwp
pc_tra = data.pc


#%% load experiment data
data = np.loadtxt("../figures/guyes2017b/mip-data.csv",
                  delimiter=",", skiprows=1)

# retrieve data
x = data[:, 0]  # um
y = data[:, 1]  # sat

# transform to sat and Pc data
sat_exp = y/np.max(y)
pc_exp = -4 * sigma * np.cos(theta*np.pi/180) / (x/1e6)  # assume cylinderical tubes


#%% Create a Scaled network 15 by 15 by 15
spacing = 1e-5
shape_s = [10, 10, 10]
net_s = op.network.Cubic(shape=shape_s, spacing=spacing)

# get pore coords
coords_f = net["pore.coords"].copy()
coords_s = net_s["pore.coords"].copy()


def scale_coords(coords):

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x_scaled = (x - np.min(x))/(np.max(x) - np.min(x))
    y_scaled = (y - np.min(y))/(np.max(y) - np.min(y))
    z_scaled = (z - np.min(z))/(np.max(z) - np.min(z))

    return np.vstack((x_scaled, y_scaled, z_scaled)).T


# scale coords (between 0 and 1)
coords_f = scale_coords(coords_f)
coords_s = scale_coords(coords_s)

# get throat coords, already scaled!
conns_f = net['throat.conns'].copy()
conns_s = net_s['throat.conns'].copy()
throat_coords_f = np.sum(coords_f[conns_f], axis=1)/2
throat_coords_s = np.sum(coords_s[conns_s], axis=1)/2

# fit gaussian kde to D and take sample
kde = gaussian_kde(D, bw_method=0.01)
D_kde = kde.resample(net_s.Np)[0]

# fit GP to D and coords_f
kernel = C(1.0) * RBF(length_scale=0.05)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
gp.fit(coords_f, D)
print('Finished GP fit on D')

# sample from GP
tree = cKDTree(coords_f)
_, indices = tree.query(coords_s)
coords_s = coords_f[indices]
D_gp, _ = gp.predict(coords_s, return_std=True)

# sort to preserve spatial trends
indices = np.argsort(D_gp)
values = np.sort(D_kde)
# get sampled Ds
D_sampled = np.zeros_like(D_gp)
D_sampled[indices] = values
D_sampled = np.clip(D_sampled, 1e-2, 1.0)

# get sampled tsf
kde = gaussian_kde(tsf, bw_method=0.01)
tsf_kde = kde.resample(net_s.Nt)[0]

kernel = C(1.0) * RBF(length_scale=0.05)
X_train = np.hstack([coords_f[conns_f[:, 0]], coords_f[conns_f[:, 1]]])
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
gp.fit(X_train, tsf)
print('Finished GP tsf fit')

X_pred = np.hstack([coords_s[conns_s[:, 0]], coords_s[conns_s[:, 1]]])
tsf_gp, _ = gp.predict(X_pred, return_std=True)

# sort to preserve spatial trends
indices = np.argsort(tsf_gp)
values = np.sort(tsf_kde)
# get sampled Ds
tsf_sampled = np.zeros_like(tsf_gp)
tsf_sampled[indices] = values
tsf_sampled = np.clip(tsf_sampled, 1e-2, 1.0)

# plot sampled D
plt.figure(1)
plt.hist(D_sampled*spacing*1e6, label="sampled", density=True, alpha=0.5, bins=20)
plt.hist(D*spacing*1e6, label="fitted", density=True, alpha=0.5, bins=20)
plt.title("Pore Diameter (um)")
plt.legend(frameon=True)
plt.show()

# plot sampled tsf
plt.figure(2)
plt.hist(tsf_sampled, label="sampled", density=True, alpha=0.5, bins=20)
plt.hist(tsf, label="fitted", density=True, alpha=0.5, bins=20)
plt.title("Throat Size Factor")
plt.legend(frameon=True)
plt.show()

# add pore and throat diameters to network
net_s["pore.diameter"] = D_sampled * spacing
net_s["throat.diameter"] = tsf_sampled * np.min(D_sampled[conns_s], axis=1) * spacing

# add geometry models
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
del geo_mods["pore.seed"], geo_mods["pore.max_size"], geo_mods["pore.diameter"]
del geo_mods["throat.max_size"], geo_mods["throat.diameter"]
net_s.add_model_collection(models=geo_mods)
net_s.regenerate_models()

# calculate porosity
pore_volume = np.sum(net_s["throat.volume"]) + np.sum(net_s["pore.volume"])
total_volume = np.prod(np.array(shape_s)*spacing)
porosity = pore_volume/total_volume
print(f"Porosity of Sampled: {porosity}")

# properties of mercurcy invasion
sigma = 0.4791  # N/m
theta = 140  # degrees

# create phase object
phase_s = op.phase.Phase(network=net_s)
phase_s["throat.contact_angle"] = theta
phase_s["throat.surface_tension"] = sigma
phase_s["throat.viscosity"] = 1e-3 

# add physics models
phys_mods = op.models.collections.physics.basic.copy()
del phys_mods["throat.diffusive_conductance"]
phase_s.add_model_collection(models=phys_mods)
phase_s.regenerate_models()

# stokes flow simulation to estimate permeability
Pin, Pout = 1, 0
Ax = shape_s[1] * shape_s[2] * spacing ** 2
Ay = shape_s[0] * shape_s[2] * spacing ** 2
Az = shape_s[0] * shape_s[1] * spacing ** 2
Lx, Ly, Lz = np.array(shape_s) * spacing - spacing
mu = phase_s['pore.viscosity'].max()

# measure K in x-direction
flow_x = op.algorithms.StokesFlow(network=net_s, phase=phase_s)
flow_x.set_value_BC(pores=net_s.pores("xmin"), values=Pin)
flow_x.set_value_BC(pores=net_s.pores("xmax"), values=Pout)
flow_x.run()

Q_x = flow_x.rate(pores=net_s.pores("xmin"), mode='group')[0]
Ks_x = Q_x * Lx * mu / (Ax * (Pin - Pout))
print(f'Ks_x is: {Ks_x} m2')

# measure K in y-direction
flow_y = op.algorithms.StokesFlow(network=net_s, phase=phase_s)
flow_y.set_value_BC(pores=net_s.pores("ymin"), values=Pin)
flow_y.set_value_BC(pores=net_s.pores("ymax"), values=Pout)
flow_y.run()

Q_y = flow_y.rate(pores=net_s.pores("ymin"), mode='group')[0]
Ks_y = Q_y * Ly * mu / (Ay * (Pin - Pout))
print(f'Ks_y is: {Ks_y} m2')

# measure K in z-direction
flow_z = op.algorithms.StokesFlow(network=net_s, phase=phase_s)
flow_z.set_value_BC(pores=net_s.pores("zmin"), values=Pin)
flow_z.set_value_BC(pores=net_s.pores("zmax"), values=Pout)
flow_z.run()

Q_z = flow_z.rate(pores=net_s.pores("zmin"), mode='group')[0]
Ks_z = Q_z * Lz * mu / (Az * (Pin - Pout))
print(f'Ks_z is: {Ks_z} m2')

Ks_avg = np.average([Ks_x, Ks_y, Ks_z])
print(f'Ks is: {Ks_avg} m2')

# run drainage
alg_s = op.algorithms.Drainage(phase=phase_s, network=net_s)
alg_s.set_inlet_BC(pores=net_s.pores("surface"))
alg_s.run()

# get pc curve data
data = alg_s.pc_curve()
sat_sam = data.snwp
pc_sam = data.pc


#%% Stitch network


def stitch_network(network, shape, spacing, ns):
    
    # get ns
    nx, ny, nz = ns
    # create stitched network
    stitched = op.io.network_from_porespy(copy.deepcopy(network))
    
    
    def assign_labels(network):
        
        # get min and max z, y, z
        xmax = np.max(network["pore.coords"][:, 0])
        xmin = np.min(network["pore.coords"][:, 0])
        ymax = np.max(network["pore.coords"][:, 1])
        ymin = np.min(network["pore.coords"][:, 1])
        zmax = np.max(network["pore.coords"][:, 2])
        zmin = np.min(network["pore.coords"][:, 2])
        # assign x, y, z masks
        network["pore.xmax"] = network["pore.coords"][:, 0] == xmax
        network["pore.xmin"] = network["pore.coords"][:, 0] == xmin
        network["pore.ymax"] = network["pore.coords"][:, 1] == ymax
        network["pore.ymin"] = network["pore.coords"][:, 1] == ymin
        network["pore.zmax"] = network["pore.coords"][:, 2] == zmax
        network["pore.zmin"] = network["pore.coords"][:, 2] == zmin
        # assign surface labels
        labels = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
        label = np.zeros(network.Np, dtype=bool)
        label[network.pores(labels=labels, mode="or")] = True
        network["pore.surface"] = label
        
        return network


    # stitch network in x-direction
    for x in range(nx-1):
        # make donor by copying network
        donor = op.io.network_from_porespy(copy.deepcopy(network))
        # shift donor network
        donor["pore.coords"][:, 0] += (x+1)*spacing*shape[0]
        # stitch donor network
        op.topotools.stitch(stitched, donor, stitched.pores("xmax"),
                            donor.pores("xmin"))
        # clear all labels
        for label in list(stitched.labels()):
            stitched.pop(label)
        # assign x-max label for subsequent stitching
        xmax = np.max(stitched["pore.coords"][:, 0])
        stitched["pore.xmax"] = stitched["pore.coords"][:, 0] == xmax
        
    # re-assign labels
    stitched = assign_labels(stitched)
    network = op.io.network_from_porespy(copy.deepcopy(stitched))
    
    # stitch network in y-direction
    for y in range(ny-1):
        # make donor by copying network
        donor = op.io.network_from_porespy(copy.deepcopy(network))
        # shift donor network
        donor["pore.coords"][:, 1] += (y+1)*spacing*shape[1]
        # stitch donor network
        op.topotools.stitch(stitched, donor, stitched.pores("ymax"),
                            donor.pores("ymin"))
        # clear all labels
        for label in list(stitched.labels()):
            stitched.pop(label)
        # assign y-max label for subsequent stitching
        ymax = np.max(stitched["pore.coords"][:, 1])
        stitched["pore.ymax"] = stitched["pore.coords"][:, 1] == ymax
    
    # re-assign labels
    stitched = assign_labels(stitched)
    network = op.io.network_from_porespy(copy.deepcopy(stitched))
    
    # stitch network in z-direction
    for z in range(nz-1):
        # make donor by copying network
        donor = op.io.network_from_porespy(copy.deepcopy(network))
        # shift donor network
        donor["pore.coords"][:, 2] += (z+1)*spacing*shape[2]
        # stitch donor network
        op.topotools.stitch(stitched, donor, stitched.pores("zmax"),
                            donor.pores("zmin"))
        # clear all labels
        for label in list(stitched.labels()):
            stitched.pop(label)
        # assign z-max label for subsequent stitching
        zmax = np.max(stitched["pore.coords"][:, 2])
        stitched["pore.zmax"] = stitched["pore.coords"][:, 2] == zmax
    
    # re-assign labels
    stitched = assign_labels(stitched)
    network = op.io.network_from_porespy(copy.deepcopy(stitched))
    
    return network

ns = [7, 7, 3]
net_stitch = stitch_network(net_s, shape_s, spacing, ns)

# FIXME: use tsf_kde = 1 (more accurate results)
# give sizes to stitched throats
# fit gaussian kde to throat sizes
kde = gaussian_kde(tsf, bw_method=0.01)
# sample from kde
mask = np.isnan(net_stitch["throat.diameter"])
tsf_kde = kde.resample(sum(mask))[0]
# assign throat diameters
conns = net_stitch["throat.conns"][mask]
D_s = net_stitch["pore.diameter"]
net_stitch["throat.diameter"][mask] = 1.0 * np.min(np.abs(D_s[conns]), axis=1)


#%% Run Simulation on Stitched Network

'''
# change model so they allow for Dt > Dp
hsf_model = models.hydraulic_size_factors.spheres_and_cylinders
dsf_model = models.diffusive_size_factors.spheres_and_cylinders
tl_model = models.throat_length.spheres_and_cylinders
geo_mods["throat.hydraulic_size_factors"]["model"] = hsf_model
geo_mods["throat.diffusive_size_factors"]["model"] = dsf_model
geo_mods["throat.length"]["model"] = tl_model
'''

# add geometry models
net_stitch.add_model_collection(models=geo_mods)
net_stitch.regenerate_models()

# create phase object
phase_stitch = op.phase.Phase(network=net_stitch)
phase_stitch["throat.contact_angle"] = theta
phase_stitch["throat.surface_tension"] = sigma
phase_stitch["throat.viscosity"] = 1e-3 

# add physics models
phase_stitch.add_model_collection(models=phys_mods)
phase_stitch.regenerate_models()

# stokes flow simulation to estimate permeability
Pin, Pout = 1, 0
shape_stitch = [shape_s[0]*ns[0], shape_s[1]*ns[1], shape_s[2]*ns[2]]  # FIXME: temporary
Ax = shape_stitch[1] * shape_stitch[2] * spacing ** 2
Ay = shape_stitch[0] * shape_stitch[2] * spacing ** 2
Az = shape_stitch[0] * shape_stitch[1] * spacing ** 2
Lx, Ly, Lz = np.array(shape_stitch) * spacing - spacing
mu = phase['pore.viscosity'].max()

# measure K in x-direction
flow_x = op.algorithms.StokesFlow(network=net_stitch, phase=phase_stitch)
flow_x.set_value_BC(pores=net_stitch.pores("xmin"), values=Pin)
flow_x.set_value_BC(pores=net_stitch.pores("xmax"), values=Pout)
flow_x.run()

Q_x = flow_x.rate(pores=net_stitch.pores("xmin"), mode='group')[0]
Kst_x = Q_x * Lx * mu / (Ax * (Pin - Pout))
print(f'K_stitch_x is: {Kst_x} m2')

# measure K in y-direction
flow_y = op.algorithms.StokesFlow(network=net_stitch, phase=phase_stitch)
flow_y.set_value_BC(pores=net_stitch.pores("ymin"), values=Pin)
flow_y.set_value_BC(pores=net_stitch.pores("ymax"), values=Pout)
flow_y.run()

Q_y = flow_y.rate(pores=net_stitch.pores("ymin"), mode='group')[0]
Kst_y = Q_y * Ly * mu / (Ay * (Pin - Pout))
print(f'K_stitch_y is: {Kst_y} m2')

# measure K in z-direction
flow_z = op.algorithms.StokesFlow(network=net_stitch, phase=phase_stitch)
flow_z.set_value_BC(pores=net_stitch.pores("zmin"), values=Pin)
flow_z.set_value_BC(pores=net_stitch.pores("zmax"), values=Pout)
flow_z.run()

Q_z = flow_z.rate(pores=net_stitch.pores("zmin"), mode='group')[0]
Kst_z = Q_z * Lz * mu / (Az * (Pin - Pout))
print(f'K_stitch_z is: {Kst_z} m2')

Kst_avg = np.average([Kst_x, Kst_y, Kst_z])
print(f'K_stitch is: {Kst_avg} m2')

# run drainage
alg_stitch = op.algorithms.Drainage(phase=phase_stitch, network=net_stitch)
alg_stitch.set_inlet_BC(pores=net_stitch.pores("surface"))
alg_stitch.run()

# get pc curve data
data = alg_stitch.pc_curve()
sat_stitch = data.snwp
pc_stitch = data.pc


#%% plot target saturation
plt.figure(3, dpi=500)
ax = plt.gca()  # Get current axes

# re-scale calibrated and sampled data
sat_tra = sat_tra * sat_exp[26]
sat_sam = sat_sam * sat_exp[26]
sat_stitch = sat_stitch * sat_exp[26]

# FIXME: should I add these points?
# pc_tra = np.concatenate((np.array([1.5e5]), pc_tra, np.array([5e6])))
# sat_tra = np.concatenate((np.array([0.0]), sat_tra, np.array([sat_tra[-1]])))

# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3)

# Plot experimental data (green line with hollow markers)
plt.semilogx(pc_exp, sat_exp, label='Experiment',
             color='k', linestyle='solid', linewidth=3,
             marker='o', markersize=4,
             markerfacecolor='k',          # transparent center
             markeredgecolor='k', markeredgewidth=2)

# Plot calibrated data (black line with hollow markers)
plt.semilogx(pc_tra, sat_tra, label='Trained PNM',
             color='g', linestyle='solid', linewidth=3,
             marker='o', markersize=10,
             markerfacecolor='none',          # transparent center
             markeredgecolor='g', markeredgewidth=2)

# Plot sampled data (black line with hollow markers)
plt.semilogx(pc_stitch, sat_stitch, label='Stitched PNM',
             color='orange', linestyle='solid', linewidth=3,
             marker='o', markersize=8,
             markerfacecolor='none',          # transparent center
             markeredgecolor='orange', markeredgewidth=2)

plt.xlabel('Pressure (Pa)', fontsize=18)
plt.ylabel('Saturation', fontsize=18)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(frameon=True, fontsize=16)

# save
plt.savefig('../figures/unperforated_network_pc_curve.png')
plt.show()

#%% plot permeabilities
plt.figure(4, dpi=500)
ax = plt.gca()  # Get current axes

# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3) 

K_fitted = np.array([K_x, K_y, K_z, K_avg])
K_sample = np.array([Ks_x, Ks_y, Ks_z, Ks_avg])
K_stitch = np.array([Kst_x, Kst_y, Kst_z, Kst_avg])
K_target = np.array([2e-16, 2e-16, 2e-16, 2e-16])

x = np.arange(len(K_fitted))
bar_width = 0.25

plt.bar(x, K_target, width=bar_width, label='Target', color='k')
plt.bar(x + bar_width, K_fitted, width=bar_width, label='Trained', color='g')
plt.bar(x + 2 * bar_width, K_stitch, width=bar_width, label='Stitch', color='orange')
plt.ylabel('Permeability (m$^2$)', fontsize=18, fontweight='normal')
plt.xticks(x + bar_width, ['X', 'Y', 'Z', 'Avg'], fontsize=18, fontweight='normal')
plt.yticks(fontsize=18, fontweight='normal')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=16, frameon=True)
plt.tight_layout()
plt.savefig('../figures/unperforated_network_permeabilities.png')
plt.show()

#%% Export

# export to paraview
net_stitch["pore.invasion_sequence"] = alg_stitch["pore.invasion_sequence"]
net_stitch["throat.invasion_sequence"] = alg_stitch["throat.invasion_sequence"]
net_stitch["throat.radius"] = net_stitch["throat.diameter"]/2
op.io.project_to_xdmf(project=net_stitch.project,
                      filename="../paraview/scale_network_1a")

# export to .npz file
np.savez_compressed("../networks/scale_network_1a.npz", **net_s)
