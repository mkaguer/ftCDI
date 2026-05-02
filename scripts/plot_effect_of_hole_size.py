import numpy as np
import matplotlib.pyplot as plt
import properties as prpts
import models.numpy.effective_volume as effective_volume
from scipy.integrate import solve_ivp
import collection.numpy as co
import openpnm as op
import pnmlib as pnm

op.visualization.set_mpl_style()

# select params
Ca = 140
tf = 1800
V_cell = 1.0
mu_att = 0.0
cf = 5.37

# choose d
d1 = 0.0001
d2 = 0.0002
d3 = 0.0003
d4 = 0.0004

# get data
x1 = np.load("../data/x" + "_" + str(d1) + "_None_" + str(tf) + "s.npy")
y1 = np.load("../data/y" + "_" + str(d1) + "_None_" + str(tf) + "s.npy", mmap_mode='r')
x2 = np.load("../data/x" + "_" + str(d2) + "_None_" + str(tf) + "s.npy")
y2 = np.load("../data/y" + "_" + str(d2) + "_None_" + str(tf) + "s.npy", mmap_mode='r')
x3 = np.load("../data/x" + "_" + str(d3) + "_None_" + str(tf) + "s.npy")
y3 = np.load("../data/y" + "_" + str(d3) + "_None_" + str(tf) + "s.npy", mmap_mode='r')
x4 = np.load("../data/x" + "_" + str(d4) + "_None_" + str(tf) + "s.npy")
y4 = np.load("../data/y" + "_" + str(d4) + "_None_" + str(tf) + "s.npy", mmap_mode='r')

# load networks
data1 = np.load('../networks/create_full_cell' + f"_{d1}" + ".npz", mmap_mode="r")
data2 = np.load('../networks/create_full_cell' + f"_{d2}" + ".npz", mmap_mode="r")
data3 = np.load('../networks/create_full_cell' + f"_{d3}" + ".npz", mmap_mode="r")
data4 = np.load('../networks/create_full_cell' + f"_{d4}" + ".npz", mmap_mode="r")

# convert to dict
net1 = {key: np.array(data1[key]) for key in data1.files}
net2 = {key: np.array(data2[key]) for key in data2.files}
net3 = {key: np.array(data3[key]) for key in data3.files}
net4 = {key: np.array(data4[key]) for key in data4.files}

# convert to openpnm object
net1 = op.io.network_from_porespy(net1)
net2 = op.io.network_from_porespy(net2)
net3 = op.io.network_from_porespy(net3)
net4 = op.io.network_from_porespy(net4)

# get some properties
rho_sep = prpts.properties["rho_sep"]
rho_mi = prpts.properties["rho_mi"]

# set zeta
net1["throat.zeta"] = 1.0
net2["throat.zeta"] = 1.0
net3["throat.zeta"] = 1.0
net4["throat.zeta"] = 1.0

# assign throat coords
conns1 = net1["throat.conns"]
coords1 = net1["pore.coords"]
t_coords1 = np.diff(coords1[conns1], axis=1).squeeze(1)/2 + coords1[conns1[:, 0]]
net1["throat.coords"] = t_coords1
conns2 = net2["throat.conns"]
coords2 = net2["pore.coords"]
t_coords2 = np.diff(coords2[conns2], axis=1).squeeze(1)/2 + coords2[conns2[:, 0]]
net2["throat.coords"] = t_coords2
conns3 = net3["throat.conns"]
coords3 = net3["pore.coords"]
t_coords3 = np.diff(coords3[conns3], axis=1).squeeze(1)/2 + coords3[conns3[:, 0]]
net3["throat.coords"] = t_coords3
conns4 = net4["throat.conns"]
coords4 = net4["pore.coords"]
t_coords4 = np.diff(coords4[conns4], axis=1).squeeze(1)/2 + coords4[conns4[:, 0]]
net4["throat.coords"] = t_coords4

# set pore thickness for perforated pores (i.e. cylinders)
spacing = 1e-5
net1["pore.thickness"] = spacing
net2["pore.thickness"] = spacing
net3["pore.thickness"] = spacing
net4["pore.thickness"] = spacing

# add geometry models, this is important for zeta that may change!
pnm.models.apply_models(net1,
                        models=co.geometry.macropore.spheres_and_cylinders,
                        domain="macropore")
pnm.models.apply_models(net2,
                        models=co.geometry.macropore.spheres_and_cylinders,
                        domain="macropore")
pnm.models.apply_models(net3,
                        models=co.geometry.macropore.spheres_and_cylinders,
                        domain="macropore")
pnm.models.apply_models(net4,
                        models=co.geometry.macropore.spheres_and_cylinders,
                        domain="macropore")

# calculate effective volume (this is also micropore volume)
V1 = effective_volume.bcc_fast(net1, rho_sep, rho_mi)
V2 = effective_volume.bcc_fast(net2, rho_sep, rho_mi)
V3 = effective_volume.bcc_fast(net3, rho_sep, rho_mi)
V4 = effective_volume.bcc_fast(net4, rho_sep, rho_mi)

# get micropore concentration at t = tf
c1_mi = y1[-1, 2*net1.Np:3*net1.Np]
c2_mi = y2[-1, 2*net2.Np:3*net2.Np]
c3_mi = y3[-1, 2*net3.Np:3*net3.Np]
c4_mi = y4[-1, 2*net4.Np:3*net4.Np]

# get outlet concentration (for discharge curve)
c1_out = y1[:, :net1.Np][:, np.where(net1["pore.outlet"])[0][0]]
c2_out = y2[:, :net2.Np][:, np.where(net2["pore.outlet"])[0][0]]
c3_out = y3[:, :net3.Np][:, np.where(net3["pore.outlet"])[0][0]]
c4_out = y4[:, :net4.Np][:, np.where(net4["pore.outlet"])[0][0]]


def mixing(c_out, t_inter):
    
    # mixing
    t_mix = 240  # s
    def fun(x, y, t_mix):
        # get data
        tp = t_inter
        cp = c_out
        # interpolate data
        c = np.interp(x, tp, cp)
        return (c - y)/t_mix
    soln = solve_ivp(fun, (t_inter[0], t_inter[-1]), y0=c_out[0:1], method="BDF",
                     t_eval=t_inter, args=(t_mix,))
    c_out = soln.y[0, :]
    
    return c_out


c1_out = mixing(c1_out, x1)
c2_out = mixing(c2_out, x2)
c3_out = mixing(c3_out, x3)
c4_out = mixing(c4_out, x4)

# calculate "theoretical" charge capacity
V1_mi = np.sum(V1[net1["pore.micropore"]]) / 2
sac1_theory = Ca * 1e6 * V1_mi * V_cell / 2 / 96485
V2_mi = np.sum(V2[net2["pore.micropore"]]) / 2
sac2_theory = Ca * 1e6 * V2_mi * V_cell / 2 / 96485
V3_mi = np.sum(V3[net3["pore.micropore"]]) / 2
sac3_theory = Ca * 1e6 * V3_mi * V_cell / 2 / 96485
V4_mi = np.sum(V4[net4["pore.micropore"]]) / 2
sac4_theory = Ca * 1e6 * V4_mi * V_cell / 2 / 96485
sac_theory = np.array([sac1_theory, sac2_theory, sac3_theory, sac4_theory])
print(f'Theoretical Capacity of charge: {sac_theory} moles of charge')

# calculate "actual" salt capacity
V1_mi = V1[net1["pore.micropore"]]
sac1_actual = np.sum((c1_mi[net1["pore.micropore"]]/2 - cf*np.exp(mu_att))*V1_mi)
V2_mi = V2[net2["pore.micropore"]]
sac2_actual = np.sum((c2_mi[net2["pore.micropore"]]/2 - cf*np.exp(mu_att))*V2_mi)
V3_mi = V3[net3["pore.micropore"]]
sac3_actual = np.sum((c3_mi[net3["pore.micropore"]]/2 - cf*np.exp(mu_att))*V3_mi)
V4_mi = V4[net4["pore.micropore"]]
sac4_actual = np.sum((c4_mi[net4["pore.micropore"]]/2 - cf*np.exp(mu_att))*V4_mi)
sac_actual = np.array([sac1_actual, sac2_actual, sac3_actual, sac4_actual])
print(f'Total Salt Captured: {sac_actual} moles of charge')

# calculate charge efficiency!
charge_eff = sac_actual/sac_theory*100
print(f'Charge Efficiency: {charge_eff} %')


# plot
plt.figure(1)

labels = ['100 μm', '200 μm', '300 μm', '400 μm']
x = np.arange(len(labels))
width = 0.6

# compute unused capacity
unused_capacity = np.maximum(sac_theory - sac_actual, 0)

fig, ax1 = plt.subplots(figsize=(6, 4))

# --- primary axis (bars) ---
ax1.bar(x, sac_actual, width,
        label='Salt Captured', color='#4C72B0')

ax1.bar(x, unused_capacity, width, bottom=sac_actual,
        label='Unused Capacity', color='#DD8452')

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel('Capacity (mol)')
ax1.set_xlabel('Hole Size')
ax1.set_title('Actual vs Theoretical Capacity')
ax1.set_ylim(0, 2.25e-8)

ax1.grid(axis='y', linestyle='--', alpha=0.5)

# --- secondary axis (efficiency) ---
ax2 = ax1.twinx()

ax2.plot(x, charge_eff, marker='o', linestyle='-',
         color='k', linewidth=2, label='Charge Efficiency')

ax2.set_ylabel('Charge Efficiency (%)')
ax2.set_ylim(74.7, 74.8)  # since it's a percentage

# --- combined legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

plt.tight_layout()
plt.show()

# plot discharge curve
plt.figure(2)
plt.plot(x1, c1_out, label="100 μm")
plt.plot(x2, c2_out, label="200 μm")
plt.plot(x3, c3_out, label="300 μm")
plt.plot(x4, c4_out, label="400 μm")
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("Concentration (mM)", fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.legend(frameon=True, fontsize=14)
plt.ylim([4.4, 5.5])
plt.xlim([0, 1800])
plt.show()
