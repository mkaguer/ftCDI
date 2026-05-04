import numpy as np
import openpnm as op
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

op.visualization.set_mpl_style()

# select params
tf = 1800
cf = 5.37
V_cell = 1.0
zeta = "None"

# choose d
d = 0.0002

# get data
x = np.load("../data/x" + "_" + str(d) + "_" + str(zeta) + "_" + str(tf) + "s.npy")
y = np.load("../data/y" + "_" + str(d) + "_" + str(zeta) + "_" + str(tf) + "s.npy", mmap_mode='r')

# load networks
data = np.load('../networks/create_full_cell' + f"_{d}" + ".npz", mmap_mode="r")
net = {key: np.array(data[key]) for key in data.files}
net = op.io.network_from_porespy(net)

# get c, phi, c_mi, phi_d
c = y[:, :net.Np]
phi = y[:, net.Np:2*net.Np]
c_mi = y[:, 2*net.Np:3*net.Np]
phi_d = y[:, 3*net.Np:]

# get coords and conns
conns = net["throat.conns"]
coords = net["pore.coords"]

# infer spacing
spacing = np.max(np.abs(coords[conns[0, 0]] - coords[conns[0, 1]]))
spacing = np.round(spacing, 10)

# choose where to slice
x_sl = 64  # up to 78

# choose time
t = 100  # 1, 2, 10, 50, 100, 200, 300, 360


#%% plot concentration

# get c @ t
c = c[t, :]

# get x, y, and z coords
x_coords = coords[:, 0]
y_coords = coords[:, 1]
z_coords = coords[:, 2]

# slice mask (fixed)
slicce = np.isclose(x_coords, spacing * x_sl - spacing/2)

# slice data
y = y_coords[slicce]
z = z_coords[slicce]
c_slice = c[slicce]

# grid axes
y_unique = np.unique(y)
z_unique = np.unique(z)

# labels
macro = net["pore.macropore"][slicce]
perfo = net["pore.perforated"][slicce]

# ---- scatter ----
plt.figure(1, dpi=500)

sc1 = plt.scatter(z[macro], y[macro], c=c_slice[macro],
                  cmap='viridis', s=10, vmin=0, vmax=cf)
sc2 = plt.scatter(z[perfo], y[perfo], c=c_slice[perfo],
                  cmap='viridis', s=4e3, vmin=0, vmax=cf)
'''
sc1 = plt.scatter(z[macro], y[macro], c=c_slice[macro],
                  cmap='viridis', s=10, vmin=np.min(c_slice[macro]), vmax=np.max(c_slice[macro]))
sc2 = plt.scatter(z[perfo], y[perfo], c=c_slice[perfo],
                  cmap='viridis', s=4e3, vmin=np.min(c_slice[macro]), vmax=np.max(c_slice[macro]))
'''
cbar = plt.colorbar(sc1, label='Concentration (mM)')
# Set font sizes
cbar.ax.tick_params(labelsize=14)        # tick labels
cbar.set_label('Concentration (mM)', fontsize=18)  # label
plt.xlabel('z')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.title(f"t = {t*5}s", fontsize=20, fontweight="bold")
plt.axis("off")
plt.show()


#%% plot potential

# get phi @ t
phi = phi[t, :]

# get x, y, and z coords
x_coords = coords[:, 0]
y_coords = coords[:, 1]
z_coords = coords[:, 2]

# slice mask (fixed)
slicce = np.isclose(x_coords, spacing * x_sl - spacing/2)

# slice data
y = y_coords[slicce]
z = z_coords[slicce]
phi_slice = phi[slicce]

# grid axes
y_unique = np.unique(y)
z_unique = np.unique(z)

# labels
macro = net["pore.macropore"][slicce]
perfo = net["pore.perforated"][slicce]

# ---- scatter ----
plt.figure(2, dpi=500)
sc1 = plt.scatter(z[macro], y[macro], c=phi_slice[macro],
                  cmap='plasma', s=10, vmin=0.0, vmax=V_cell/2)
sc2 = plt.scatter(z[perfo], y[perfo], c=phi_slice[perfo],
                  cmap='plasma', s=4e3, vmin=0.0, vmax=V_cell/2)
'''
sc1 = plt.scatter(z[macro], y[macro], c=phi_slice[macro],
                  cmap='plasma', s=10, vmin=np.min(phi_slice), vmax=np.max(phi_slice))
sc2 = plt.scatter(z[perfo], y[perfo], c=phi_slice[perfo],
                  cmap='plasma', s=4e3, vmin=np.min(phi_slice), vmax=np.max(phi_slice))
'''
cbar = plt.colorbar(sc1, label='Potential (V)')
# Set font sizes
cbar.ax.tick_params(labelsize=14)        # tick labels
cbar.set_label('Potential (V)', fontsize=18)  # label
plt.xlabel('z')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.title(f"t = {t*5}s", fontsize=20, fontweight="bold")
plt.axis("off")
plt.show()


#%% plot micropore concentration

# get phi @ t
c_mi_max = np.max(c_mi)
c_mi = c_mi[t, :]

# get x, y, and z coords
x_coords = coords[:, 0]
y_coords = coords[:, 1]
z_coords = coords[:, 2]

# slice mask (fixed)
slicce = np.isclose(x_coords, spacing * x_sl)

# slice data
y = y_coords[slicce]
z = z_coords[slicce]
c_mi_slice = c_mi[slicce]

# grid axes
y_unique = np.unique(y)
z_unique = np.unique(z)

# labels
micro = net["pore.micropore"][slicce]  # don't actually NEED

# ---- scatter ----
plt.figure(3, dpi=500)
sc1 = plt.scatter(z[micro], y[micro], c=c_mi_slice[micro],
                  cmap='viridis', s=10, vmin=0.0, vmax=c_mi_max)
'''
sc1 = plt.scatter(z[micro], y[micro], c=c_mi_slice[micro],
                  cmap='viridis', s=10, vmin=np.min(c_mi_slice[micro]), vmax=np.max(c_mi_slice[micro]))
'''
cbar = plt.colorbar(sc1, label='Micro Concentration (mM)')
# Set font sizes
cbar.ax.tick_params(labelsize=14)        # tick labels
cbar.set_label('Micro Concentration (mM)', fontsize=18)  # label
plt.xlabel('z')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.title(f"t = {t*5}s", fontsize=20, fontweight="bold")
plt.axis("off")
plt.show()


#%% plot donnan potential

# get phi @ t
phi_d = phi_d[t, :]

# get x, y, and z coords
x_coords = coords[:, 0]
y_coords = coords[:, 1]
z_coords = coords[:, 2]

# slice mask (fixed)
slicce = np.isclose(x_coords, spacing * x_sl)

# slice data
y = y_coords[slicce]
z = z_coords[slicce]
phi_d_slice = phi_d[slicce]

# grid axes
y_unique = np.unique(y)
z_unique = np.unique(z)

# labels
micro = net["pore.micropore"][slicce]

# ---- scatter ----
plt.figure(4, dpi=500)
sc1 = plt.scatter(z[micro], y[micro], c=phi_d_slice[micro],
                  cmap='plasma', s=10, vmin=0.0, vmax=V_cell/2)
'''
sc1 = plt.scatter(z[micro], y[micro], c=phi_d_slice[micro],
                  cmap='plasma', s=10, vmin=np.min(phi_d_slice[micro]), vmax=np.max(phi_d_slice[micro]))
'''
cbar = plt.colorbar(sc1, label='Donnan Potential (V)')
# Set font sizes
cbar.ax.tick_params(labelsize=14)        # tick labels
cbar.set_label('Donnan Potential (V)', fontsize=18)  # label
plt.xlabel('z')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.title(f"t = {t*5}s", fontsize=20, fontweight="bold")
plt.axis("off")
plt.show()
