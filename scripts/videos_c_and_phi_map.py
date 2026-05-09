import numpy as np
import openpnm as op
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation

# mpl.rcParams["animation.ffmpeg_path"] = r"D:\anaconda3\envs\jaxCDI\Library\bin\ffmpeg.exe"

op.visualization.set_mpl_style()

# select params
tf = 1800
cf = 5.37
V_cell = 1.0
zeta = "None"

# choose d
d = 0.0004

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

# get x, y, and z coords
x_coords = coords[:, 0]
y_coords = coords[:, 1]
z_coords = coords[:, 2]

# slice mask (fixed)
slicce = np.isclose(x_coords, spacing * x_sl - spacing/2)
slicce_mi = np.isclose(x_coords, spacing * x_sl)

# choose t
t = 0

# get c, phi, c_mi, phi_d @ t
c_t = c[t, :]
phi_t = phi[t, :]
c_mi_t = c_mi[t, :]
phi_d_t = phi_d[t, :]

# slice y and z
y = y_coords[slicce]
z = z_coords[slicce]
y_mi = y_coords[slicce_mi]
z_mi = z_coords[slicce_mi]

# slice data
c_slice = c_t[slicce]
phi_slice = phi_t[slicce]
c_mi_slice = c_mi_t[slicce_mi]
phi_d_slice = phi_d_t[slicce_mi]

# get masks
macro = net["pore.macropore"][slicce]
perfo = net["pore.perforated"][slicce]
micro = net["pore.micropore"][slicce_mi]


fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200, constrained_layout=True)

# global title
fig.suptitle(f"t = {t*5}s", fontsize=28, fontweight="bold")

# ---- Concentration ----
ax = axes[0, 0]
sc1 = ax.scatter(z[macro], y[macro], c=c_slice[macro],
                 cmap="viridis", s=20, vmin=0, vmax=cf)
sc2 = ax.scatter(z[perfo], y[perfo], c=c_slice[perfo],
                 cmap="viridis", s=2.6e4, vmin=0, vmax=cf)  # 1.6e3, 7.4e3, 1.5e4, 2.6e4

cbar = fig.colorbar(sc1, ax=ax)
cbar.ax.tick_params(labelsize=16)
ax.set_title("Concentration (mM)", fontsize=20, fontweight="bold")
ax.set_aspect("equal")
ax.axis("off")


# ---- Potential ----
ax = axes[0, 1]
sc3 = ax.scatter(z[macro], y[macro], c=phi_slice[macro],
                 cmap="plasma", s=20, vmin=0, vmax=V_cell/2)
sc4 = ax.scatter(z[perfo], y[perfo], c=phi_slice[perfo],
                 cmap="plasma", s=2.6e4, vmin=0, vmax=V_cell/2)

cbar = fig.colorbar(sc3, ax=ax)
cbar.ax.tick_params(labelsize=16)
ax.set_title("Potential (V)", fontsize=20, fontweight="bold")
ax.set_aspect("equal")
ax.axis("off")


# ---- Micro concentration ----
ax = axes[1, 0]
sc5 = ax.scatter(z_mi[micro], y_mi[micro], c=c_mi_slice[micro],
                 cmap="viridis", s=20, vmin=0, vmax=np.max(c_mi))

cbar = fig.colorbar(sc5, ax=ax)
cbar.ax.tick_params(labelsize=16)
ax.set_title("Micro Concentration (mM)", fontsize=20, fontweight="bold")
ax.set_aspect("equal")
ax.axis("off")


# ---- Donnan potential ----
ax = axes[1, 1]
sc6 = ax.scatter(z_mi[micro], y_mi[micro], c=phi_d_slice[micro],
                 cmap="plasma", s=20, vmin=0, vmax=V_cell/2)

cbar = fig.colorbar(sc6, ax=ax)
cbar.ax.tick_params(labelsize=16)
ax.set_title("Donnan Potential (V)", fontsize=20, fontweight="bold")
ax.set_aspect("equal")
ax.axis("off")

plt.show()


# --- Update function ---
def update(frame):
    # update t data
    c_t = c[frame, :]
    phi_t = phi[frame, :]
    c_mi_t = c_mi[frame, :]
    phi_d_t = phi_d[frame, :]
    # slice data
    c_slice = c_t[slicce]
    phi_slice = phi_t[slicce]
    c_mi_slice = c_mi_t[slicce_mi]
    phi_d_slice = phi_d_t[slicce_mi]
    # update scatters
    sc1.set_array(c_slice[macro])
    sc2.set_array(c_slice[perfo])
    sc3.set_array(phi_slice[macro])
    sc4.set_array(phi_slice[perfo])
    sc5.set_array(c_mi_slice[micro])
    sc6.set_array(phi_d_slice[micro])
    # update figure title
    fig.suptitle(f"t = {frame*5}s", fontsize=28, fontweight="bold")
    # return *both* artists that change
    return sc1, sc2, sc3, sc4, sc5, sc6

# ani = FuncAnimation(fig, update, frames=c.shape[0], interval=300, blit=False)
ani = FuncAnimation(fig, update, frames=c.shape[0], interval=300, blit=False)
ani.save(f"../movies/cdi_simulation_{d}_{zeta}.mp4", writer="ffmpeg", fps=6, dpi=200)