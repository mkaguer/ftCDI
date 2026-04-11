import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import openpnm as op
from scipy.integrate import solve_ivp
import properties as prpts

op.visualization.set_mpl_style()

# mpl.rcParams["animation.ffmpeg_path"] = r"D:\anaconda3\envs\jaxCDI\Library\bin\ffmpeg.exe"

# get filename
tf = 50
Ca = prpts.properties["Ca"] * 1e-6  # F/mL
filename = "../data/CDI_simulation_1a_" + str(tf) + "s"

# load data dict
data = np.load(filename + ".npz")

# parse out data
x, y, z = data["coords"][:, 0], data["coords"][:, 1], data["coords"][:, 2]
c = data["c"]
c_mi = data["c_mi"]
phi = data["phi"]
phi_d = data["phi_d"]
I = data["I"]
t = data["t"]

# get outlet concentration
c_out = np.average(c[:, data["pore.outlet"]], axis=1)

# scale t based on size!
L_net = np.max(y) - np.min(y) + 1e-5  # FIXME: spacing is 1e-5
L_actual = 7e-4
L_star = L_actual/L_net
ts = L_star ** 2 * t

# shift t, append shift
t_delta = 300
ts += t_delta

# append feed concentration
ts = np.concatenate((np.array([0]), ts))
c_out = np.concatenate((np.array([c_out[0]]), c_out))

# interpolate
t_step = 10
t_inter = np.arange(np.min(ts), np.max(ts) + t_step, t_step)
c_out = np.interp(t_inter, ts, c_out)

# mixing
t_mix = 60 # s
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

# get experiment data
data_exp = np.loadtxt("../data/discharge-perforated-cycle10.csv",
                      delimiter=",", skiprows=1)
t_exp = data_exp[:, 0]
c_out_exp = data_exp[:, 1]

# shift experimental data, start at zero
t_exp -= np.min(t_exp)

# interpolate experimental data
c_out_exp = np.interp(t_inter, t_exp, c_out_exp)



# discharge curve
plt.figure(1)
plt.plot(t_inter, c_out, color="orange", linewidth=3, label=f"PNM (Ca={int(Ca)}F/mL)")
plt.plot(t_inter, c_out_exp, color="k", linewidth=3, label="Experiment")
plt.legend(frameon=True)
plt.title("Discharge Curve", fontweight="bold")
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Concentration (mM)", fontsize=14)
plt.savefig("../figures/discharge_" + str(Ca) + ".png", dpi=500)
plt.show()

# assume flow rate of 0.5 mL/min
Q = 0.5 / 1e6 / 60  # m3/s

# eletrode properties
rho = 0.55  # FIXME: g/cm3, bulk density, this is an estimate!
M_salt = 58.44  # g/mol
lx = 300 * 1e-6
ly = 1.75 / 100
lz = 1.75 / 100
m_electrode = 0.95 * lx * ly * lz * rho * 1e6

# calculate salt captured, model
cf = c_out[0]
tf = t_inter[-1]
dt = t_inter[1] - t_inter[0]
sac_m = cf * Q * tf - np.sum(c_out[0:-1]*Q*dt)  # moles of salt
m_salt = M_salt * sac_m * 1000  # mg, FIXME: use 1 or 2
sac_m = m_salt / m_electrode / 2
print(f"Model mSAC: {sac_m} mg/g")
xx
# calculate salt captured, experiment
cf = c_out_exp[0]
tf = t_inter[-1]
dt = t_inter[1] - t_inter[0]
sac_e = cf * Q * tf - np.sum(c_out_exp[0:-1]*Q*dt)  # moles of salt
m_salt = M_salt * sac_e * 1000  # mg, FIXME: use 1 or 2
sac_e = m_salt / m_electrode / 2
print(f"Experiment mSAC: {sac_e} mg/g")

# plot mSAC
plt.figure(2, dpi=500)
ax = plt.gca()  # Get current axes
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3)
plt.bar(1, sac_m, width=0.7, label='PNM', color='orange')
plt.bar(2, sac_e, width=0.7, label='Experiment', color='black')
plt.ylabel('mSAC (mg/g)', fontsize=18, fontweight='normal')
plt.yticks(fontsize=16, fontweight='normal')
plt.xticks(np.array([1.0, 2.0]), ['Model', 'Experiment'], fontsize=18, fontweight='normal')
plt.savefig("../figures/sac_" + str(Ca) + ".png", dpi=500)
plt.show()

# make I same length as t
I = np.concatenate((np.array([np.nan]), I))

# load current data
data_exp = np.loadtxt("../data/current-perforated-cycle10.csv",
                      delimiter=",", skiprows=1)
t_exp = data_exp[:, 0]
I_exp = data_exp[:, 1]

# interpolate experimental data
I_exp = np.interp(t, t_exp, I_exp)
I_exp -= 1.8  # FIXME: tweaked this number to get it to match

# calculate capacitance
dt = t[1] - t[0]
Vcell = 1.0  # V
Ca_exp = np.sum(I_exp[0:-1]/1000*dt) / Vcell * 2 / m_electrode / 2  # F/g
print(f"Capacitance: {Ca_exp} F/g")

# discharge curve
plt.figure(2)
plt.plot(t, I, color="orange", linewidth=3, label=f"PNM (Ca={Ca}F/mL)")
plt.plot(t, I_exp, color="k", linewidth=3, label="Experiment")
plt.legend(frameon=True)
plt.title("Discharge Curve", fontweight="bold")
plt.xlabel("Time (s)")
plt.ylabel("Current (mA)")
plt.show()
'''
# concentration and potential profiles
fig, ax = plt.subplots(2, 3, figsize=(17, 10))

# get data slice
frames = np.arange(0, 10000, 10)
c = c[:, np.where(z==4.5e-5)[0]]
c_mi = c_mi[:, np.where(z==4.5e-5)[0]]
phi = phi[:, np.where(z==4.5e-5)[0]]
phi_d = phi_d[:, np.where(z==4.5e-5)[0]]
x = x[np.where(z==4.5e-5)[0]]
y = y[np.where(z==4.5e-5)[0]]

# top left: scatter plot
# vmin, vmax = np.min(c), np.max(c)
vmin, vmax = 0, 5.37
sc1 = ax[0, 0].scatter(x*1e3, y*1e3, c=c[0, :], cmap='viridis', s=30, vmin=vmin, vmax=vmax)
ax[0, 0].set_title("Concentration (mM)", fontsize=16, fontweight="bold")
ax[0, 0].set_xlabel("x (mm)", fontsize=14)
ax[0, 0].set_ylabel("y (mm)", fontsize=14)
plt.colorbar(sc1, ax=ax[0, 0])

# top middle: discharge curve
(line1,) = ax[0, 1].plot([], [], color='tab:blue')
ax[0, 1].set_title("Discharge Curve", fontsize=16, fontweight="bold")
ax[0, 1].set_xlabel("Time (s)", fontsize=14)
ax[0, 1].set_ylabel("Concentration (mM)", fontsize=14)
ax[0, 1].set_xlim([0, t.max()])
ax[0, 1].set_ylim([4, 5.5])

# top right: scatter plot
vmin, vmax = np.min(c_mi[~np.isnan(c_mi)]), np.max(c_mi[~np.isnan(c_mi)])
sc2 = ax[0, 2].scatter(x*1e3, y*1e3, c=c_mi[0, :], cmap='viridis', s=30, vmin=vmin, vmax=vmax)
ax[0, 2].set_title("Micropore Concentration (mM)", fontsize=16, fontweight="bold")
ax[0, 2].set_xlabel("x (mm)", fontsize=14)
ax[0, 2].set_ylabel("y (mm)", fontsize=14)
plt.colorbar(sc2, ax=ax[0, 2])

# bottom left: scatter plot
vmin, vmax = np.min(phi), np.max(phi)
sc3 = ax[1, 0].scatter(x*1e3, y*1e3, c=phi[0, :], cmap='viridis', s=30, vmin=vmin, vmax=vmax)
ax[1, 0].set_title("Potential (V)", fontsize=16, fontweight="bold")
ax[1, 0].set_xlabel("x (mm)", fontsize=14)
ax[1, 0].set_ylabel("y (mm)", fontsize=14)
plt.colorbar(sc3, ax=ax[1, 0])

# bottom middle: discharge curve
(line2,) = ax[1, 1].plot([], [], color='tab:blue')
ax[1, 1].set_title("Discharge Curve", fontsize=16, fontweight="bold")
ax[1, 1].set_xlabel("Time (s)", fontsize=14)
ax[1, 1].set_ylabel("Current (C/s)", fontsize=14)
ax[1, 1].set_xlim([0, t.max()])
ax[1, 1].set_ylim([0, 20])

# bottom right: scatter plot
vmin, vmax = np.min(phi_d), np.max(phi_d)
mask = np.isnan(phi_d)
phi_d[mask] = 0  # FIXME: should we use nan? 
sc4 = ax[1, 2].scatter(x*1e3, y*1e3, c=phi_d[0, :], cmap='viridis', s=30, vmin=vmin, vmax=vmax)
ax[1, 2].set_title("Donnan Potential (V)", fontsize=16, fontweight="bold")
ax[1, 2].set_xlabel("x (mm)", fontsize=14)
ax[1, 2].set_ylabel("y (mm)", fontsize=14)
plt.colorbar(sc4, ax=ax[1, 2])

fig.suptitle(f"t = {t[0]:.2f}s", fontsize=20, fontweight="bold", y=0.975)

# --- Update function ---
def update(frame):
    # update scatters
    sc1.set_array(c[frame, :])
    sc2.set_array(c_mi[frame, :])
    sc3.set_array(phi[frame, :])
    sc4.set_array(phi_d[frame, :])
    # update lines
    line1.set_data(t[:frame+1], c_out[:frame+1])
    line2.set_data(t[:frame+1], I[:frame+1])
    # ax[1].relim()           # recompute data limits based on current line data
    # ax[1].autoscale_view()  # rescale view to fit new limits
    # update figure title
    fig.suptitle(f"t = {t[frame]:.2f}s", fontsize=20, fontweight="bold", y=0.975)
    # return *both* artists that change
    return sc1, sc2, sc3, sc4, line1, line2

ani = FuncAnimation(fig, update, frames=frames, interval=300, blit=False)
ani.save(f"../movies/cdi_unperforated_" + str(Ca) + ".mp4", writer="ffmpeg", fps=10, dpi=300, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
'''