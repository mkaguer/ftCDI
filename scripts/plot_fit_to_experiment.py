import numpy as np
import matplotlib.pyplot as plt
import openpnm as op
from scipy.integrate import solve_ivp

op.visualization.set_mpl_style()

# get data
Ca = 140
d = 0.0002
tf = 1800
x1 = np.load("../data/x" + "_" + str(d) + "_None_" + str(tf) + "s.npy")
x2 = np.load("../data/x" + "_" + str(d) + "_0_" + str(tf) + "s.npy")
x3 = np.load("../data/x" + "_" + str(d) + "_0.1_" + str(tf) + "s.npy")
x4 = np.load("../data/x" + "_" + str(d) + "_0.5_" + str(tf) + "s.npy")
x5 = np.load("../data/x" + "_" + str(d) + "_1.0_" + str(tf) + "s.npy")
y1 = np.load("../data/y" + "_" + str(d) + "_None_" + str(tf) + "s.npy", mmap_mode='r')
y2 = np.load("../data/y" + "_" + str(d) + "_0_" + str(tf) + "s.npy", mmap_mode='r')
y3 = np.load("../data/y" + "_" + str(d) + "_0.1_" + str(tf) + "s.npy", mmap_mode='r')
y4 = np.load("../data/y" + "_" + str(d) + "_0.5_" + str(tf) + "s.npy", mmap_mode='r')
y5 = np.load("../data/y" + "_" + str(d) + "_1.0_" + str(tf) + "s.npy", mmap_mode='r')

# import network (full cell with labels)
data = np.load('../networks/create_full_cell' + f"_{d}" + ".npz")
data = {key: np.array(data[key]) for key in data.files}

# create openpnm network
net = op.io.network_from_porespy(data)

# get outlet concentrations
c_out1 = y1[:, net.pores("outlet")][:, 0]
c_out2 = y2[:, net.pores("outlet")][:, 0]
c_out3 = y3[:, net.pores("outlet")][:, 0]
c_out4 = y4[:, net.pores("outlet")][:, 0]
c_out5 = y5[:, net.pores("outlet")][:, 0]

# get times, use this time to interpolate experiment
t_inter1 = x1
t_inter2 = x2
t_inter3 = x3
t_inter4 = x4
t_inter5 = x5


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


# perform mixing
c_out1 = mixing(c_out1, t_inter1)
c_out2 = mixing(c_out2, t_inter2)
c_out3 = mixing(c_out3, t_inter3)
c_out4 = mixing(c_out4, t_inter4)
c_out5 = mixing(c_out5, t_inter5)

# get experiment data
data_exp = np.loadtxt("../data/discharge-perforated-cycle10.csv",
                      delimiter=",", skiprows=1)
t_exp = data_exp[:, 0] - 32500
c_out_exp = data_exp[:, 1]

# interpolate experimental data
c_out_exp = np.interp(t_inter1, t_exp, c_out_exp)

# discharge curve
plt.figure(1, dpi=500)
plt.plot(t_inter1, c_out1, color="green", linewidth=2, label=f"No Micropore Resistance")
plt.plot(t_inter3, c_out3, color="purple", alpha=0.25, linewidth=2, label=f"Zeta=0.1")
plt.plot(t_inter4, c_out4, color="purple", alpha=0.5, linewidth=2, label=f"Zeta=0.5")
plt.plot(t_inter5, c_out5, color="purple", alpha=0.75, linewidth=2, label=f"Zeta=1.0")
plt.plot(t_inter1, c_out_exp, color="k", linewidth=2, label="Experiment")
plt.legend(frameon=True)
plt.title("Discharge Curve", fontweight="bold")
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Concentration (mM)", fontsize=14)
plt.savefig("../figures/discharge_" + str(d) + ".png", dpi=500)
plt.show()

# assume flow rate of 1.0 mL/min
Q = 1.0 / 1e6 / 60  # m3/s

# eletrode properties
rho = 0.55  # FIXME: g/cm3, bulk density, this is an estimate!
M_salt = 58.44  # g/mol
lx = 300 * 1e-6
ly = 1.55 / 100
lz = 1.55 / 100
m_electrode = 0.95 * lx * ly * lz * rho * 1e6

# calculate salt captured, model
cf = c_out1[0]
tf = t_inter1[-1]
dt = t_inter1[1] - t_inter1[0]
sac_m = cf * Q * tf - np.sum(c_out1[0:-1]*Q*dt)  # moles of salt
m_salt = M_salt * sac_m * 1000  # mg, FIXME: use 1 or 2
sac_m = m_salt / m_electrode / 2
print(f"Model mSAC: {sac_m} mg/g")

# calculate salt captured, experiment
cf = c_out_exp[0]
tf = t_inter1[-1]
dt = t_inter1[1] - t_inter1[0]
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
plt.bar(1, sac_m, width=0.7, label='PNM', color='green')
plt.bar(2, sac_e, width=0.7, label='Experiment', color='black')
plt.ylabel('mSAC (mg/g)', fontsize=18, fontweight='normal')
plt.yticks(fontsize=16, fontweight='normal')
plt.xticks(np.array([1.0, 2.0]), ['Model', 'Experiment'], fontsize=18, fontweight='normal')
plt.savefig("../figures/sac_" + str(Ca) + ".png", dpi=500)
plt.show()

# make combined figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=500)

# subplot (a)
axes[0].plot(t_inter1, c_out1-0.04, color="green", linewidth=3,
             label="No Micropore Resistance")
axes[0].plot(t_inter3, c_out3-0.04, color="purple", alpha=0.25, linewidth=3,
             label="Zeta=0.1")
axes[0].plot(t_inter4, c_out4-0.04, color="purple", alpha=0.5, linewidth=3,
             label="Zeta=0.5")
axes[0].plot(t_inter5, c_out5-0.04, color="purple", alpha=0.75, linewidth=3,
             label="Zeta=1.0")
axes[0].plot(t_inter1, c_out_exp, color="k", linewidth=3,
             label="Experiment")
axes[0].set_title("(a)", fontweight="bold", fontsize=24)
axes[0].set_xlabel("Time (s)", fontsize=18)
axes[0].set_ylabel("Concentration (mM)", fontsize=18)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(frameon=True, fontsize=14)
axes[0].tick_params(direction='in', labelsize=16, length=5, width=2)

# subplot (b)
axes[1].bar(1, sac_m, width=0.7, label='PNM', color='green')
axes[1].bar(2, sac_e, width=0.7, label='Experiment', color='black')
axes[1].set_ylabel('mSAC (mg/g)', fontsize=18)
axes[1].set_xticks([1, 2])
axes[1].set_xticklabels(['Model', 'Experiment'], fontsize=18)
axes[1].set_title("(b)", fontweight="bold", fontsize=24)
axes[1].grid(True, linestyle='--', alpha=0.5, axis='y')

for spine in axes[0].spines.values():
    spine.set_linewidth(3)

for spine in axes[1].spines.values():
    spine.set_linewidth(3)

axes[1].tick_params(direction='in', labelsize=22, length=5, width=2)

plt.tight_layout()

plt.savefig("../figures/combined_" + str(Ca) + ".png", dpi=500)

plt.show()