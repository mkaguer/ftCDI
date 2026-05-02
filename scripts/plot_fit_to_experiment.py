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
y1 = np.load("../data/y" + "_" + str(d) + "_None_" + str(tf) + "s.npy")
y2 = np.load("../data/y" + "_" + str(d) + "_0_" + str(tf) + "s.npy")
y3 = np.load("../data/y" + "_" + str(d) + "_0.1_" + str(tf) + "s.npy")
y4 = np.load("../data/y" + "_" + str(d) + "_0.5_" + str(tf) + "s.npy")
y5 = np.load("../data/y" + "_" + str(d) + "_1.0_" + str(tf) + "s.npy")

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
plt.figure(1)
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