import numpy as np
import matplotlib.pyplot as plt
import openpnm as op
from scipy.integrate import solve_ivp

op.visualization.set_mpl_style()

# get filenames
Ca = 35
zeta = 0
mu_att = 0.0
tf = 2500
y_filename = "data/y" + "_" + str(mu_att) + "_" + str(zeta) + "_" + str(tf) + "s"
x_filename = "data/x" + "_" + str(mu_att) + "_" + str(zeta) + "_" + str(tf) + "s"

# load simulated data dict
y = np.load(y_filename + ".npy")
x = np.load(x_filename + ".npy")

# get outlet concentration
c_out = y[:, 533286]

# get time, use this time to interpolate experiment
t_inter = x

# mixing
t_mix = 120  # s
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
t_exp = data_exp[:, 0] - 32500
c_out_exp = data_exp[:, 1]

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
plt.savefig("../figures/discharge_" + str(mu_att) + "_" + str(zeta) + ".png", dpi=500)
plt.show()
