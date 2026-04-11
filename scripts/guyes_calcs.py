import numpy as np
import matplotlib.pyplot as plt
import openpnm as op

op.visualization.set_mpl_style()

# %% CONCENTRATION

# import discharge guyes data
data = np.loadtxt("../data/discharge-perforated-cycle10.csv",
                  delimiter=",", skiprows=1)
t = data[:, 0] - 9*3600  # subtract first 9 cycles (9 hours)
c_out = data[:, 1]

# get data betweeon 0 and 1800
mask = (t > 0) * (t < 1800)
t = t[mask]
c_out = c_out[mask]

# plot concentration
plt.figure(1)
plt.plot(t, c_out, marker="*")
plt.xlim([0, 1800])

# calculate sac
c_in = c_out[0]
Q = 1 * 1e-6 * 1/60  # m3/s
dt = t[1:] - t[0:-1]
sac = np.sum(c_in * Q * dt) - np.sum(c_out[:-1] * Q * dt)  # moles

# convert to mass
M = 58.44  # g/mol
sac = sac * M * 1000  # mg

# get mass of electrode
rho_e = 0.8 * 1e6  # g/cm3
V_e = 1.75/100 * 1.75/100 * 300 * 1e-6  # m3
m_e = rho_e * V_e * (1.0 - 0.062)
print(f"SAC from mass estimate: {sac/m_e/2} mg/g")

# sac reported
sac_reported = 2.69185  # mg/g, got from engauge
print(f"SAC Reported: {sac_reported} mg/g")

# back calculate to get mass, this is probably best!
m_e_reported = sac/sac_reported/2
print(f"Mass of Electrode Reported: {m_e_reported} g")


# %% CHARGE

# import current guyes data
data = np.loadtxt("../data/current-perforated-cycle10v2.csv",
                  delimiter=",", skiprows=1)
t = data[:, 0] - 9*3600  # subtract first 9 cycles (9 hours)
I = data[:, 1]/1000  # A

# plot current
plt.figure(2)
plt.plot(t, I, marker="*")

# fisrt calculate charge by integrating I vs. t data
dt = t[1:] - t[:-1]
# Q = np.sum(I[:-1] * dt)  # overestimate
Q = np.sum(I[1:] * dt)  # underestimate
# Q = np.sum((I[:-1] - I[-1]) * dt)  # with leak

# calculate capacitance
V_cell = 1.0  # V
C = Q / V_cell / 2 / m_e_reported
print(f"Capacitance: {C} F/g")