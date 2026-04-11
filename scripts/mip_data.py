import numpy as np
import matplotlib.pyplot as plt
import openpnm as op

op.visualization.set_mpl_style()

# load mip data
data = np.loadtxt("../figures/guyes2017b/mip-data.csv",
                  delimiter=",", skiprows=1)

# retrieve data
x = data[:, 0]  # um
y = data[:, 1]  # sat

# properties of mercurcy invasion
sigma = 0.4791  # N/m
theta = 140  # degrees

# transform to sat and Pc data
sat = y/np.max(y)
pc = -4 * sigma * np.cos(theta*np.pi/180) / (x/1e6)  # assume cylinderical tubes

# get and process macro data
x_ma = x[26:]
y_ma = y[26:]
x_ma = np.concatenate((np.array([0.3]), x_ma, np.array([10])))
y_ma = np.concatenate((np.array([y_ma[0]]), y_ma, np.array([0.0])))
sat_ma = y_ma/np.max(y_ma)
pc_ma = -4 * sigma * np.cos(theta*np.pi/180) / (x_ma/1e6) 

# get and process micro data
x_mi = x[0:26]
y_mi = y[0:26]
x_mi = np.concatenate((x_mi, np.array([x_mi[-1]*1.1])))
y_mi = np.concatenate((y_mi, np.array([y_mi[-1]])))
sat_mi = (y_mi - np.min(y_mi))/np.max(y_mi - np.min(y_mi))
pc_mi = -4 * sigma * np.cos(theta*np.pi/180) / (x_mi/1e6)

# plot data
plt.figure(1)
ax = plt.gca()  # Get current axes
# Make the bounding box bold
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(direction='in', length=6, width=3) 
plt.semilogx(pc, sat, 'k-o', label="Guyes et al.", linewidth=4, markersize=12)
plt.semilogx(pc_ma, sat_ma, 'b-o', label="Macro", linewidth=4, markersize=12, markerfacecolor="white")
plt.semilogx(pc_mi, sat_mi, 'y-o', label="Micro", linewidth=4, markersize=12, markerfacecolor="white")
plt.legend(fontsize=15, frameon=True)
plt.yticks(fontsize=18, fontweight='normal')
plt.xticks(fontsize=18, fontweight='normal')
# plt.title('Experiment', fontsize=18, fontweight='semibold')
plt.xlabel(r"Capillary Pressure (Pa)", fontsize=18)
plt.ylabel('Saturation', fontsize=18)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('../figures/mip_data_1a', dpi=500)
plt.show()

# export macro data
pc_ma = np.array([pc_ma])
sat_ma = np.array([sat_ma])
data_ma = np.concatenate((pc_ma, sat_ma), axis=0).T
np.savetxt('../data/mip_data_ma_1a' + '.csv', data_ma, delimiter=',')

# export micro data
pc_mi = np.array([pc_mi])
sat_mi = np.array([sat_mi])
data_mi = np.concatenate((pc_mi, sat_mi), axis=0).T
np.savetxt('../data/mip_data_mi_1a' + '.csv', data_mi, delimiter=',')
