"""

These times were measured using notebooks charge-time-pypardiso,
mass-time-pypardiso, and jax solver-speeds

For jax speeds, took 2nd Gummel iteration on first time step

To get cpu times, I had to comment out preconditioner in newton_krylov and
use M = None

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import openpnm as op

op.visualization.set_mpl_style()


mass_times_cg = np.array([17.64460778236389,
                          0.4079403877258301,
                          1.5305004119873047,
                          1.1313042640686035,
                          0.7307939529418945,
                          0.04706931114196777])
charge_times_cg = np.array([19.051827669143677,
                            0.49213194847106934,
                            2.4904632568359375,
                            2.222822666168213,
                            0.7350387573242188,
                            0.12909221649169922])

# %% AVERAGE TIMES OVER GUMMEl ITERS

# solver times
# [pypardiso 1st, pypardiso 2nd, jax cpu, jax jit cpu, jax gpu, jax jit gpu, jax jit gpu precon] in seconds
t_m_cg = np.array([17.64460778236389,
                   0.4079403877258301,
                   2.204037075950986,
                   1.9737697215307326,
                   0.6592289266132173,
                   0.03497189567202613,
                   0.01747470810299828])
t_c_cg = np.array([19.051827669143677,
                   0.49213194847106934,
                   5.84635641461327,
                   5.022325095676241,
                   0.6997517631167457,
                   0.10740383466084798,
                   0.036458276567004975])
t_m_gm = np.array([17.64460778236389,
                   0.4079403877258301,
                   4.383389529727754,
                   3.775220905031477,
                   1.0619820526668005,
                   0.08983030773344494,
                   0.03192878904796782])
t_c_gm = np.array([19.051827669143677,
                   0.49213194847106934,
                   127.44596709523883,
                   122.74862888881138,
                   3.8959467865171886,
                   2.927033151899065,
                   0.32910609245300293])

# build A and b times
# [jax cpu, jax jit cpu, jax gpu, jax jit gpu] in seconds
t_Ab_m = np.array([1.4376780192057292,
                   0.1787747542063395,
                   0.04308236212957473,
                   0.0026406447092692056])
t_Ab_c = np.array([1.096004429317656,
                   0.17914401917230516,
                   0.036493051619756786,
                   0.002556051526750837])

# build source
# [jax cpu, jax jit cpu, jax gpu, jax jit gpu, jax jit gpu precon] in seconds
t_s_m_cg = np.array([2.6008197352999733,
                     1.4794169721149264,
                     1.473358052117484,
                     0.05027571178617932,
                     0.006484769639514741])
t_s_c_cg = np.array([2.7145475205920993,
                     1.5153200172242665,
                     1.5326891853695823,
                     0.05425646191551572,
                     0.007202330089750744])
t_s_m_gm = np.array([4.4404306298210505,
                     3.3003901072910855,
                     1.842628436210828,
                     0.10017169089544387,
                     0.01670331046694801])
t_s_c_gm = np.array([5.295851639338902,
                     3.6966688746497747,
                     1.805764118830363,
                     0.11134733472551618,
                     0.024384930020286924])

# update ionic conductance
# [jax cpu, jax jit cpu, jax gpu, jax jit gpu] in seconds
t_k = np.array([0.21843739918300084,
                0.028949419657389324,
                0.01243841080438523,
                0.0014759018307640439])


# plot comparison to pypardiso, bicgstab solver
plt.figure(1, dpi=500)
y1 = t_m_cg[[0, 1, 2, 4, 5]]
y2 = t_c_cg[[0, 1, 2, 4, 5]]
labels = ["1st", "2nd", "cpu", "gpu", "jit"]
x = np.arange(len(labels))
colors = ["tab:orange", "tab:orange", "tab:purple", "tab:purple", "tab:purple"]
legend_elements = [
    Patch(facecolor='tab:orange', label='pypardiso'),
    Patch(facecolor='tab:purple', label='JAX')
]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# first subplot
for spine in axes[0].spines.values():
    spine.set_linewidth(2.5)
axes[0].bar(x, y1, color=colors)
axes[0].set_yscale('log')  # log scale
axes[0].tick_params(direction='in', length=6, width=3, labelsize=16)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=16)
axes[0].set_title('a) Mass Solve', fontsize=18, fontweight="normal")
axes[0].legend(handles=legend_elements, frameon=True)
axes[0].grid(True, which='both', axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
axes[0].set_ylabel("Time (s)", fontsize=16)
# second subplot
for spine in axes[1].spines.values():
    spine.set_linewidth(2.5)
axes[1].bar(x, y2, color=colors)
axes[1].set_yscale('log')  # log scale
axes[1].tick_params(direction='in', length=6, width=3, labelsize=16)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=16)
axes[1].set_title('b) Charge Solve', fontsize=18, fontweight="normal")
axes[1].legend(handles=legend_elements, frameon=True)
axes[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
plt.savefig("../figures/solver-speeds-bicgstab.png")


# plot comparison to pypardiso, bicgstab solver
plt.figure(2, dpi=500)
y1 = t_m_gm[[0, 1, 2, 4, 5]]
y2 = t_c_gm[[0, 1, 2, 4, 5]]
labels = ["1st", "2nd", "cpu", "gpu", "jit"]
x = np.arange(len(labels))
colors = ["tab:orange", "tab:orange", "tab:purple", "tab:purple", "tab:purple"]
legend_elements = [
    Patch(facecolor='tab:orange', label='pypardiso'),
    Patch(facecolor='tab:purple', label='JAX')
]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# first subplot
for spine in axes[0].spines.values():
    spine.set_linewidth(2.5)
axes[0].bar(x, y1, color=colors)
axes[0].set_yscale('log')  # log scale
axes[0].tick_params(direction='in', length=6, width=3, labelsize=16)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=16)
axes[0].set_title('a) Mass Solve', fontsize=18, fontweight="normal")
axes[0].legend(handles=legend_elements, frameon=True)
axes[0].grid(True, which='both', axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
axes[0].set_ylabel("Time (s)", fontsize=16)
# second subplot
for spine in axes[1].spines.values():
    spine.set_linewidth(2.5)
axes[1].bar(x, y2, color=colors)
axes[1].set_yscale('log')  # log scale
axes[1].tick_params(direction='in', length=6, width=3, labelsize=16)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=16)
axes[1].set_title('b) Charge Solve', fontsize=18, fontweight="normal")
axes[1].legend(handles=legend_elements, frameon=True)
axes[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
plt.savefig("../figures/solver-speeds-gmres.png")


# plot breakdown
plt.figure(3, dpi=500)
labels = ["cpu", "gpu", "jit", "condition"]
x = np.arange(len(labels))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# first subplot
y1 = t_Ab_m[[0, 2, 3, 3]]  # build A and b
y2 = t_s_m_gm[[0, 2, 3, 4]]  # source term
y3 = t_m_cg[[2, 4, 5, 6]]  # solver
for spine in axes[0].spines.values():
    spine.set_linewidth(2.5)
axes[0].set_yscale('log')  # log scale
axes[0].set_ylim([1e-3, 2e1])
axes[0].tick_params(direction='in', length=6, width=3, labelsize=16)
axes[0].bar(x, y1, label="Build A and b", color="tab:blue")
axes[0].bar(x, y2, bottom=y1, label="Update Source", color="tab:green")
axes[0].bar(x, y3, bottom=y1+y2, label="Solver", color="tab:orange")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=16)
axes[0].grid(True, which='both', axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
axes[0].set_title('a) Mass', fontsize=18, fontweight="normal")
axes[0].legend(frameon=True)
axes[0].set_ylabel("Time (s)", fontsize=16)
# second subplot
y1 = t_Ab_c[[0, 2, 3, 3]]  # build A and b
y2 = t_s_c_gm[[0, 2, 3, 4]]  # source term
y3 = t_c_cg[[2, 4, 5, 6]]  # solver
for spine in axes[1].spines.values():
    spine.set_linewidth(2.5)
axes[1].set_yscale('log')  # log scale
axes[1].set_ylim([1e-3, 2e1])
axes[1].tick_params(direction='in', length=6, width=3, labelsize=16)
axes[1].bar(x, y1, label="Build A and b", color="tab:blue")
axes[1].bar(x, y2, bottom=y1, label="Update Source", color="tab:green")
axes[1].bar(x, y3, bottom=y1+y2, label="Solver", color="tab:orange")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=16)
axes[1].set_title('b) Charge', fontsize=18, fontweight="normal")
axes[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
axes[1].legend(frameon=True)
plt.savefig("../figures/solver-breakdown-bicgstab.png")


# plot breakdown
plt.figure(4, dpi=500)
labels = ["cpu", "gpu", "jit", "condition"]
x = np.arange(len(labels))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# first subplot
y1 = t_Ab_m[[0, 2, 3, 3]]  # build A and b
y2 = t_s_m_gm[[0, 2, 3, 4]]  # source term
y3 = t_m_gm[[2, 4, 5, 6]]  # solver
for spine in axes[0].spines.values():
    spine.set_linewidth(2.5)
axes[0].set_yscale('log')  # log scale
axes[0].set_ylim([1e-3, 2e1])
axes[0].tick_params(direction='in', length=6, width=3, labelsize=16)
axes[0].bar(x, y1, label="Build A and b", color="tab:blue")
axes[0].bar(x, y2, bottom=y1, label="Update Source", color="tab:green")
axes[0].bar(x, y3, bottom=y1+y2, label="Solver", color="tab:orange")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=16)
axes[0].grid(True, which='both', axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
axes[0].set_title('a) Mass', fontsize=18, fontweight="normal")
axes[0].legend(frameon=True)
axes[0].set_ylabel("Time (s)", fontsize=16)
# second subplot
y1 = t_Ab_c[[0, 2, 3, 3]]  # build A and b
y2 = t_s_c_gm[[0, 2, 3, 4]]  # source term
y3 = t_c_gm[[2, 4, 5, 6]]  # solver
for spine in axes[1].spines.values():
    spine.set_linewidth(2.5)
axes[1].set_yscale('log')  # log scale
axes[1].set_ylim([1e-3, 2e2])
axes[1].tick_params(direction='in', length=6, width=3, labelsize=16)
axes[1].bar(x, y1, label="Build A and b", color="tab:blue")
axes[1].bar(x, y2, bottom=y1, label="Update Source", color="tab:green")
axes[1].bar(x, y3, bottom=y1+y2, label="Solver", color="tab:orange")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=16)
axes[1].set_title('b) Charge', fontsize=18, fontweight="normal")
axes[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
axes[1].legend(frameon=True)
plt.savefig("../figures/solver-breakdown-gmres.png")


# %% 2ND TIME OVER GUMMEL ITERS

# [pypardiso 1st, pypardiso 2nd, jax cpu, jax jit cpu, jax gpu, jax jit gpu, jax jit gpu precon] in seconds
t_m_cg = np.array([17.64460778236389,
                   0.4079403877258301,
                   2.791536569595337,
                   2.894580841064453,
                   0.7380788326263428,
                   0.0506749153137207,
                   0.02360224723815918])
t_c_cg = np.array([19.051827669143677,
                   0.49213194847106934,
                   5.897434949874878,
                   5.03666353225708,
                   0.683293342590332,
                   0.11058759689331055,
                   0.05090665817260742])
t_m_gm = np.array([17.64460778236389,
                   0.4079403877258301,
                   5.719867944717407,
                   5.112181901931763,
                   1.0529067516326904,
                   0.12267804145812988,
                   0.031914710998535156])
t_c_gm = np.array([19.051827669143677,
                   0.49213194847106934,
                   127.03911900520325,
                   122.72305822372437,
                   3.922062397003174,
                   2.9205970764160156,
                   0.45574164390563965])

# build A and b times
# [jax cpu, jax jit cpu, jax gpu, jax jit gpu] in seconds
t_Ab_m = np.array([1.4270174503326416,
                   0.16344475746154785,
                   0.04457688331604004,
                   0.0027141571044921875])
t_Ab_c = np.array([0.9974195957183838,
                   0.19212698936462402,
                   0.03458261489868164,
                   0.0026504993438720703])

# build source
# [jax cpu, jax jit cpu, jax gpu, jax jit gpu, jax jit gpu precon] in seconds
t_s_m_cg = np.array([4.546096086502075,
                     2.542302370071411,
                     2.0098066329956055,
                     0.06862497329711914,
                     0.009592533111572266])
t_s_c_cg = np.array([4.547966718673706,
                     1.8655173778533936,
                     1.958979845046997,
                     0.08231163024902344,
                     0.01003122329711914])
t_s_m_gm = np.array([8.535809755325317,
                     4.632161855697632,
                     6.443504095077515,
                     0.12093281745910645,
                     0.011586189270019531])
t_s_c_gm = np.array([7.151159048080444,
                     4.498913049697876,
                     3.3096137046813965,
                     0.1407151222229004,
                     0.05567169189453125])

# update ionic conductance
# [jax cpu, jax jit cpu, jax gpu, jax jit gpu] in seconds
t_k = np.array([0.1550734043121338,
                0.02975606918334961,
                0.011397838592529297,
                0.001470804214477539])
