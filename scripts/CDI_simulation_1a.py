import numpy as np
import openpnm as op
import collection
import models.misc as mods
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import cg, LinearOperator
import properties as prpts
import pyamg
import time
import network
import models
import algorithms
import matplotlib.pyplot as plt

op.visualization.set_mpl_style()

# set dimensions
l_sep = 2.714e-5  # should be 190um (Guyes)
d = 3e-5

# load network
data = np.load('../networks/perforated_network_1a_test.npz')
data = {key: np.array(data[key]) for key in data.files}

# convert to openpnm object
net = op.io.network_from_porespy(data)

# infer spacing
conns = net["throat.conns"]
coords = net["pore.coords"]
spacing = np.max(np.abs(coords[conns[0, 0]] - coords[conns[0, 1]]))
spacing = np.round(spacing, 10)

# create full cell
net = network.create_full_cell(net, l_separator=l_sep)

# set pore/throat diameter of separator
net["pore.diameter@separator"] = d
net["throat.diameter@separator"] = d

# re-label perforated throats connected to a macropore as "macropore"
throats = op.topotools.find_interface_throats(net,
                                              P1=net.pores("macropore"),
                                              P2=net.pores("perforated"))
net["throat.perforated"][throats] = False
net["throat.macropore"][throats] = True

# calculate throats coords
conns = net["throat.conns"]
coords = net["pore.coords"]
t_coords = np.diff(coords[conns], axis=1).squeeze(1)/2 + coords[conns[:, 0]]
net["throat.coords"] = t_coords

# set pore thickness for perforated pores (i.e. cylinders)
net["pore.thickness"] = spacing

# set zeta
net["throat.zeta"] = 1.0
net["throat.zeta@micropore"] = 0.1  # this is the one "trick" I need to make work!

# add geometry models to ALL domains
geo_mods_micro = collection.geometry.micropore.spheres_and_cylinders
geo_mods_macro = collection.geometry.macropore.spheres_and_cylinders
geo_mods_separ = collection.geometry.separator.continuum
geo_mods_perfo = collection.geometry.macropore.intersecting_cylinders
geo_mods_test = collection.geometry.interface.spheres_and_cylinders
net.add_model_collection(models=geo_mods_macro,
                         domain="macropore",
                         regen_mode="normal")
net.add_model_collection(models=geo_mods_micro,
                         domain="micropore",
                         regen_mode="normal")
net.add_model_collection(models=geo_mods_separ,
                         domain="separator",
                         regen_mode="normal")
net.add_model_collection(models=geo_mods_perfo,
                         domain="perforated",
                         regen_mode="normal")

# retrieve properties
rho_sep = prpts.properties["rho_sep"]
rho_mi = prpts.properties["rho_mi"]
V_cell = prpts.properties["V_cell"]
Ca = prpts.properties["Ca"]
mu_att = prpts.properties["mu_att"]
P_in = prpts.properties["P_in"]
P_out = prpts.properties["P_out"]

# FIXME: there is a small but negligeable volume added to perforated pores
# calculate effective volume
start = time.time()
eff_vol = models.effective_volume.bcc_fast(network=net,
                                           rho_sep=rho_sep, rho_mi=rho_mi)
stop = time.time()
print(f"time to calc volumes: {stop - start}s")
net["pore.effective_volume"] = eff_vol

# assign micropore volume
theta = 1.0
net["pore.micro_volume"] = theta * net["pore.effective_volume"]

# save as xdmf for visualization
net["throat.radius"] = net["throat.diameter"]/2
D = net["pore.diameter"].copy()
D[net.pores("micropore")] = spacing/10
net["pore.diameter"] = D
project = net.project
op.io.project_to_xdmf(project,
                      filename="../paraview/" + "CDI_simulation_1a_test" + ".xdmf")

# create phase object
phase = op.phase.Phase(network=net)
phase.settings["auto_interpolate"] = False

# initialize concentration, pressure, and potential
cf = prpts.properties["cf"]  # 100  # mM
phase["pore.concentration"] = np.ones(net.Np)*cf
phase["pore.pressure"] = np.ones(net.Np)

# add phase models to ALL domains
phase.add_model_collection(models=collection.phase.macropore,
                           domain="macropore",
                           regen_mode="normal")
phase.add_model_collection(models=collection.phase.micropore,
                           domain="micropore",
                           regen_mode="normal")
phase.add_model_collection(models=collection.phase.separator,
                           domain="separator",
                           regen_mode="normal")
phase.add_model_collection(models=collection.phase.macropore,
                           domain="perforated",
                           regen_mode="normal")

# interpolate to find throat concentration and temperature
phase.add_model(propname="throat.concentration",
                model=op.models.misc.from_neighbor_pores,
                prop="pore.concentration",
                mode="mean")
phase.add_model(propname="throat.temperature",
                model=op.models.misc.from_neighbor_pores,
                prop="pore.temperature",
                mode="mean")

# set "old" concentrations
phase["pore.concentration_old"] = phase["pore.concentration"].copy()
phase["throat.concentration_old"] = phase["throat.concentration"].copy()

# add physics models
phase.add_model_collection(models=collection.physics.macropore,
                           domain="perforated",
                           regen_mode="normal")
phase.add_model_collection(models=collection.physics.macropore,
                           domain="macropore",
                           regen_mode="normal")
phase.add_model_collection(models=collection.physics.micropore,
                           domain="micropore",
                           regen_mode="normal")
phase.add_model_collection(models=collection.physics.separator,
                           domain="separator",
                           regen_mode="normal")

# set potential
phase["pore.potential@cathode"] = -V_cell/2
phase["pore.potential@anode"] = V_cell/2
phase["pore.potential@separator"] = 0

# set electrode potential
phase["pore.electrode_potential@cathode"] = -V_cell/2
phase["pore.electrode_potential@anode"] = V_cell/2
phase["pore.electrode_potential@separator"] = 0

# set attraction term
phase["pore.attraction_term@macropore"] = 0
phase["pore.attraction_term@micropore"] = mu_att
phase["pore.attraction_term@separator"] = 0
phase["pore.attraction_term@perforated"] = 0

# set capacitance
phase["pore.capacitance@macropore"] = 1
phase["pore.capacitance@micropore"] = Ca
phase["pore.capacitance@separator"] = 1
phase["pore.capacitance@perforated"] = 1

# set surface area
phase["pore.surface_area@macropore"] = 1
phase["pore.surface_area@micropore"] = 1
phase["pore.surface_area@separator"] = 1
phase["pore.surface_area@perforated"] = 1

# select time step
dt = 0.1

# add source term models
phase["pore.donnan_potential"] = np.zeros(net.Np)  # initial guess
phase["pore.concentration_old"] = cf
phase.add_model(propname="pore.donnan_potential",
                model=mods.donnan_potential,
                domain="micropore",
                pore_potential="pore.potential",
                pore_electrode_potential="pore.electrode_potential",
                pore_attraction_term="pore.attraction_term",
                pore_temperature="pore.temperature",
                pore_capacitance="pore.capacitance",
                pore_surface_area="pore.surface_area",
                pore_concentration="pore.concentration_old")
phase.add_model(propname="pore.micro_concentration",
                model=mods.micropore_concentration,
                domain="micropore",
                pore_concentration="pore.concentration_old",
                pore_donnan_potential="pore.donnan_potential",
                pore_temperature="pore.temperature",
                pore_attraction_term="pore.attraction_term")
phase["pore.micro_concentration_old"] = phase["pore.micro_concentration"].copy()
phase.add_model(propname="pore.mass_source",
                model=mods.mass_source,
                domain="micropore",
                pore_micro_concentration="pore.micro_concentration",
                pore_micro_concentration_old="pore.micro_concentration_old",
                pore_micro_volume="pore.micro_volume",
                time_step=dt)
phase["pore.donnan_potential_old"] = phase["pore.donnan_potential"].copy()
phase.add_model(propname="pore.charge_source",
                model=mods.charge_source,
                domain="micropore",
                pore_donnan_potential="pore.donnan_potential",
                pore_donnan_potential_old="pore.donnan_potential_old",
                pore_volume="pore.effective_volume",
                pore_capacitance="pore.capacitance",
                pore_surface_area="pore.surface_area",
                time_step=dt)

# add effective mass source
phase.add_model(propname="pore.mass_source_effective",
                model=mods.mass_source_effective,
                pore_mass_source="pore.mass_source",
                pore_concentration="pore.concentration_old",
                pore_volume="pore.effective_volume",
                time_step=dt)

# run stokes flow
sf = op.algorithms.StokesFlow(network=net, phase=phase)
sf.set_BC(pores=net.pores('inlet'), bctype="value", bcvalues=P_in)
sf.set_BC(pores=net.pores('outlet'), bctype="value", bcvalues=P_out)
sf.run()

# calculate the flow rate
Q = sf.rate(pores=net.pores("inlet"), mode="group")[0]
print(f"PNM Flow Rate: {Q*1e6*60} mL/min")

# scale to cell flow rate
Ax = (np.max(coords[:, 1]) + spacing/2) * (np.max(coords[:, 2]) + spacing/2)
Acell = 1.55/100 * 1.55/100  # m2
f = Acell/Ax
print(f"Cell Flow Rate: {f*Q*1e6*60} mL/min")

# create mass transport algorithm
mt = op.algorithms.TransientAdvectionDiffusion(network=net,
                                               phase=phase)
# set settings
mt.settings["conductance"] = "throat.mass_conductance"
mt.settings["quantity"] = "pore.concentration"
mt.settings["pore_volume"] = "pore.effective_volume"

# set inflow BC
phase.add_model(propname="pore.inflow",
                model=mods.inflow,
                cf=cf,
                throat_hydraulic_conductance="throat.hydraulic_conductance",
                pore_pressure="pore.pressure")
phase.regenerate_models()
mt.set_source(pores=net.pores("inlet"), propname="pore.inflow")

# set outflow BC
mt.set_outflow_BC(pores=net.pores("outlet"))

# set source terms
mt.set_source(pores=net.pores("micropore"),
              propname="pore.mass_source_effective")

# Finally, apply BCs and source terms to instantiate A and b
mt._apply_BCs()
mt._apply_sources()

# create charge transport algorithm
# ct = op.algorithms.TransientReactiveTransport(network=net, phase=phase)
ct = algorithms.CustomTransientReactiveTransport(network=net, phase=phase)

# set settings
ct.settings["conductance"] = "throat.ionic_conductance"
ct.settings["quantity"] = "pore.potential"
ct.settings["pore_volume"] = "pore.effective_volume"

# set source terms
ct.set_source(pores=net.pores("micropore"),
              propname="pore.charge_source")

# Finally, apply BCs and source terms to instantiate A and b
ct._apply_BCs()
ct._apply_sources()

# get A and b for charge
Ac = ct.A
bc = ct.b

# break up b into steady and transient parts
mask_t = net["pore.micropore"]
bc_s = bc[~mask_t]
bc_t = bc[mask_t]

# create masks
idx_s = np.where(~mask_t)[0]  # indices of macropores (steady)
idx_t = np.where(mask_t)[0]


def _break_up_A(A, idx_s, idx_t):

    A_ss = A[idx_s, :][:, idx_s]  # steady-steady
    A_st = A[idx_s, :][:, idx_t]  # steady-transient
    A_ts = A[idx_t, :][:, idx_s]  # transient-steady
    A_tt = A[idx_t, :][:, idx_t]  # transient-transient

    return A_ss, A_st, A_ts, A_tt


# break up A for charge transport
Ac_ss, Ac_st, Ac_ts, Ac_tt = _break_up_A(Ac, idx_s, idx_t)

# let's use a preconditioner for charge transport
start = time.time()
ml = pyamg.smoothed_aggregation_solver(Ac_ss)
M = ml.aspreconditioner()
stop = time.time()
print(f"Preconditioner Time: {stop - start}s")


def solve_ss(rhs):

    z, _ = cg(Ac_ss, rhs, M=M)

    return z


def schur_matvec(x):

    y = Ac_tt @ x
    z = solve_ss(Ac_st @ x)
    y -= Ac_ts @ z

    return y


nT = len(idx_t)
L_eff = LinearOperator(shape=(nT, nT),
                       matvec=schur_matvec,
                       dtype=float)


def rhs_charge(t, x):

    # retrieve properties
    C = phase["pore.capacitance@micropore"]
    a = phase["pore.surface_area@micropore"]
    V = phase["pore.effective_volume@micropore"]
    # mat-vec multiplication with schur complement
    y = L_eff @ x  # schur_matvec(x)
    # calcualte dxdt
    dxdt = (bc_t - y)/C/a/V

    return dxdt


def rhs_mass(t, c):

    # FIXME: use alg volume
    # retrieve properties
    V = phase["pore.effective_volume"]
    # get A and b for mass
    Am = mt.A
    bm = mt.b
    # calculate dcdt
    dcdt = (-Am.dot(c) + bm)/V

    return dcdt


# choose t0, dt, and tf
t0 = 0
dt = dt
tf = 50
t_save = np.arange(0, tf + dt, dt)
# get initial condition of ALL properties
c = phase["pore.concentration"]
phi = phase["pore.potential"]
c_mi = phase["pore.micro_concentration"]
phi_d = phase["pore.donnan_potential"]
# store solution for first time!
y = np.concatenate((c, phi, c_mi, phi_d))
x = np.array([t0])
# initialize current
I = []  # FIXME: is this right?
# perform time stepping
for t in np.arange(t0 + dt, tf+dt, dt):
    print(f"Time: {t}s")
    # define initial condition for this time step
    c0 = c.copy()
    phi0 = phi.copy()
    phi0_t = phi0[mask_t]
    # choose c_old and phi_old as initial condition
    c_old = c0.copy()
    phi_old = phi0.copy()
    # define gummel convergence
    g_res = np.array([100, 100])
    g_tol = np.array([1e-4, 1e-4])
    g_max_iter = 10
    for g_iter in range(g_max_iter):
        print(f"  Gummel Iteration No. {g_iter + 1}")
        # solve mass balance
        # start = time.time()
        sol_m = solve_ivp(rhs_mass,
                          t_span=(0, dt),
                          y0=c0,
                          t_eval=[dt],
                          mode="RK45")
        # stop = time.time()
        # print(f"Mass time: {stop - start}s")
        c = sol_m.y[:, -1]  # FIXME: this will violate mass balance!
        # solve charge balance
        # start = time.time()
        sol_c = solve_ivp(rhs_charge,
                          t_span=(0, dt),
                          y0=phi0_t,
                          t_eval=[dt],
                          mode="RK45")
        phi_t = sol_c.y[:, -1]
        phi_s = solve_ss(-Ac_st @ phi_t)
        # stop = time.time()
        # print(f"Charge time: {stop - start}s")
        # update concentration and potential
        phase["pore.concentration"] = c
        phase["pore.potential"][mask_t] = phi_t
        phase["pore.potential"][~mask_t] = phi_s
        # regenerate physics based on c and phi
        # start = time.time()
        phase.regenerate_models()
        # end = time.time()
        # print(f"Regenerate models took {end - start}s")
        # update transport algs
        mt["pore.concentration"] = c
        ct["pore.potential"] = phi
        mt._update_A_and_b()
        ct._update_A_and_b()
        # retrieve A and b for charge
        Ac = ct.A
        bc = ct.b
        # break up A
        Ac_ss, Ac_st, Ac_ts, Ac_tt = _break_up_A(Ac, idx_s, idx_t)
        # break up b
        bc_t = bc[mask_t]
        # break if g_tol reached
        print(f"  Residual: {g_res}")
        if np.all(g_res < g_tol):
            print(f"  Convergence criteria met: {g_res}")
            break
        # break if max no. of iterations reached
        if g_iter == g_max_iter - 1:
            break
        # calculate new residual
        g_res = np.array([np.sum((c - c_old)**2),
                          np.sum((phi - phi_old)**2)])
        # updated old c and phi
        c_old = c.copy()
        phi_old = phi.copy()
    # update old concentration
    phase["pore.concentration_old"] = phase["pore.concentration"].copy()
    phase["throat.concentration_old"] = phase["throat.concentration"].copy()
    # regenerate models
    phase.regenerate_models()
    # store old c_mi and phi_d
    phase["pore.micro_concentration_old"] = phase["pore.micro_concentration"].copy()
    phase["pore.donnan_potential_old"] = phase["pore.donnan_potential"].copy()
    # regenerate models
    phase.regenerate_models()
    # store results if t is in tsave
    if np.any(np.isclose(t, t_save, atol=1e-10)):
        # get phi_d and c_mi
        c_mi = phase["pore.micro_concentration"]
        phi_d = phase["pore.donnan_potential"]
        # save results as y and x, assume that everything gets saved!
        y = np.vstack((y, np.concatenate((c, phi, c_mi, phi_d))))
        x = np.concatenate((x, np.array([t])))
        # calculate the current
        # divide by two because there are two times the number of throats
        # FIXME: watch this when you scale, I think it works but be careful
        throats = net.throats("separator")
        current = ct.rate(throats=throats, mode="group")/2
        I.append(current[0])

# see cdi_simulation_1a branch for other calcs
# calculate "theoretical" charge capacity
V = np.sum(net["pore.micro_volume@micropore"]) / 2
capacity_theory = Ca * V * V_cell / 2 / 96485
print(f'Theoretical Capacity of charge: {capacity_theory} moles of charge')

# calculate "actual" salt capacity
V_mi = net["pore.micro_volume@micropore"]
c_mi = phase["pore.micro_concentration@micropore"]/2 - cf
print(f'Total Salt Captured 2: {np.sum(c_mi*V_mi)} moles of salt')

plt.figure(1)
c_out = np.average(y[:, 0:net.Np][:, net["pore.outlet"]], axis=1)
plt.plot(t_save, c_out, label="outlet")
plt.legend(frameon=True)
plt.title("Discharge Curve", fontweight="bold")
plt.xlabel("Time (s)")
plt.ylabel("Concentration (mM)")

# plot current
plt.figure(2)
plt.plot(t_save[1:], I, label="Discharge")
plt.legend(frameon=True)
plt.title("Discharge Curve", fontweight="bold")
plt.xlabel("Time (s)")
plt.ylabel("Current (C/s)")

# save data
data = {}
data["coords"] = net["pore.coords"]
data["pore.macropore"] = net["pore.macropore"]
data["pore.micropore"] = net["pore.micropore"]
data["pore.separator"] = net["pore.separator"]
data["pore.inlet"] = net["pore.inlet"]
data["pore.outlet"] = net["pore.outlet"]
data["c"] = y[:, 0:net.Np]
data["phi"] = y[:, net.Np:net.Np*2]
data["c_mi"] = y[:, net.Np*2:net.Np*3]
data["phi_d"] = y[:, net.Np*3:net.Np*4]
data["t"] = t_save
data["I"] = I
np.savez("../data/CDI_simulation_1a_" + str(tf) + "s" + ".npz", **data)
