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

op.visualization.set_mpl_style()

# set dimensions
l_sep = 8e-5  # should be 190um (Guyes)
d = 5e-5

# load network
data = np.load('../networks/perforated_network_1a.npz')
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

# add geometry models to ALL domains
geo_mods_micro = collection.geometry.micropore.spheres_and_cylinders
geo_mods_macro = collection.geometry.macropore.spheres_and_cylinders
geo_mods_separ = collection.geometry.separator.continuum
geo_mods_perfo = collection.geometry.macropore.intersecting_cylinders
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
P_in = prpts.properties["P_in"]  # FIXME: not using these currently
P_out = prpts.properties["P_out"]

# FIXME: this is SLOW, Fix!
# calculate effective volume
eff_vol = models.effective_volume.bcc(network=net,
                                      rho_sep=rho_sep, rho_mi=rho_mi)
net["pore.effective_volume"] = eff_vol

'''
# save as xdmf for visualization
net["throat.radius"] = net["throat.diameter"]/2
D = net["pore.diameter"].copy()
D[net.pores("micropore")] = spacing/10
net["pore.diameter"] = D
project = net.project
op.io.project_to_xdmf(project,
                      filename="../paraview/" + "CDI_simulation_1a" + ".xdmf")
'''

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

# add donnan potential model
phase["pore.donnan_potential_old"] = np.zeros(net.Np)  # guess
phase.add_model(propname="pore.donnan_potential",
                model=mods.donnan_potential)

# select time step
dt = 0.01

# add source term model
phase.add_model(propname="pore.charge_source",
                model=mods.charge_source,
                domain="micropore",
                pore_donnan_potential="pore.donnan_potential",
                pore_donnan_potential_old="pore.donnan_potential_old",
                pore_volume="pore.effective_volume",
                pore_capacitance="pore.capacitance",
                pore_surface_area="pore.surface_area",
                time_step=dt)

# create charge transport algorithm
ct = op.algorithms.TransientReactiveTransport(network=net, phase=phase)
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

# get A and b matrix
A = ct.A
b = ct.b

# break up b into steady and transient parts
mask_t = net["pore.micropore"]
b_s = b[~mask_t]
b_t = b[mask_t]  # FIXME: b changes so we have to get new every time

# conver A to csr
start = time.time()
A_csr = A.tocsr()
stop = time.time()
print(f"Time to convert to csr: {stop - start}s")

# create masks
idx_s = np.where(~mask_t)[0]  # indices of macropores (steady)
idx_t = np.where(mask_t)[0]

A_ss = A_csr[idx_s, :][:, idx_s]  # steady-steady
A_st = A_csr[idx_s, :][:, idx_t]  # steady-transient
A_ts = A_csr[idx_t, :][:, idx_s]  # transient-steady
A_tt = A_csr[idx_t, :][:, idx_t]  # transient-transient

# let's use a preconditioner
start = time.time()
ml = pyamg.smoothed_aggregation_solver(A_ss)
M = ml.aspreconditioner()
stop = time.time()
print(f"Preconditioner Time: {stop - start}s")


def solve_ss(rhs):

    z, _ = cg(A_ss, rhs, M=M)

    return z


def schur_matvec(x):

    y = A_tt @ x
    z = solve_ss(A_st @ x)
    y -= A_ts @ z

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
    dxdt = (b_t - y)/C/a/V

    return dxdt


# perform time stepping
t0 = 0
dt = dt
tf = 0.1
for t in np.arange(t0 + dt, tf+dt, dt):
    print(f"Simulation time: {t}s")
    # get phi0
    phi0_t = phase["pore.potential@micropore"].copy()
    # solve charge balance
    start = time.time()
    sol = solve_ivp(rhs_charge,
                    t_span=(0, dt),
                    y0=phi0_t,
                    t_eval=[dt])
    phi_t = sol.y[:, -1]
    phi_s = solve_ss(-A_st @ phi_t)
    stop = time.time()
    print(f"Time: {stop - start}s")
    # update potential
    phase["pore.potential"][mask_t] = phi_t
    phase["pore.potential"][~mask_t] = phi_s
    # update donnan potential old
    phase["pore.donnan_potential_old"] = phase["pore.donnan_potential"].copy()
    # regenerate phase models
    phase.regenerate_models()
    # update algorithm
    ct._apply_BCs()
    ct._apply_sources()
    # get new b_t
    b_t = b[mask_t]
