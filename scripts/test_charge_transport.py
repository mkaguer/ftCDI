import numpy as np
import openpnm as op
import collection
import models.misc as mods
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import cg, LinearOperator
import pyamg
import time

# create network
spacing = 1e-5
coords = np.array([[0.5, 1.5, 0.5],
                   [1.5, 1.5, 0.5],
                   [2.5, 1.5, 0.5],
                   [0.5, 0.5, 0.5],
                   [1.5, 0.5, 0.5],
                   [2.5, 0.5, 0.5],
                   [1.0, 1.0, 0.5],
                   [2.0, 1.0, 0.5]]) * spacing
conns = np.array([[0, 1],
                  [0, 3],
                  [0, 6],
                  [1, 2],
                  [1, 4],
                  [1, 6],
                  [1, 7],
                  [2, 5],
                  [2, 7],
                  [3, 4],
                  [3, 6],
                  [4, 5],
                  [4, 6],
                  [4, 7],
                  [5, 7]])
net = op.network.Network(coords=coords, conns=conns)
# net = op.network.BodyCenteredCubic(shape=[3, 2], spacing=1e-5)

# set pore labels
net.set_label(label="micropore", pores=[6, 7])
net["pore.macropore"] = ~net["pore.micropore"]
net["pore.cathode"] = net["pore.coords"][:, 0] < 1.5*spacing
net["pore.separator"] = net["pore.coords"][:, 0] == 1.5*spacing
net["pore.anode"] = net["pore.coords"][:, 0] > 1.5*spacing

# set throat labels
throats = net.find_neighbor_throats(pores=net.pores("macropore"), mode='xnor')
net.set_label(label='macropore', throats=throats)
net["throat.micropore"] = ~net["throat.macropore"]

print(net)

# add geometry to macropore
net["pore.diameter@macropore"] = 0.5*spacing
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
del geo_mods["pore.diameter"], geo_mods["pore.seed"], geo_mods["pore.max_size"]
net.add_model_collection(models=geo_mods, domain="macropore")

# add geometry to micropores
net["pore.diameter@micropore"] = 0.25*spacing
del geo_mods["throat.volume"]  # delete so doesn't get added to eff macropore vol
net.add_model_collection(models=geo_mods, domain="micropore")

# regenerate models
net.regenerate_models()

# FIXME: will need new model here when we scale!
# calculate effective volume of micropores, this is hard coded!
Vp = net["pore.volume@macropore"][0]
Vt = net["throat.volume@macropore"][0]
net["pore.effective_volume@micropore"] = 0.2*(spacing**3 - Vp - Vt/2*4)  # FIXME: multiply by porosity
eff_vol_mod = op.models.geometry.pore_volume.effective
net.add_model(propname="pore.effective_volume",
              model=eff_vol_mod,
              domain="macropore")

# create phase object
phase = op.phase.Phase(net)

# add properties to phase object
phase["pore.concentration"] = 100
phase["pore.potential@cathode"] = -0.5
phase["pore.potential@anode"] = 0.5
phase["pore.potential@separator"] = 0.0
phase["pore.electrode_potential@cathode"] = -0.5
phase["pore.electrode_potential@anode"] = 0.5
phase["pore.electrode_potential@separator"] = 0.0
phase["pore.attraction_term"] = 0
phase["pore.capacitance"] = 145 * 1e6
phase["pore.surface_area"] = 1
phase["pore.diffusivity@macropore"] = 1.68e-9
phase["pore.diffusivity@micropore"] = 6e-11
phase["pore.temperature"] = 298

# add physics models for macropore
phys_mods_ma = collection.physics.macropore
del phys_mods_ma["throat.diffusive_conductance"]
del phys_mods_ma["throat.hydraulic_conductance"]
del phys_mods_ma["throat.mass_conductance"]
phys_mods_ma["throat.conductivity"]["concentration"] = "throat.concentration"
phys_mods_ma["pore.conductivity"]["concentration"] = "pore.concentration"
phase.add_model_collection(models=phys_mods_ma, domain="macropore")

# add physics models for micropore
phys_mods_mi = collection.physics.micropore
del phys_mods_mi["throat.mass_conductance"]
del phys_mods_mi["throat.hydraulic_conductance"]
phys_mods_mi["throat.conductivity"]["concentration"] = "throat.concentration"
phys_mods_mi["pore.conductivity"]["concentration"] = "pore.concentration"
phase.add_model_collection(models=phys_mods_mi, domain="micropore")

# regenerate models
phase.regenerate_models()

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

# get dimensions
Ns = np.sum(net["pore.macropore"])
Nt = np.sum(net["pore.micropore"])

# break up b into steady and transient parts
mask_s = net["pore.macropore"]
b_s = b[mask_s]
b_t = b[~mask_s]  # FIXME: b changes so we have to get new every time

# conver A to csr
A_csr = A.tocsr()

# create masks
idx_s = np.where(mask_s)[0]  # indices of macropores (steady)
idx_t = np.where(~mask_s)[0]

A_ss = A_csr[idx_s, :][:, idx_s]  # steady-steady
A_st = A_csr[idx_s, :][:, idx_t]  # steady-transient
A_ts = A_csr[idx_t, :][:, idx_s]  # transient-steady
A_tt = A_csr[idx_t, :][:, idx_t]  # transient-transient

# let's use a preconditioner
ml = pyamg.smoothed_aggregation_solver(A_ss)
M = ml.aspreconditioner()


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
tf = 1
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
    phase["pore.potential@micropore"] = phi_t
    phase["pore.potential@macropore"] = phi_s
    # update donnan potential old
    phase["pore.donnan_potential_old"] = phase["pore.donnan_potential"].copy()
    # regenerate phase models
    phase.regenerate_models()
    # update algorithm
    ct._apply_BCs()
    ct._apply_sources()
    # get new b_t
    b_t = b[~mask_s]
