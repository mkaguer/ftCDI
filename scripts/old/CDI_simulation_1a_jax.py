"""

First attempt at writing an implicit solver in jax! Does not update A and b
on the fly. We observed divergence after a few time steps.

Doesn't work because we had to update pnm.algorithms

"""
import numpy as _np
import openpnm as op
import pnmlib as pnm
import jax
import jax.numpy as jnp
import collection.jax as co
from jax import config
import time
import models.numpy.effective_volume as effective_volume
import properties as prpts
from collections import ChainMap
from algorithms import implicit_solve
from algorithms import dae_solve_v2 as dae_solve
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

# FIXME: should be able to remove this!
import sys
sys.path.append(r"D:\OneDrive\UW files\code\mypnmlib")

# create blank project dict
proj = {}

# import network (full cell with labels)
data = _np.load('../networks/create_full_cell_1a_test.npz')
data = {key: _np.array(data[key]) for key in data.files}

# convert to jax
net = pnm.io.numpy_to_jax(data)

# add network to project (for organization)
proj["network"] = net

# use this project format:
# when we update network dict it gets automatically changed in project
# add blank dict to project, and changes to that dict will appear in project! 
'''
project = {
    "network": ...,
    "phase": ...,
    "alg1": ...,
    "alg2": ...,
    "alg3": ...,
    }
'''

# get Np and Nt
Np = len(net["pore.coords"])
Nt = len(net["throat.conns"])

# calculate throats coords, for intersecting_cylinders!
conns = net["throat.conns"]
coords = net["pore.coords"]
t_coords = jnp.diff(coords[conns], axis=1).squeeze(1)/2 + coords[conns[:, 0]]
net["throat.coords"] = t_coords

# infer spacing
conns = net["throat.conns"]
coords = net["pore.coords"]
spacing = jnp.max(jnp.abs(coords[conns[0, 0]] - coords[conns[0, 1]]))
spacing = jnp.round(spacing, 10)

# set pore thickness for perforated pores (i.e. cylinders)
net["pore.thickness"] = spacing

# add zeta parameter
mask = net["throat.micropore"]
net["throat.zeta"] = jnp.where(mask, 0.1, 1.0)

# add geometry models, this is important for zeta that may change!
pnm.models.apply_models(net,
                        models=co.geometry.macropore.spheres_and_cylinders,
                        domain="macropore")
pnm.models.apply_models(net,
                        models=co.geometry.micropore.spheres_and_cylinders,
                        domain="micropore")
pnm.models.apply_models(net,
                        models=co.geometry.separator.continuum,
                        domain="separator")
pnm.models.apply_models(net,
                        models=co.geometry.macropore.intersecting_cylinders,
                        domain="perforated")

# retrieve properties
rho_sep = prpts.properties["rho_sep"]
rho_mi = prpts.properties["rho_mi"]
V_cell = prpts.properties["V_cell"]
Ca = prpts.properties["Ca"]
mu_att = prpts.properties["mu_att"]
P_in = prpts.properties["P_in"]
P_out = prpts.properties["P_out"]
cf = prpts.properties["cf"]
T = prpts.properties["T"]

# Quick fix: offload to numpy!
temp = pnm.io.jax_to_numpy(net)
temp = op.io.network_from_porespy(temp)

# calculate effective volume
start = time.time()
eff_vol = effective_volume.bcc_fast(network=temp,
                                    rho_sep=rho_sep,
                                    rho_mi=rho_mi)
stop = time.time()
print(f"time to calc volumes: {stop - start}s")
net["pore.effective_volume"] = jnp.asarray(eff_vol)
del temp

# assign micropore volume
theta = 1.0
net["pore.micro_volume"] = theta * net["pore.effective_volume"]

# create phase dict
phase = {}
proj["phase"] = phase
phase = ChainMap(phase, net)  # chain to net, phase[key] work when key on net
# FIXME: future work: write our own getter function that works like ChainMap but better!
# The problem with ChainMap is that phase returned is NOT proj["phase"]

# set concentration and pressure
phase["pore.concentration"] = jnp.ones(Np) * cf
phase["pore.pressure"] = jnp.ones(Nt)
phase["pore.temperature"] = jnp.ones(Np) * T

# add phase models, I added interpolation to models
pnm.models.apply_models(phase,
                        models=co.phase.macropore,
                        domain="macropore")
pnm.models.apply_models(phase,
                        models=co.phase.micropore,
                        domain="micropore")
pnm.models.apply_models(phase,
                        models=co.phase.separator,
                        domain="separator")
pnm.models.apply_models(phase,
                        models=co.phase.macropore,
                        domain="perforated")

# set "old" concentrations
phase["pore.concentration_old"] = phase["pore.concentration"].copy()
phase["throat.concentration_old"] = phase["throat.concentration"].copy()

# add physics models to phase
pnm.models.apply_models(phase,
                        models=co.physics.perforated,
                        domain="perforated")
pnm.models.apply_models(phase,
                        models=co.physics.macropore,
                        domain="macropore")
pnm.models.apply_models(phase,
                        models=co.physics.micropore,
                        domain="micropore")
pnm.models.apply_models(phase,
                        models=co.physics.separator,
                        domain="separator")

# add properties to phase, these get from imported properties dict!
# set potential
phi = jnp.where(net["pore.anode"], V_cell/2, 0.0)
phi = jnp.where(net["pore.cathode"], -V_cell/2, phi)
phase["pore.potential"] = phi
phase["pore.electrode_potential"] = phi
phase["pore.attraction_term"] = jnp.where(net["pore.micropore"], mu_att, 0.0)
phase["pore.capacitance"] = jnp.where(net["pore.micropore"], Ca, 1.0)
phase["pore.surface_area"] = jnp.ones(Np)*1

# select time step
dt = 0.1
phase["time_step"] = dt

# set iniital guess for donnan potential
phase["pore.donnan_potential"] = jnp.zeros(Np)

# add phi_d and c_mi properties
pnm.models.apply_models(phase,
                        models=co.physics.properties,
                        domain="micropore")

# set old c_mi and phi_d
phase["pore.micro_concentration_old"] = phase["pore.micro_concentration"].copy()
phase["pore.donnan_potential_old"] = phase["pore.donnan_potential"].copy()

# add source term models
pnm.models.apply_models(phase,
                        models=co.physics.sources,
                        domain="micropore")

# stokes flow algorithm
sf = {}
proj["alg1"] = sf
sf["phase"] = "phase"  # conect phase to sf

# write keys to sf
sf["quantity"] = "pore.pressure"
sf["conductance"] = "throat.hydraulic_conductance"

# set BCs
inlet = net["pore.inlet"]
outlet = net["pore.outlet"]
pnm.algorithms.set_BC(proj, "alg1", inlet, bctype='value',
                      bcvalues=P_in, mode="overwrite")
pnm.algorithms.set_BC(proj, "alg1", outlet, bctype='value',
                      bcvalues=P_out, mode="add")

# build A and b
sf["A"] = pnm.algorithms.build_A(proj, "alg1")
sf["b"] = pnm.algorithms.build_b(proj, "alg1")

# apply BCs
sf["A"], sf["b"] = pnm.algorithms.apply_BC(proj, alg="alg1")

# solve
x = pnm.algorithms.solve(proj, "alg1")

# calculate the rate
Q = pnm.algorithms.rate(proj, "alg1", x, pores=inlet)
print(f"PNM Flow Rate: {Q*1e6*60} mL/min")

# scale to cell flow rate
coords = net["pore.coords"]
Ax = (jnp.max(coords[:, 1]) + spacing/2) * (jnp.max(coords[:, 2]) + spacing/2)
Acell = 1.55/100 * 1.55/100  # m2
f = Acell/Ax
print(f"Cell Flow Rate: {f*Q*1e6*60} mL/min")

# regenerate conductance models!
phase["pore.pressure"] = x
co.regenerate_models(proj, phase)

# mass transport algorithm
mt = {}
proj["alg2"] = mt
mt["phase"] = "phase"  # connect phase to mt

# write keys to mt
mt["conductance"] = "throat.mass_conductance"
mt["hydraulic_conductance"] = "throat.hydraulic_conductance"
mt["quantity"] = "pore.concentration"
mt["pressure"] = "pore.pressure"
mt["pore_volume"] = "pore.effective_volume"

# set value BC
pnm.algorithms.set_BC(proj, "alg2", inlet, bctype="value",
                      bcvalues=cf, mode="overwrite")

# set outflow BC
pnm.algorithms.set_outflow_BC(proj, "alg2", pores=outlet, mode="overwrite")

# set source
mt["sources"] = ["mass_source_effective"]
pnm.algorithms.set_source(proj, alg="alg2",
                          pores=net["pore.micropore"],
                          propname="mass_source_effective")

# build A and b
mt["A"] = pnm.algorithms.build_A(proj, "alg2")
mt["b"] = pnm.algorithms.build_b(proj, "alg2")

# apply BCs
mt["A"], mt["b"] = pnm.algorithms.apply_BC(proj, alg="alg2")

# apply sources
mt["A"], mt["b"] = pnm.algorithms.apply_sources(proj, alg="alg2")

# charge transport algorithm
ct = {}
proj["alg3"] = ct
ct["phase"] = "phase"

# write keys to ct
ct["conductance"] = "throat.ionic_conductance"
ct["quantity"] = "pore.potential"
ct["pore_volume"] = "pore.effective_volume"

# set source, charge_source
ct["sources"] = ["charge_source"]
pnm.algorithms.set_source(proj, alg="alg3",
                          pores=net["pore.micropore"],
                          propname="charge_source")

# build A and b
ct["A"] = pnm.algorithms.build_A(proj, "alg3")
ct["b"] = pnm.algorithms.build_b(proj, "alg3")

# apply sources (there are no BCs)
ct["A"], ct["b"] = pnm.algorithms.apply_sources(proj, alg="alg3")


def rhs_mass(y, args):
    A, b, V = args
    dcdt = (-A @ y + b) / V
    return dcdt


# get implicit solver for mass transport
static_argnames = ["fun", "t_span", "dt", "method", "iters", "check_convergence"]
implicit_solve = jax.jit(implicit_solve, static_argnames=static_argnames)

# get dae solver for charge transport
static_argnames = ["t_span", "dt"]
dae_solve = jax.jit(dae_solve, static_argnames=static_argnames)

is_transient=net["pore.micropore"]

'''
# time stepping
t0 = 0
dt = dt
tf = 1.0
# get propeties
c = phase["pore.concentration"]
phi = phase["pore.potential"]
# get volume
V = phase["pore.effective_volume"]
# get capacitance
C = phase["pore.capacitance"]
a = phase["pore.surface_area"]
CaV = C*a*V
for t in jnp.arange(t0+dt, tf+dt, dt):
    print(f"Time: {t}s")
    # get initial condition
    c0 = c.copy()
    phi0 = phi.copy()
    # choose starting c_old and phi_old
    c_old = c.copy()
    phi_old = phi.copy()
    # initialize phi_guess
    phi_guess = phi0.copy()
    # define gummel convergence
    g_res = jnp.array([100, 100])
    g_tol = jnp.array([1e-4, 1e-8])
    g_max_iter = 20
    for g_iter in range(g_max_iter):
        # solve mass balance
        ts, ys = implicit_solve(rhs_mass,
                                t_span=(0, dt),
                                y0=c0,
                                dt=dt,
                                method="euler",
                                args=(mt["A"], mt["b"], V),
                                tol=1e-10,  # change to 1e-4 when scaled up
                                iters=20,
                                check_convergence=False)
        c = ys[-1, :]
        # solve charge balance
        ts, ys = dae_solve(phi0,
                           phi_guess=phi_guess,
                           t_span=(0, dt),
                           dt=dt/10,
                           args=(ct["A"], ct["b"], CaV),
                           is_transient=net["pore.micropore"],
                           tol=1e-6,
                           atol=1e-8,
                           maxiter=100)
        phi = ys[-1, :]
        # update concentration and potential
        # update phi_guess
        phi_guess = phi
        alpha = 0.8
        c = (1-alpha)*c + alpha*c_old
        phi = (1-alpha)*phi + alpha*phi_old
        phase["pore.concentration"] = c
        phase["pore.potential"] = phi
        # regenerate ALL models
        # print(jnp.sum(phase["throat.concentration"]))
        # print(jnp.sum(phase["throat.ionic_conductance"]))
        # print(jnp.sum(phase["pore.mass_source"]["rate"]))
        # print(jnp.sum(jnp.abs(phase["pore.charge_source"]["rate"])))
        co.regenerate_models(proj, phase)
        # print(jnp.sum(phase["throat.concentration"]))
        # print(jnp.sum(phase["throat.ionic_conductance"]))
        # print(jnp.sum(phase["pore.mass_source"]["rate"]))
        # print(jnp.sum(jnp.abs(phase["pore.charge_source"]["rate"])))
        # update A and b
        pnm.algorithms.update_A_and_b(proj, "alg2")
        pnm.algorithms.update_A_and_b(proj, "alg3")
        # break if g_tol reached
        print(f"  Residual: {g_res}")
        if jnp.all(g_res < g_tol):
            print(f"  Convergence criteria met: {g_res}")
            break
        # break if max no. of iterations reached
        if g_iter == g_max_iter - 1:
            break
        # calculate new residual
        g_res = jnp.array([jnp.sum((c - c_old)**2),
                           jnp.sum((phi - phi_old)**2)])
        # print(c)
        # print(phi)
        # update old c and phi
        c_old = c.copy()
        phi_old = phi.copy()
    # store old c_mi and phi_d
    phase["pore.micro_concentration_old"] = phase["pore.micro_concentration"].copy()
    phase["pore.donnan_potential_old"] = phase["pore.donnan_potential"].copy()
    # regenerate models
    co.regenerate_models(proj, phase)
    # update transport algorithms
    pnm.algorithms.update_A_and_b(proj, "alg2")
    pnm.algorithms.update_A_and_b(proj, "alg3")
'''

# time stepping
# choose t0, dt, and tf
t0 = 0
dt = dt
tf = 1.0
t_save = jnp.arange(0, tf + dt, dt)
# get initial condition of ALL properties
c = phase["pore.concentration"]
phi = phase["pore.potential"]
c_mi = phase["pore.micro_concentration"]
phi_d = phase["pore.donnan_potential"]
# get V
V = net["pore.effective_volume"]
# get CaV
C = phase["pore.capacitance"]
a = phase["pore.surface_area"]
CaV = C*a*V
# store solution for first time!
y = jnp.concatenate((c, phi, c_mi, phi_d))
x = jnp.array([t0])
# initialize current
I = []  # FIXME: is this right?
# perform time stepping
for t in jnp.arange(t0 + dt, tf+dt, dt):
    print(f"Time: {t}s")
    # define initial condition for this time step
    c0 = c.copy()
    phi0 = phi.copy()
    # phi0_t = phi0[mask_t]
    # choose c_old and phi_old as initial condition
    c_old = c0.copy()
    phi_old = phi0.copy()
    # initialize phi_guess as phi0
    # phi_guess = phi0
    # define gummel convergence
    g_res = jnp.array([100, 100])
    g_tol = jnp.array([1e-4, 1e-8])
    g_max_iter = 20
    for g_iter in range(g_max_iter):
        print(f"  Gummel Iteration No. {g_iter + 1}")
        # solve mass balance
        # start = time.time()
        ts, ys = implicit_solve(rhs_mass,
                                t_span=(0, dt),
                                y0=c0,
                                dt=dt,
                                method="crank-nicolson",
                                args=(mt["A"], mt["b"], V),
                                tol=1e-4,  # change to 1e-4 when scaled up
                                iters=25,
                                alpha=0.5,
                                check_convergence=False)
        # ys.block_until_ready()
        # stop = time.time()
        # print(f"Mass time: {stop - start}s")
        c = ys[-1, :]
        # solve charge balance
        # start = time.time()
        ts, ys = dae_solve(phi0,
                           u_guess=phi0,
                           t_span=(0, dt),
                           dt=dt,
                           args=(ct["A"], ct["b"], CaV),
                           is_transient=net["pore.micropore"],
                           tol=1e-6,
                           atol=1e-8,
                           maxiter=100)
        # ys.block_until_ready()
        # stop = time.time()
        # print(f"Charge time: {stop - start}s")
        phi = ys[-1, :]
        # update concentration and potential
        phase["pore.concentration"] = c
        phase["pore.potential"] = phi
        # regenerate physics based on c and phi
        co.regenerate_models(proj, phase)
        # update transport algs
        pnm.algorithms.update_A_and_b(proj, "alg2")
        pnm.algorithms.update_A_and_b(proj, "alg3")
        # break if g_tol reached
        print(f"  Residual: {g_res}")
        if jnp.all(g_res < g_tol):
            print(f"  Convergence criteria met: {g_res}")
            break
        # break if max no. of iterations reached
        if g_iter == g_max_iter - 1:
            break
        # calculate new residual
        g_res = jnp.array([jnp.sum((c - c_old)**2),
                           jnp.sum((phi - phi_old)**2)])
        # calcualte phi guess
        # phi_guess = (phi + phi_old)/2
        # updated old c and phi
        c_old = c.copy()
        phi_old = phi.copy()
    # update old concentration
    phase["pore.concentration_old"] = phase["pore.concentration"].copy()
    phase["throat.concentration_old"] = phase["throat.concentration"].copy()
    # regenerate models
    co.regenerate_models(proj, phase)
    # store old c_mi and phi_d
    phase["pore.micro_concentration_old"] = phase["pore.micro_concentration"].copy()
    phase["pore.donnan_potential_old"] = phase["pore.donnan_potential"].copy()
    # regenerate models
    co.regenerate_models(proj, phase)
    # update transport algorithms
    # pnm.algorithms.update_A_and_b(proj, "alg2")
    # pnm.algorithms.update_A_and_b(proj, "alg3")
    # store results if t is in tsave
    if jnp.any(jnp.isclose(t, t_save, atol=1e-10)):
        # get phi_d and c_mi
        c_mi = phase["pore.micro_concentration"]
        phi_d = phase["pore.donnan_potential"]
        # save results as y and x, assume that everything gets saved!
        y = jnp.vstack((y, jnp.concatenate((c, phi, c_mi, phi_d))))
        x = jnp.concatenate((x, jnp.array([t])))
        # calculate the current
        # divide by two because there are two times the number of throats
        # FIXME: watch this when you scale, I think it works but be careful
        mask = net["throat.separator"]
        mask = mask.at[-1].set(False)  # FIXME: assume last
        throats = jnp.where(net["throat.separator"] * mask)
        current = pnm.algorithms.rate(proj, alg="alg3", x=phi,
                                      throats=throats, mode='group')
        I.append(current[0])


# calculate "theoretical" charge capacity
mi_mask = net["pore.micropore"]
V = jnp.sum(net["pore.micro_volume"][mi_mask]) / 2
capacity_theory = Ca * V * V_cell / 2 / 96485
print(f'Theoretical Capacity of charge: {capacity_theory} moles of charge')

# calculate "actual" salt capacity
V_mi = net["pore.micro_volume"][mi_mask]
c_mi = phase["pore.micro_concentration"][mi_mask]/2 - cf
print(f'Total Salt Captured 2: {jnp.sum(c_mi*V_mi)} moles of salt')

# convert data to numpy arrays for plotting
y = _np.asarray(y)
t_save = _np.asarray(t_save)
I = _np.asarray(I)

plt.figure(1)
c_out = _np.average(y[:, 0:Np][:, net["pore.outlet"]], axis=1)
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
data["c"] = y[:, 0:Np]
data["phi"] = y[:, Np:Np*2]
data["c_mi"] = y[:, Np*2:Np*3]
data["phi_d"] = y[:, Np*3:Np*4]
data["t"] = t_save
data["I"] = I
# _np.savez("../data/CDI_simulation_1a_" + str(tf) + "s" + ".npz", **data)