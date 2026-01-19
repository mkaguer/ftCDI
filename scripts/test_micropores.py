"""
The purpose of this script is to test modelling an extra transport length
from the macropore to the unresolved microporosity.

Created by: Mike McKague
Date: July 2, 2025
"""

import openpnm as op
import numpy as np
import models.misc as mods
from scipy.integrate import solve_ivp
from scipy_dae.integrate import solve_dae
import matplotlib.pyplot as plt

op.visualization.set_mpl_style()

# create network
spacing = 1e-4
coords = np.array([[0.5, 0.5, 0.5],
                   [1.5, 0.5, 0.5],
                   [2.5, 0.5, 0.5],
                   [3.5, 0.5, 0.5],
                   [4.5, 0.5, 0.5],
                   [0.5, 1.0, 0.5],
                   [1.5, 1.0, 0.5],
                   [3.5, 1.0, 0.5],
                   [4.5, 1.0, 0.5]])
conns = np.array([[0, 1],
                  [0, 5],
                  [1, 2],
                  [1, 6],
                  [2, 3],
                  [3, 4],
                  [3, 7],
                  [4, 8]])
net = op.network.Network(coords=coords*spacing, conns=conns)

# set labels
pores = net['pore.coords'][:, 0] == 2.5*spacing
throats = net.find_neighbor_throats(pores=pores, asmask=True)
net.set_label("separator", pores=pores, throats=throats)
net.set_label("electrode", pores=~pores, throats=~throats)

# set cathode label
pores = net['pore.coords'][:, 0] < 2.5*spacing
throats = net.find_neighbor_throats(pores=pores, asmask=True)
throats = net.filter_by_label(throats=throats, labels="electrode")
net.set_label("cathode", pores=pores, throats=throats)

# set anode label
pores = net['pore.coords'][:, 0] > 2.5*spacing
throats = net.find_neighbor_throats(pores=pores, asmask=True)
throats = net.filter_by_label(throats=throats, labels="electrode")
net.set_label("anode", pores=pores, throats=throats)

# set micropore label
pores = net['pore.coords'][:, 1] == 1.0*spacing
throats = net.find_neighbor_throats(pores=pores, asmask=True)
net.set_label("micropore", pores=pores, throats=throats)

# set macropore label
pores = net.pores(labels=["separator", "micropore"], mode="nor")
throats = net.throats(labels=["separator", "micropore"], mode="nor")
net.set_label("macropore", pores=pores, throats=throats)

# set inlet/outlet label
net.set_label(label="inlet", pores=[0])
net.set_label(label="outlet", pores=[4])

# add geometry models to macropores
geo_models = op.models.collections.geometry.cubes_and_cuboids.copy()
geo_models["pore.seed"]["num_range"] = [1.0, 1.0]  # this will fix diameter!
geo_models["throat.diameter"]["factor"] = 1.0
net.add_model_collection(models=geo_models, domain="macropore")
net.regenerate_models()

# add geometry models to separator
net["throat.diameter@separator"] = spacing
net["throat.length@separator"] = spacing
net.add_model(propname='throat.diffusive_size_factors',
              model=mods.continuum_size_factor,
              domain='separator',
              throat_diameter='throat.diameter',
              throat_length='throat.length')
net.add_model(propname='throat.hydraulic_size_factors',
              model=mods.continuum_size_factor,
              domain='separator',
              throat_diameter='throat.diameter',
              throat_length='throat.length')
net['pore.volume@separator'] = 2 * spacing ** 3
net["throat.volume@separator"] = 0

# calculate effective volume
net["throat.volume@micropore"] = 0
vol_model = op.models.geometry.pore_volume.effective
net.add_model(propname="pore.volume_effective",
              model=vol_model,
              domain="all")
net["pore.volume_actual"] = net["pore.volume_effective"].copy()

# add geometry models to micropores, wrote custom model!
# FIXME: micropore must be pore 2!
net["throat.diameter@micropore"] = 0.5 * spacing
net["pore.diameter@micropore"] = 1e-16
net.add_model(propname="throat.diffusive_size_factors",
              model=mods.micropore_size_factor,
              domain="micropore",
              throat_diameter='throat.diameter',
              pore_diameter='pore.diameter')
net.add_model(propname="throat.hydraulic_size_factors",
              model=op.models.misc.constant,
              domain="micropore",
              value=1e-16)  # FIXME: should be smaller!
net.add_model(propname="throat.length",
              model=op.models.geometry.throat_length.cubes_and_cuboids,
              domain="micropore",
              pore_diameter="pore.diameter",
              throat_diameter="throat.diameter")
net.add_model(propname="throat.volume",
              model=op.models.geometry.throat_volume.cuboid,
              domain="micropore",
              throat_diameter="throat.diameter",
              throat_length="throat.length")

# add volumes to micropores
# FIXME: did not subtract out micropore volume overlapping separator
pores = net.pores("micropore")
throats = net.find_neighbor_throats(pores=pores, flatten=False)
throats = np.array(throats)
Vt = net["throat.volume"][throats]
net["pore.volume_actual@micropore"] = np.sum(Vt, axis=1)

# subtract continuum volume that overlaps with electrode pore
pores = net['throat.conns@separator'].flatten()
pores = pores[np.isin(pores, net.pores('electrode'))]
mask = np.isin(np.arange(net.Np), pores)
net["pore.volume_actual"][mask] -= 0.5 * net['pore.volume'][mask]

# multiply seperator volume by porosity of separator
rho_sep = 0.75
net["pore.volume_actual"][net.pores("separator")] *= rho_sep

# create phase object
phase = op.phase.Phase(network=net)
phase.settings["auto_interpolate"] = False

# set properties
Vcell = 1  # V
cf = 100  # mol/m3
phase["pore.concentration"] = cf
phase["pore.potential@anode"] = Vcell/2 
phase["pore.potential@cathode"] = -Vcell/2
phase["pore.potential@separator"] = 0 
phase["pore.electrode_potential@cathode"] = -Vcell/2
phase["pore.electrode_potential@anode"] = Vcell/2
phase["pore.electrode_potential@separator"] = 0
phase["pore.pressure"] = 101325  # Pa
phase["pore.temperature"] = 298  # K

# concentration, pressure, temperature we do want to interpolate
phase.add_model(propname="throat.concentration",
                model=op.models.misc.from_neighbor_pores,
                prop="pore.concentration",
                mode="mean")
phase.add_model(propname="throat.pressure",
                model=op.models.misc.from_neighbor_pores,
                prop="pore.pressure",
                mode="mean")
phase.add_model(propname="throat.temperature",
                model=op.models.misc.from_neighbor_pores,
                prop="pore.temperature",
                mode="mean")

# add macropore properties
mu = 1e-3  # Pa s
D = 1.68e-9  # m/s
phase["pore.viscosity@macropore"] = mu
phase["throat.viscosity@macropore"] = mu    
phase["pore.diffusivity@macropore"] = D
phase["throat.diffusivity@macropore"] = D
phase["pore.attraction_term@macropore"] = 0
phase["pore.capacitance@macropore"] = 1  # (145 F/mL, Guyes 2017)
phase["pore.surface_area@macropore"] = 1  # lumped with capacitance

# add micropore properties
epsilon_mi = 0.1375
tau_mi = 1/epsilon_mi**(0.5)
D_eff = D*epsilon_mi/tau_mi
Km = D  # FIXME: use a mass transfer coefficient, some fitting factor!
mu_att = 0  # FIXME: add model to calcualte from E/cmi,ions
phase["pore.viscosity@micropore"] = mu
phase["throat.viscosity@micropore"] = mu  
phase["pore.diffusivity@micropore"] = Km # epsilon_mi * D_eff  # FIXME: Mo thinks not right!
phase["throat.diffusivity@micropore"] = Km # epsilon_mi * D_eff  # FIXME: Mo thinks not right!
phase["pore.attraction_term@micropore"] = mu_att
phase["pore.capacitance@micropore"] = 145 * 100 ** 3  # (145 F/mL, Guyes 2017)
phase["pore.surface_area@micropore"] = 1  # lumped with capacitance

# add separator properties
K = 1e-11  # m2
epsilon_sep = rho_sep
tau_sep = 1/rho_sep**(0.5)
D_eff = D*epsilon_sep/tau_sep
phase["pore.viscosity@separator"] = mu/K
phase["throat.viscosity@separator"] = mu/K
phase["pore.diffusivity@separator"] = epsilon_sep * D_eff  # FIXME: Mo thinks not right!
phase["throat.diffusivity@separator"] = epsilon_sep * D_eff  # FIXME: Mo thinks not right!
phase["pore.attraction_term@separator"] = 0
phase["pore.capacitance@separator"] = 1  # (145 F/mL, Guyes 2017)
phase["pore.surface_area@separator"] = 1  # lumped with capacitance


# add conductivity model
phase["pore.concentration_old"] = phase["pore.concentration"].copy()
phase.add_model(propname="pore.conductivity",
                model=mods.conductivity,
                domain="all",
                concentration="pore.concentration_old",
                diffusivity="pore.diffusivity",
                temperature="pore.temperature")
phase["throat.concentration_old"] = phase["throat.concentration"].copy()
phase.add_model(propname="throat.conductivity",
                model=mods.conductivity,
                domain="all",
                concentration="throat.concentration_old",
                diffusivity="throat.diffusivity",
                temperature="throat.temperature")

# add hydraulic conductance to all
gh_mod = op.models.physics.hydraulic_conductance.generic_hydraulic
phase.add_model(propname='throat.hydraulic_conductance',
                model=gh_mod,
                domain="all",
                pore_viscosity='pore.viscosity',
                throat_viscosity='throat.viscosity',
                size_factors='throat.hydraulic_size_factors')

# add mass conductance models
gd_mod = op.models.physics.diffusive_conductance.generic_diffusive
phase.add_model(propname='throat.diffusive_conductance',
                model=gd_mod,
                domain="macropore",
                pore_diffusivity="pore.diffusivity",
                throat_diffusivity="throat.diffusivity",
                size_factors="throat.diffusive_size_factors")
phase.add_model(propname='throat.diffusive_conductance',
                model=gd_mod,
                domain="separator",
                pore_diffusivity="pore.diffusivity",
                throat_diffusivity="throat.diffusivity",
                size_factors="throat.diffusive_size_factors")
# FIXME: need to write custom mass conductance that is Nt by 2
phase.add_model(propname='throat.mass_conductance',
                model=mods.generic_diffusive,
                domain="micropore",
                pore_diffusivity="pore.diffusivity",
                throat_diffusivity="throat.diffusivity",
                size_factors="throat.diffusive_size_factors")
gad_mod = op.models.physics.ad_dif_conductance.ad_dif
phase.add_model(propname='throat.mass_conductance',
                model=gad_mod,
                domain="macropore",
                pore_pressure='pore.pressure',
                throat_hydraulic_conductance='throat.hydraulic_conductance',
                throat_diffusive_conductance='throat.diffusive_conductance',
                s_scheme='powerlaw')
phase.add_model(propname='throat.mass_conductance',
                model=gad_mod,
                domain="separator",
                pore_pressure='pore.pressure',
                throat_hydraulic_conductance='throat.hydraulic_conductance',
                throat_diffusive_conductance='throat.diffusive_conductance',
                s_scheme='powerlaw')

# add charge conductance model
phase.add_model(propname='throat.ionic_conductance',
                model=gd_mod,
                domain="all",
                pore_diffusivity="pore.conductivity",
                throat_diffusivity="throat.conductivity",
                size_factors="throat.diffusive_size_factors")

# # visualize network in paraview
net['pore.radius'] = net['pore.diameter']/2
net['pore.radius@micropore'] = spacing/15  # so we can see micropore nodes!
net['throat.radius'] = net['throat.diameter']/2
net["pore.color"] = net["pore.macropore"].astype(int)*1
net["pore.color"] += net["pore.separator"].astype(int)*2
net["pore.color"] += net["pore.micropore"].astype(int)*4
net["throat.color"] = net["throat.macropore"].astype(int)*1
net["throat.color"] += net["throat.separator"].astype(int)*2
net["throat.color"] += net["throat.micropore"].astype(int)*3
proj = net.project
op.io.project_to_xdmf(project=proj, filename='../paraview/test_micropores')

# choose time step
dt = 0.01  # FIXME: change when we change spacing
# I could decreas dt when I added extra diffusive length

# set donnan potential old since it is used as initial guess
phase["pore.donnan_potential_old"] = np.zeros(net.Np)

# add source term models
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
                pore_micro_volume="pore.volume_actual",
                time_step=dt)
phase["pore.donnan_potential_old"] = phase["pore.donnan_potential"].copy()
phase.add_model(propname="pore.charge_source",
                model=mods.charge_source,
                domain="micropore",
                pore_donnan_potential="pore.donnan_potential",
                pore_donnan_potential_old="pore.donnan_potential_old",
                pore_volume="pore.volume_actual",
                pore_capacitance="pore.capacitance",
                pore_surface_area="pore.surface_area",
                time_step=dt)

# run stokes flow
sf = op.algorithms.StokesFlow(network=net, phase=phase)
sf.set_BC(pores=net.pores('inlet'), bctype="value", bcvalues=1)
sf.set_BC(pores=net.pores('outlet'), bctype="value", bcvalues=1)
sf.run()  # This will update ad-dif conductance

# create mass transport algorithm
mt = op.algorithms.TransientAdvectionDiffusion(network=net, phase=phase)
mt.settings["conductance"] = "throat.mass_conductance"
mt.settings["quantity"] = "pore.concentration"
mt.settings["pore_volume"] = "pore.volume_actual"

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
mt.set_source(pores=net.pores("micropore"), propname="pore.mass_source")

# create charge transport algorithm
ct = op.algorithms.TransientReactiveTransport(network=net, phase=phase)
ct.settings["conductance"] = "throat.ionic_conductance"
ct.settings["quantity"] = "pore.potential"
ct.settings["pore_volume"] = "pore.volume_actual"

# set source terms
ct.set_source(pores=net.pores("micropore"), propname="pore.charge_source")

# Finally, apply BCs and source terms to instantiate A and b
mt._apply_BCs()
ct._apply_BCs()
mt._apply_sources()
ct._apply_sources()

# write rhs for solving mass transport
def rhs_mass(t, c):
    
    # get volume
    V = net[mt.settings["pore_volume"]]
    # get A and b, assume BCs/sources already applied
    A = mt.A
    b = mt.b
    # calculate dcdt
    dcdt = (-A.dot(c) + b)/V
    
    return dcdt

# write rhs for solving charge transport
def rhs_charge(t, phi):
    
    # get volume
    V = net[ct.settings["pore_volume"]]
    # get capacitance, surface area
    C = phase["pore.capacitance"]
    a = phase["pore.surface_area"]
    # get A and b
    A = ct.A
    b = ct.b
    # calculate dphidt
    dphidt = (-A.dot(phi) + b)/C/a/V
    
    return dphidt

# write implicit F for solve_dae
def F(t, phi, phip):
    
    # get dphidt
    F = rhs_charge(t, phi)  # F = dphidt
    # get mask
    mask = net["pore.micropore"]
    F[mask] -= phip[mask]
    
    return F

# check mass balance to start
c_mi = phase["pore.micro_concentration"]
c = phase["pore.concentration"]
V = net["pore.volume_actual"]
c_mi[~net["pore.micropore"]] = 0
mt0 = 0.5*np.dot(c_mi, V) + np.dot(c, V)

# get initial condition of ALL properties to be solved
c = phase["pore.concentration"]
phi = phase["pore.potential"]
c_mi = phase["pore.micro_concentration"]
phi_d = phase["pore.donnan_potential"]

# store solution for first time!
y = np.concatenate((c, phi, c_mi, phi_d))
x = np.array([0])

# perform time stepping
t0 = 0
dt = dt
tf = 50
t_save = np.array([0, 0.05, 0.5, 5, 50])
for t in np.arange(t0+dt, tf+dt, dt):
    print(f"Time: {t}s")
    # define initial condition for this time step
    c0 = c.copy()
    print(c0)
    phi0 = phi.copy()
    phip0 = rhs_charge(t, phi0)
    # choose c_old and phi_old as initial condition
    # FIXME: first g_res is c - c0 NOT c - c_old
    c_old = c0.copy()
    phi_old = phi0.copy()
    # define gummel convergence
    g_res = np.array([100, 100])
    g_tol = np.array([1e-4, 1e-4])
    g_max_iter = 10
    for g_iter in range(g_max_iter):
        print(f"  Gummel Iteration No. {g_iter + 1}")
        # solve mass balance first
        sol_m = solve_ivp(rhs_mass, t_span=(0, dt), y0=c0, t_eval=[dt])
        c = sol_m.y[:, -1]
        print(c)
        # solve for potential second, use solve_dae
        sol_c = solve_dae(F, t_span=(0, dt), y0=phi0, yp0=phip0, t_eval=[dt])
        phi = sol_c.y[:, -1]
        # update concentration and potential
        phase["pore.concentration"] = c
        phase["pore.potential"] = phi
        # regenerate physics based on c and phi
        phase.regenerate_models()
        # update transport algs
        mt["pore.concentration"] = c
        ct["pore.potential"] = phi
        mt._update_A_and_b()
        ct._update_A_and_b()
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
    # store old c_mi and phi_d
    phase["pore.micro_concentration_old"] = phase["pore.micro_concentration"].copy()
    phase["pore.donnan_potential_old"] = phase["pore.donnan_potential"].copy()
    # regenerate models
    phase.regenerate_models()
    # store results if t is in tsave
    if np.isin(t, t_save):
        # get phi_d and c_mi
        c_mi = phase["pore.micro_concentration"]
        phi_d = phase["pore.donnan_potential"]
        # save results as y and x, assume that everything gets saved!
        y = np.vstack((y, np.concatenate((c, phi, c_mi, phi_d))))
        x = np.concatenate((x, np.array([t])))

# retrieve final properties
inds_ma = net.pores(["macropore", "separator"], mode="or")
inds_mi = net.pores("micropore")
cs = y[:, inds_ma]
phis = y[:, inds_ma+net.Np]
c_mis= y[:, inds_mi+2*net.Np]
phi_ds = y[:, inds_mi+3*net.Np]

# plot concentration
plt.figure(1)
plt.plot(net["pore.coords"][:, 0][inds_ma]*1e3, cs.T)
plt.title("Concentration (mol/m3)", fontsize=16)
plt.xlabel("Length (mm)", fontsize=14)
plt.legend(labels=[str(t) + "s" for t in t_save],
           ncol=len(t_save),
           loc="lower center",
           bbox_to_anchor=(0.5, -0.4),
           fontsize=11)

# plot potential
plt.figure(2)
plt.plot(net["pore.coords"][:, 0][inds_ma]*1e3, phis.T)
plt.title("Potential (V)", fontsize=16)
plt.xlabel("Length (mm)", fontsize=14)
plt.legend(labels=[str(t) + "s" for t in t_save],
           ncol=len(t_save),
           loc="lower center",
           bbox_to_anchor=(0.5, -0.4),
           fontsize=11)

# plot micropore concentration
plt.figure(3)
plt.plot(net["pore.coords"][:, 0][inds_mi]*1e3, c_mis.T)
plt.title("Micropore Concentration (mol/m3)", fontsize=16)
plt.xlabel("Length (mm)", fontsize=14)
plt.legend(labels=[str(t) + "s" for t in t_save],
           ncol=len(t_save),
           loc="lower center",
           bbox_to_anchor=(0.5, -0.4),
           fontsize=11)

# plot donnan potential
plt.figure(4)
plt.plot(net["pore.coords"][:, 0][inds_mi]*1e3, phi_ds.T)
plt.title("Donnan Potential (V)", fontsize=16)
plt.xlabel("Length (mm)", fontsize=14)
plt.legend(labels=[str(t) + "s" for t in t_save],
           ncol=len(t_save),
           loc="lower center",
           bbox_to_anchor=(0.5, -0.4),
           fontsize=11)

# check mass balance to start
c_mi = phase["pore.micro_concentration"]
c = phase["pore.concentration"]
V = net["pore.volume_actual"]
c_mi[~net["pore.micropore"]] = 0
mtf = 0.5*np.dot(c_mi, V) + np.dot(c, V)

# print mass balance error
# note that it can be improved by controlling time step
print(f"Percent mass balance error: {(mtf-mt0)/mt0*100:.3f}%")  
