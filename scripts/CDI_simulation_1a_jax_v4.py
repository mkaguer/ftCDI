"""

Segregated Solves!

"""
# FIXME: should be able to remove this!
import sys
sys.path.append(r"D:\OneDrive\UW files\code\mypnmlib")

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
import matplotlib.pyplot as plt
import pnmlib.models as mods
import models.jax as models

config.update("jax_enable_x64", True)

# create blank project dict
proj = {}

# import network (full cell with labels)
data = _np.load('../networks/create_full_cell_1a_test.npz')
data = {key: _np.array(data[key]) for key in data.files}

# convert to jax
net = pnm.io.numpy_to_jax(data)

# add network to project (for organization)
proj["network"] = net

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

# set concentration and pressure
net["pore.concentration"] = jnp.ones(Np) * cf
net["pore.pressure"] = jnp.ones(Nt)
net["pore.temperature"] = jnp.ones(Np) * T

# add phase models, I added interpolation to models
pnm.models.apply_models(net,
                        models=co.phase.macropore,
                        domain="macropore")
pnm.models.apply_models(net,
                        models=co.phase.micropore,
                        domain="micropore")
pnm.models.apply_models(net,
                        models=co.phase.separator,
                        domain="separator")
pnm.models.apply_models(net,
                        models=co.phase.macropore,
                        domain="perforated")

# set "old" concentrations
net["pore.concentration_old"] = net["pore.concentration"].copy()
net["throat.concentration_old"] = net["throat.concentration"].copy()

# add physics models to phase
pnm.models.apply_models(net,
                        models=co.physics.perforated,
                        domain="perforated")
pnm.models.apply_models(net,
                        models=co.physics.macropore,
                        domain="macropore")
pnm.models.apply_models(net,
                        models=co.physics.micropore,
                        domain="micropore")
pnm.models.apply_models(net,
                        models=co.physics.separator,
                        domain="separator")

# add properties to phase, these get from imported properties dict!
# set potential
phi = jnp.where(net["pore.anode"], V_cell/2, 0.0)
phi = jnp.where(net["pore.cathode"], -V_cell/2, phi)
net["pore.potential"] = phi
net["pore.electrode_potential"] = phi
net["pore.attraction_term"] = jnp.where(net["pore.micropore"], mu_att, 0.0)
net["pore.capacitance"] = jnp.where(net["pore.micropore"], Ca, 0.0)
net["pore.surface_area"] = jnp.ones(Np)*1

# select time step
dt = 0.1
net["time_step"] = dt

# set iniital guess for donnan potential
net["pore.donnan_potential"] = jnp.zeros(Np)

# add phi_d and c_mi properties
pnm.models.apply_models(net,
                        models=co.physics.properties,
                        domain="micropore")

# set old c_mi and phi_d
net["pore.micro_concentration_old"] = net["pore.micro_concentration"].copy()
net["pore.donnan_potential_old"] = net["pore.donnan_potential"].copy()

# add source term models
pnm.models.apply_models(net,
                        models=co.physics.sources,
                        domain="micropore")

# stokes flow algorithm
sf = {}
proj["alg1"] = sf

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
net["pore.pressure"] = x
co.regenerate_models(proj, net)

