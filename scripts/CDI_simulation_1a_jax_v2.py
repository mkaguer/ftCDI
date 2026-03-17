"""

Explicit A and b. Updates A and b on the fly.

Coupling did not work because mass and charge residuals have different
magnitudes. Also, this script is not in working order since we updated
models.electrical_double_layer to be jittable!

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
net["pore.capacitance"] = jnp.where(net["pore.micropore"], Ca, 1.0)
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
'''
# get initial potential
ic = {}
proj["alg4"] = ic

# write keys to sf
ic["quantity"] = "pore.potential"
ic["conductance"] = "throat.ionic_conductance"

# set BCs
anode = net["pore.anode"] & net["pore.micropore"]
cathode = net["pore.cathode"] & net["pore.micropore"]
pnm.algorithms.set_BC(proj, "alg4", anode, bctype='value',
                      bcvalues=V_cell/2, mode="overwrite")
pnm.algorithms.set_BC(proj, "alg4", cathode, bctype='value',
                      bcvalues=-V_cell/2, mode="add")

# build A and b
ic["A"] = pnm.algorithms.build_A(proj, "alg4")
ic["b"] = pnm.algorithms.build_b(proj, "alg4")

# apply BCs
ic["A"], ic["b"] = pnm.algorithms.apply_BC(proj, alg="alg4")


# solve
phi0 = pnm.algorithms.solve(proj, "alg4")
net["pore.potential"] = phi0
'''

def _get_mass_A_and_b(network):
    
    # create proj
    proj = {}
    proj["network"] = network
    
    # mass transport algorithm
    mt = {}
    proj["alg2"] = mt

    # write keys to mt
    mt["conductance"] = "throat.mass_conductance"
    mt["hydraulic_conductance"] = "throat.hydraulic_conductance"
    mt["quantity"] = "pore.concentration"
    mt["pressure"] = "pore.pressure"
    mt["pore_volume"] = "pore.effective_volume"

    # set value BC
    inlet = network["pore.inlet"]
    pnm.algorithms.set_BC(proj, "alg2", inlet, bctype="value",
                          bcvalues=cf, mode="overwrite")
    
    # set outflow BC
    outlet = network["pore.outlet"]
    throats = network["throat.outlet"]
    pnm.algorithms.set_outflow_BC(proj, "alg2", pores=outlet, throats=throats, mode="overwrite")
    
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
    
    return mt["A"], mt["b"]


def _get_charge_A_and_b(network):
    
    # create proj
    proj = {}
    proj["network"] = network
    
    # charge transport algorithm
    ct = {}
    proj["alg3"] = ct

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
    
    return ct["A"], ct["b"]


def fun(network, u):
    
    # split up u
    Np = len(network["pore.coords"])
    c = u[:Np]
    phi = u[Np:]
    # update on network
    network["pore.concentration"] = c
    network["pore.potential"] = phi
    # get throat concentration
    conns = network["throat.conns"]
    network["throat.concentration"] = jnp.average(c[conns], axis=1)
    # update ionic conductance
    Kp = mods.misc.conductivity(network,
                                concentration="pore.concentration",
                                diffusivity="pore.diffusivity",
                                temperature="pore.temperature")
    network["pore.conductivity"] = Kp
    Kt = mods.misc.conductivity(network,
                                concentration="throat.concentration",
                                diffusivity="throat.diffusivity",
                                temperature="throat.temperature")
    network["throat.conductivity"] = Kt
    K = mods.physics.diffusive_conductance.generic_diffusive(network,
                                pore_diffusivity="pore.conductivity",
                                throat_diffusivity="throat.diffusivity",
                                size_factors='throat.diffusive_size_factors')
    network["throat.ionic_conductance"] = K
    # update donnan potential
    phi_d = models.electrical_double_layer.donnan_potential(network,
                                pore_potential="pore.potential",
                                pore_electrode_potential="pore.electrode_potential",
                                pore_attraction_term="pore.attraction_term",
                                pore_temperature="pore.temperature",
                                pore_capacitance="pore.capacitance",
                                pore_surface_area="pore.surface_area",
                                pore_concentration="pore.concentration")
    network["pore.donnan_potential"] = phi_d
    # update micropore concentration
    c_mi = models.electrical_double_layer.micropore_concentration(network,
                                pore_concentration="pore.concentration",
                                pore_donnan_potential="pore.donnan_potential",
                                pore_temperature="pore.temperature",
                                pore_attraction_term="pore.attraction_term")
    network["pore.micro_concentration"] = c_mi
    # update mass source
    Rm = models.electrical_double_layer.mass_source(network,
                    pore_micro_concentration="pore.micro_concentration",
                    pore_micro_concentration_old="pore.micro_concentration_old",
                    pore_micro_volume="pore.micro_volume",
                    time_step="time_step")
    network["pore.mass_source_effective"] = Rm 
    # update charge source
    Rc = models.electrical_double_layer.charge_source(network,
                    pore_donnan_potential="pore.donnan_potential",
                    pore_donnan_potential_old="pore.donnan_potential_old",
                    pore_volume="pore.effective_volume",
                    pore_capacitance="pore.capacitance",
                    pore_surface_area="pore.surface_area",
                    time_step="time_step")
    network["pore.charge_source"] = Rc 
    # re-build A and b
    Am, bm = _get_mass_A_and_b(network)
    Ac, bc = _get_charge_A_and_b(network)
    
    # dcdt and dphidt, note V is missing!
    dcdt = (-Am @ c + bm)
    dphidt = (-Ac @ phi + bc)
    
    # get dudt
    dudt = jnp.concatenate((dcdt, dphidt))
    
    return dudt


def residual(network, u, u_prev, dt):
    
    # get volume
    C = network["pore.capacitance"]
    a = network["pore.surface_area"]
    V = network["pore.effective_volume"]
    # V = jnp.concatenate((V, C*a*V))
    CaV = jnp.where(network["pore.micropore"], C*a*V, 0.0)
    
    # calculate residual using backward euler
    # F = V*(u - u_prev)/dt - fun(network, u)
    
    Np = len(network["pore.coords"])
    
    dudt = fun(network, u)
    
    dcdt = dudt[:Np]
    dphidt = dudt[Np:]
    
    c = u[:Np]
    phi = u[Np:]
    
    c_prev = u_prev[:Np]
    phi_prev = u_prev[Np:]
    
    F_mass = V*(c - c_prev)/dt - dcdt
    F_charge = CaV*(phi - phi_prev)/dt - dphidt
    
    F = jnp.concatenate((F_mass, F_charge))
    
    return F


def implicit_solve(network,
                   residual,
                   u0,
                   t_span,
                   dt,
                   iters=10,
                   tol=1e-18,
                   newton_maxiter=50,
                   newton_tol=1e-18,
                   alpha=0.2, 
                   check_convergence=False):
        
    
    def Jv(delta, u, u_prev, dt):
        F = lambda u: residual(network, u, u_prev, dt)
        return jax.jvp(F, (u,), (delta,))[1]
        
    
    def linear_map(u, u_prev, dt):
        def matvec(v):
            return Jv(v, u, u_prev, dt)
        return matvec
    
    
    def newton_step(u, u_prev, dt, tol=1e-6, maxiter=50):
        
        # calculate residual
        F = residual(network, u, u_prev, dt)
        # get matvec product
        matvec = linear_map(u, u_prev, dt)
        # Solve J δ = -F using cg
        delta, info = jax.scipy.sparse.linalg.cg(
            matvec,
            -F,
            tol=tol,
            maxiter=maxiter
        )
        
        return delta


    def implicit_step(u_prev,
                      dt,
                      iters=5,
                      tol=1e-14,
                      newton_maxiter=50,
                      newton_tol=1e-6):
        
        # guess u
        u = u_prev

        for i in range(iters):
            print(i)
            print(u)
            F = residual(network, u, u_prev, dt)
            print(F)
            c = u[:1209]
            print(jnp.sum(c < 0))
            # do one newton step
            delta = newton_step(u, u_prev, dt,
                                newton_tol, newton_maxiter)
            print(delta)
            # update u for next step
            u = u + alpha*delta  # damping is important for rxn systems

            # set c < 0 to zero
            c = u[:1209]
            phi = u[1209:]
            c = jnp.where(c < 0, 1e-5, c)  # avoids negative c
            u = jnp.concatenate((c, phi))
            
            # set to True if not jitted
            if check_convergence:
                # FIXME: should we calculate residual before or after?
                # calculate residual
                F = residual(u, u_prev, dt)
                normF = jnp.linalg.norm(F)
                # print(normF)
                # break if normF < tolerance 
                if normF < tol:
                    print(normF)
                    print(f"converged at residual: {F}")
                    break
                
                # raise error if not converged
                if i == iters - 1:
                    print(normF)
                    raise ValueError("Newton did not converge. Max steps reached!")
            
        return u
    
    # FIXME: don't use python loop, although if small OKAY
    # perform time stepping
    t0 = t_span[0]
    tf = t_span[1]
    # get u_prev from initial condition
    u_prev = u0
    # initialize solution
    us = jnp.array([u_prev])
    ts = jnp.array([t0])
    for t in jnp.arange(t0+dt, tf+dt, dt):
        # do full implicit solve
        u = implicit_step(u_prev, dt,
                          iters, tol,
                          newton_maxiter, newton_tol)
        # get new u_prev
        u_prev = u
        # FIXME: make t_eval argument!
        # save u at every time step
        us = jnp.concatenate((us, jnp.array([u_prev])))
        ts = jnp.concatenate((ts, jnp.array([t])))

    return ts, us
    

pores = jnp.arange(Np)[net["pore.outlet"]]
net["throat.outlet"] = mods.misc.find_neighbor_throats(net, pores=pores)


c0 = net["pore.concentration"]
phi0 = net["pore.potential"]
u0 = jnp.concatenate((c0, phi0))

# res = jax.jit(Residual)
# grad_res = jax.grad(Residual, argnums=2)
res = residual(net, u0, u0, dt)
# grad_res(net, u*0.95, u, dt)

# res = residual(net, u=u0, u_prev=u0, dt=dt)

ts, us = implicit_solve(net,
                        residual,
                        u0=u0,
                        t_span=(0, dt),
                        dt=dt)

c = us[-1, :Np]
phi = us[-1, Np:]

