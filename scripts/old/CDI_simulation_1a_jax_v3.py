"""

Explicit A and b. Updates A and b on the fly.

Wrote a lot of jittable functions. While I used this script to prepare
the functions saved in electrical_double_layer.py it is not in a working state

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

'''
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

# create proj
proj = {}
proj["network"] = net

# mass transport algorithm
mt = {}
proj["alg2"] = mt

# write keys to mt
mt["conductance"] = "throat.mass_conductance"
mt["hydraulic_conductance"] = "throat.hydraulic_conductance"
mt["quantity"] = "pore.concentration"
mt["pressure"] = "pore.pressure"
mt["pore_volume"] = "pore.effective_volume"

# set source
mt["sources"] = ["mass_source_effective"]
pnm.algorithms.set_source(proj, alg="alg2",
                          pores=net["pore.micropore"],
                          propname="mass_source_effective")

# build A and b
mt["A"] = pnm.algorithms.build_A(proj, "alg2")
mt["b"] = pnm.algorithms.build_b(proj, "alg2")

# apply sources
mt["A"], mt["b"] = pnm.algorithms.apply_sources(proj, alg="alg2")

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


def conductivity(c, D, T):
    
    R = 8.31446262  # J/mol/K
    F = 96485.3399  # C/mol
    
    K = 2*F**2*c*D/R/T
    
    return K


def _poisson_conductance(network, Kp, Kt, sf):
    
    # throat conns
    cn = network["throat.conns"]
    # get conducitivities
    Dt = Kt
    D1, D2 = Kp[cn].T
    # If individual size factors for conduit constiuents are known
    SF = sf
    # calcualte conductance
    F1, Ft, F2 = SF.T
    g1 = D1 * F1
    gt = Dt * Ft
    g2 = D2 * F2
    
    G = 1 / (1 / g1 + 1 / gt + 1 / g2)

    return jnp.vstack((G, G)).T 


def donnan_potential(phi_d0,
                     phi, 
                     c,
                     phi_e,
                     mu_att,
                     T,
                     C,
                     a):
    
    # set constants
    R = 8.31446262  # J/mol/K
    F = 96485.3399  # C/mol
    
    def micropore_charge_concentration(phi_d):
        
        sigma = -2 * c * jnp.exp(mu_att) * jnp.sinh(F/R/T * phi_d)
        
        return sigma
    
    def stern_potential(sigma):
        
        phi_st = -sigma * F / C / a
        
        return phi_st
    
    def potential_balance(phi_d):
        
        sigma = micropore_charge_concentration(phi_d)
        phi_st = stern_potential(sigma)
        f = phi_e - phi - phi_d - phi_st
        
        return f
    
    # use newton_krylov to get donnan potential
    phi_d = pnm.optimize.newton_krylov_scan(potential_balance, phi_d0, tol=6e-6)
    
    return phi_d


def micropore_concentration(c, phi_d, T, mu_att):
    
    # set constants
    R = 8.31446262  # J/mol/K
    F = 96485.3399  # C/mol
    
    c_mi = 2 * c * jnp.exp(mu_att) * jnp.cosh(F/R/T*phi_d)
    
    return c_mi


def mass_source(c_mi_f, c_mi_i, V_mi, dt):
    
    S1 = 0
    S2 = - 1/2 * V_mi * (c_mi_f - c_mi_i) / dt
    rate = S2
    values = {"S1": S1, "S2": S2, "rate": rate}
    
    return values


def mass_source_effective(R, c, V, dt):
    
    # Maximum removable mass rate (positive number)
    R_max = c * V / dt
    
    # Start with original rate
    R_eff = R.copy()
    
    # Identify sinks (negative rates)
    sink = R < 0
    
    # Limit sink magnitude
    R_eff = jnp.where(sink, jnp.maximum(R, -R_max), R_eff)
    
    S1 = 0
    S2 = R_eff
    rate = S2
    values = {"S1": S1, "S2": S2, "rate": rate}
    
    return values


def charge_source(phi_d_f, phi_d_i, C, a, V, dt):
    
    S1 = 0
    S2 = - C * a * V * (phi_d_f - phi_d_i) / dt
    rate = S2
    values = {"S1": S1, "S2": S2, "rate": rate}
    
    return values
    
    
def effective_micropore_concentration(Rm_eff, c_mi_i, V_mi, dt):
    
    c_mi_f = -2*dt/V_mi*Rm_eff + c_mi_i
    
    return c_mi_f


def fun(network, u):
    
    # get Np
    Np = len(network["pore.coords"])
    # get conns
    conns = network["throat.conns"]
    # split up u
    c = u[:Np]
    phi = u[Np:]
    # update ionic conductivity (pore and throat)
    Kp = conductivity(c=c,
                      D=network["pore.diffusivity"],
                      T=network["pore.temperature"])
    Kt = conductivity(c=jnp.average(c[conns], axis=1),  # throat c is avg
                      D=network["throat.diffusivity"],
                      T=network["throat.temperature"])
    # update ionic conductance
    sf = network["throat.diffusive_size_factors"]
    K = _poisson_conductance(network, Kp, Kt, sf)
    # update donnan potential
    phi_d0 = network["pore.donnan_potential_old"]  # from previous time step
    phi_d = donnan_potential(phi_d0, phi, c,
                             phi_e=network["pore.electrode_potential"],
                             mu_att=network["pore.attraction_term"],
                             T=network["pore.temperature"],
                             C=network["pore.capacitance"],
                             a=network["pore.surface_area"])
    network["pore.donnan_potential"] = phi_d
    # update micropore concentration
    c_mi = micropore_concentration(c, phi_d,
                                   T=network["pore.temperature"],
                                   mu_att=network["pore.attraction_term"])
    # update mass source
    Rm = mass_source(c_mi,
                     c_mi_i=network["pore.micro_concentration_old"],
                     V_mi=network["pore.micro_volume"],
                     dt=network["time_step"])
    Rm_eff = mass_source_effective(R=Rm["rate"],
                                   c=c,
                                   V=network["pore.volume"],
                                   dt=network["time_step"])
    network["pore.mass_source_effective"] = Rm  # FIXMe: Rm_eff
    # update charge source
    Rc = charge_source(phi_d,
                       phi_d_i=network["pore.donnan_potential_old"],
                       C=network["pore.capacitance"],
                       a=network["pore.attraction_term"],
                       V=network["pore.volume"],
                       dt=network["time_step"])
    network["pore.charge_source"] = Rc
    # build mass A and b
    # Am, bm = _update_A_and_b()
    # build charge A and b
    # Ac, bc = build_charge_A_and_b()
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
    # set source
    mt["sources"] = ["mass_source_effective"]
    pnm.algorithms.set_source(proj, alg="alg2",
                              pores=net["pore.micropore"],
                              propname="mass_source_effective")
    # build A and b
    mt["A"] = pnm.algorithms.build_A(proj, "alg2")
    mt["b"] = pnm.algorithms.build_b(proj, "alg2")
    # apply sources
    mt["A"], mt["b"] = pnm.algorithms.apply_sources(proj, alg="alg2")
    # update A and b
    # pnm.algorithms.update_A_and_b(proj, "alg2")
    # pnm.algorithms.update_A_and_b(proj, "alg3")
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
    # get dudt
    dcdt = (-mt["A"] @ c + mt["b"])
    dphidt = (-ct["A"] @ phi + ct["b"])
    dudt = jnp.concatenate((dcdt, dphidt))
    
    return dudt

'''
fun = jax.jit(fun)

c = net["pore.concentration"]
phi = net["pore.potential"]
u = jnp.concatenate((c, phi))
dudt = fun(net, u)
'''

def implicit_solve(proj,
                   fun,
                   t_span,
                   u0,
                   dt,
                   method="euler",
                   iters=10,
                   tol=1e-18,
                   newton_maxiter=50,
                   newton_tol=1e-18,
                   alpha=0.9, 
                   check_convergence=False):
    
    
    def residual(u, u_prev, dt):
        
        V = proj["solver_volume"]
        
        if method == "euler":
            F = V * u - V * u_prev - dt * fun(proj, u)
        
        elif method == "crank-nicolson":
            F = V * u - V * u_prev - dt / 2 * (fun(proj, u) + fun(proj, u_prev))  # crank-nicolson won't work, because @ u_prev sources are zero
            
        else:
            raise ValueError("method must be either euler or crank-nicolson")
        
        return F
        
    
    def Jv(delta, u, u_prev, dt):
        F = lambda u: residual(u, u_prev, dt)
        return jax.jvp(F, (u,), (delta,))[1]
        
    
    def linear_map(u, u_prev, dt):
        def matvec(v):
            return Jv(v, u, u_prev, dt)
        return matvec
    
    
    def newton_step(u, u_prev, dt, tol=1e-6, maxiter=50):
        
        # calculate residual
        F = residual(u, u_prev, dt)
        # get matvec product
        matvec = linear_map(u, u_prev, dt)
        # Solve J δ = -F using cg
        delta, info = jax.scipy.sparse.linalg.gmres(
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
            F = residual(u, u_prev, dt)
            print(F)
            # do one newton step
            delta = newton_step(u, u_prev, dt,
                                newton_tol, newton_maxiter)
            print(delta)
            # update u for next step
            u = u + alpha*delta  # damping is important for rxn systems

            # set c < 0 to zero
            c = u[:8]
            phi = u[8:]
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

    '''
    def time_step(i, us):
        
        def implicit_step(state):
            
            # get u and k from state
            u, k, _ = state
            # do one newton step
            delta = newton_step(u, u_prev, dt, newton_tol, newton_maxiter)

            u += alpha * delta
            F = residual(u, u_prev, dt)
            normF = jnp.linalg.norm(F)

            return (u, k + 1, normF)

        def cond(state):
            _, k, normF = state
            return jnp.logical_and(normF > tol,
                                   k < iters)
        
        # get u_prev
        u_prev = us[i-1, :]

        # initial state
        F0 = residual(u_prev, u_prev, dt)
        normF0 = jnp.linalg.norm(F0)
        init_state = (u_prev, 0, normF0)

        # implicit_step
        x_final, _, _ = lax.while_loop(cond, implicit_step, init_state)
        
        return us.at[i].set(u)
    
    
    # get network
    network = proj["network"]
    Np = len(network["pore.coords"])
    
    # get t0 and tf
    t0 = t_span[0]
    tf = t_span[1]
    
    # initialize t0 and tf
    ts = jnp.arange(t0, tf+dt, dt)
    us = jnp.zeros((len(ts), 2*Np))
    us = us.at[0].set(u0)

    # perform time stepping
    lax.fori_loop(1, len(ts)-1, time_step, us)
    '''
    
    return ts, us