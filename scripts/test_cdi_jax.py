import openpnm as op
import numpy as _np
import jax.numpy as jnp
from jax import config
import pnmlib as pnm
import collection.jax as co
from collections import ChainMap
import properties as prpts
import jax
from jax import lax
import pnmlib.models as mods
import models.jax as models
from pnmlib.network import create_adjacency_matrix, graph_laplacian


from jax import config
config.update("jax_disable_jit", False)

# FIXME: remove?
import sys
sys.path.append(r"D:\OneDrive\UW files\code\mypnmlib")

config.update("jax_enable_x64", True)

# create network
spacing = 1e-5
shape = [3, 2]
net = op.network.BodyCenteredCubic(shape=shape, spacing=spacing)

# shift body pores
coords = net["pore.coords"]
net["pore.coords"][net.pores("body"), 2] -= spacing/2

# delete body to body throats
throats = net["throat.body_to_body"]
op.topotools.trim(net, throats=throats)

# add pore labels
net["pore.macropore"] = net["pore.corner"]
net["pore.micropore"] = net["pore.body"]
net["pore.cathode"] = net["pore.coords"][:, 0] < spacing*shape[0]/2
net["pore.separator"] = net["pore.coords"][:, 0] == spacing*shape[0]/2
net["pore.anode"] = net["pore.coords"][:, 0] > spacing*shape[0]/2

# add throat labels
net["throat.macropore"] = net["throat.corner_to_corner"]
net["throat.micropore"] = net["throat.corner_to_body"]

# remove labels
del net["pore.corner"]
del net["pore.body"]
del net["throat.corner_to_body"]
del net["throat.corner_to_corner"]
del net["throat.body_to_body"]
del net["pore.surface"]

# convert network to jax
net = pnm.io.numpy_to_jax(net)

# get Np and Nt
Np = len(net["pore.coords"])
Nt = len(net["throat.conns"])

# add diameter
net["pore.diameter"] = jnp.ones(Np)*0.3*spacing
net["throat.diameter"] = 0.5 * jnp.min(net["pore.diameter"][net["throat.conns"]], axis=1)

# add zeta parameter
mask = net["throat.micropore"]
net["throat.zeta"] = jnp.where(mask, 1.0, 1.0)

# add geo model collections
pnm.models.apply_models(net,
                        models=co.geometry.macropore.spheres_and_cylinders,
                        domain="macropore")
pnm.models.apply_models(net,
                        models=co.geometry.macropore.spheres_and_cylinders,  # FIXME: micropore!
                        domain="micropore")

# get effective volume
net["pore.effective_volume"] = net["pore.volume"]

# get micropore volume
net["pore.micro_volume"] = net["pore.volume"]

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

# set concentration and pressure
net["pore.concentration"] = jnp.ones(Np) * cf
net["pore.pressure"] = jnp.ones(Nt)
net["pore.temperature"] = jnp.ones(Np) * T

# add phase models, I added interpolation to models
pnm.models.apply_models(net,
                        models=co.phase.macropore,
                        domain="macropore")
pnm.models.apply_models(net,
                        models=co.phase.macropore,  # FIXME: D for Micro!
                        domain="micropore")

# set "old" concentrations
net["pore.concentration_old"] = net["pore.concentration"].copy()
net["throat.concentration_old"] = net["throat.concentration"].copy()

# apply phase
pnm.models.apply_models(net,
                        models=co.physics.macropore,  # FIXME: gh model
                        domain="macropore")
pnm.models.apply_models(net,
                        models=co.physics.micropore,
                        domain="micropore")

# add properties to phase
phi = jnp.where(net["pore.anode"], V_cell/2, 0.0)
phi = jnp.where(net["pore.cathode"], -V_cell/2, phi)
net["pore.potential"] = phi
net["pore.electrode_potential"] = phi
net["pore.attraction_term"] = jnp.zeros(Np)  # jnp.where(net["pore.micropore"], mu_att, 0.0)
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
    
    # set BCs, set phi @ separator to zero helped stabilize gmres
    pores=net["pore.separator"]
    pnm.algorithms.set_BC(proj, "alg3", pores, bctype='value',
                          bcvalues=0, mode="overwrite")

    # apply sources
    ct["A"], ct["b"] = pnm.algorithms.apply_sources(proj, alg="alg3")
    
    # apply BCs
    ct["A"], ct["b"] = pnm.algorithms.apply_BC(proj, alg="alg3")
    
    return ct["A"], ct["b"]


def fun(network, u):
    
    # get coords and conns
    coords = network["pore.coords"]
    conns = network["throat.conns"]

    # split up u
    Np = len(coords)
    c = u[:Np]
    phi = u[Np:]
    
    # update ionic conductivity (pore and throat)
    conductivity = models.electrical_double_layer.conductivity
    Kp = conductivity(c=c,
                      D=network["pore.diffusivity"],
                      T=network["pore.temperature"])
    Kt = conductivity(c=jnp.average(c[conns], axis=1),
                      D=network["throat.diffusivity"],
                      T=network["throat.temperature"])

    # update ionic conductance
    poisson_conductance = models.electrical_double_layer.poisson_conductance
    K = poisson_conductance(network, Kp, Kt,
                            sf=network["throat.diffusive_size_factors"])
    network["throat.ionic_conductance"] = K
    
    # update donnan potential
    donnan_potential = models.electrical_double_layer.donnan_potential
    phi_d0 = network["pore.donnan_potential_old"]
    phi_d = donnan_potential(phi_d0, phi, c,
                             phi_e=network["pore.electrode_potential"],
                             mu_att=network["pore.attraction_term"],
                             T=network["pore.temperature"],
                             C=network["pore.capacitance"],
                             a=network["pore.surface_area"])
    network["pore.donnan_potential"] = phi_d

    # update micropore concentration
    mi_concentration = models.electrical_double_layer.micropore_concentration
    c_mi = mi_concentration(c, phi_d,
                            T=network["pore.temperature"],
                            mu_att=network["pore.attraction_term"])

    # update mass source
    mass_source = models.electrical_double_layer.mass_source
    Rm = mass_source(c_mi,
                     c_mi_i=network["pore.micro_concentration_old"],
                     V_mi=network["pore.micro_volume"],
                     dt=network["time_step"])
    network["pore.mass_source_effective"] = Rm

    # update charge source
    charge_source = models.electrical_double_layer.charge_source
    Rc = charge_source(phi_d,
                       phi_d_i=network["pore.donnan_potential_old"],
                       C=network["pore.capacitance"],
                       a=network["pore.attraction_term"],
                       V=network["pore.volume"],
                       dt=network["time_step"])
    network["pore.charge_source"] = Rc

    # re-build A and b
    Am, bm = _get_mass_A_and_b(network)
    Ac, bc = _get_charge_A_and_b(network)
    
    # calculate flux!
    flux_mass = (-Am @ c + bm)
    flux_charge = (-Ac @ phi + bc)
    
    # get flux
    flux = jnp.concatenate((flux_mass, flux_charge))
    
    return flux


def residual(network, u, u_prev, dt):
    
    # get Np
    Np = len(network["pore.coords"])
    
    # get volume
    C = network["pore.capacitance"]
    a = network["pore.surface_area"]
    V = network["pore.effective_volume"]
    CaV = jnp.where(network["pore.micropore"], C*a*V, 0.0)
    
    # get dudt
    flux = fun(network, u)
    
    # split up dudt
    flux_mass = flux[:Np]
    flux_charge = flux[Np:]
    
    # split up u
    c = u[:Np]
    phi = u[Np:]
    
    # split up u_prev
    c_prev = u_prev[:Np]
    phi_prev = u_prev[Np:]
    
    F_mass = V*(c - c_prev)/dt - flux_mass
    F_charge = CaV*(phi - phi_prev)/dt - flux_charge
    
    F = jnp.concatenate((F_mass, F_charge))
    
    return F



def implicit_solve(network,
                   residual,
                   t_span,
                   u0,
                   dt,
                   iters=20,
                   tol=1e-6,
                   newton_maxiter=50,
                   newton_tol=1e-8,
                   alpha=0.9, 
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
            # print(u)
            F = residual(network, u, u_prev, dt)
            print(jnp.linalg.norm(F))
            # do one newton step
            delta = newton_step(u, u_prev, dt,
                                newton_tol, newton_maxiter)
            # print(delta)

            # update
            u_old = u.copy()
            u = u_old + alpha*delta
            
            # line search for alpha
            c = u[:Np]  # this is the problem
            alpha_ls = alpha
            while jnp.any(c < 0):
                # get new alpha
                alpha_ls *= 0.8  # FIXME: make 0.5 argument
                # update c
                u = u_old + alpha_ls*delta
                c = u[:Np]
            
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
    
    Np = len(network["pore.coords"])
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
    # get Np
    Np = len(network["pore.coords"])

    # get t0 and tf
    t0 = t_span[0]
    tf = t_span[1]

    # initialize t0 and tf
    ts = jnp.arange(t0, tf+dt, dt)
    us = jnp.zeros((len(ts), 2*Np))
    us = us.at[0].set(u0)

    # perform time stepping
    us = lax.fori_loop(1, len(ts), time_step, us)
    '''
    
    return ts, us


# try running implicit solve
c0 = net["pore.concentration"]
phi0 = net["pore.potential"]
u0 = jnp.concatenate((c0, phi0))

# calculate initial mass
c_mi = net["pore.micro_concentration"][net["pore.micropore"]]
V_mi = net["pore.micro_volume"][net["pore.micropore"]]
c = net["pore.concentration"]
V = net["pore.volume"]
m0 = jnp.dot(c, V) + jnp.dot(c_mi/2, V_mi)

# time stepping
t0 = 0
dt = dt
tf = dt
# initialize c and phi
c = net["pore.concentration"]
phi = net["pore.potential"]
for t in jnp.arange(t0+dt, tf+dt, dt):
    print(f"Time: {t}s")
    # get initial conditions
    c0 = c.copy()
    phi0 = phi.copy()
    # combine initial coniditons
    u0 = jnp.concatenate((c0, phi0))
    # do an implicit solve
    ts, us = implicit_solve(net, residual, (0, dt), u0, dt)
    # get out c and phi
    c = us[-1, :Np]
    phi = us[-1, Np:]
    # update donnan potential old
    donnan_potential = models.electrical_double_layer.donnan_potential
    phi_d0 = net["pore.donnan_potential_old"]
    phi_d = donnan_potential(phi_d0, phi, c,
                             phi_e=net["pore.electrode_potential"],
                             mu_att=net["pore.attraction_term"],
                             T=net["pore.temperature"],
                             C=net["pore.capacitance"],
                             a=net["pore.surface_area"])
    net["pore.donnan_potential_old"] = phi_d.copy()
    # update micro concentration old
    mi_concentration = models.electrical_double_layer.micropore_concentration
    c_mi = mi_concentration(c, phi_d,
                            T=net["pore.temperature"],
                            mu_att=net["pore.attraction_term"])
    net["pore.micro_concentration_old"] = c_mi.copy()
    # update concentration old
    # net["pore.concentration_old"] = c.copy()
    # conns = net["throat.conns"]
    # net["throat.concentration_old"] = jnp.average(c[conns], axis=1) 
    # net["pore.concentration"] = c
    # net["pore.potential"] = phi

# calculate final mass
c_mi = c_mi[net["pore.micropore"]]
mf = jnp.dot(c, V) + jnp.dot(c_mi/2, V_mi)

# calculate mass balance error
print(f"Mass Balance Error: {abs(mf-m0)/m0*100}%")  # 0.00010472375798286593%
print(jnp.average(c))  # 0.681904403267796
print(jnp.average(abs(phi)))  # 0.020685176432581213

'''
def time_step(i, us):
    
    def implicit_step(state):
        
        # get u and k from state
        u, k, _ = state
        
        # do one newton step
        delta = newton_step(u, u_prev, dt, newton_tol, newton_maxiter)
        
        # update u
        u += alpha * delta
        
        # set c < 0 to zero
        c = u[:Np]
        phi = u[Np:]
        c = jnp.where(c < 0, 1e-5, c)  # avoids negative c
        u = jnp.concatenate((c, phi))
        
        # calculate error
        normF = jnp.linalg.norm(alpha * delta)

        return (u, k + 1, normF)

    def cond(state):
        _, k, normF = state
        return jnp.logical_and(normF > tol, k < iters)
    
    # get u_prev
    u_prev = us[i-1, :]

    # initial state
    normF0 = 1e3
    init_state = (u_prev, 0, normF0)

    # implicit_step
    u, _, _ = lax.while_loop(cond, implicit_step, init_state)
    
    return us.at[i].set(u)
'''