"""

NOT WORKABLE

"""
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

    # apply sources (there are no BCs)
    ct["A"], ct["b"] = pnm.algorithms.apply_sources(proj, alg="alg3")
    
    return ct["A"], ct["b"]


def fun(network, u):

    # get Np
    Np = len(network["pore.coords"])
    
    # split up u
    c = u[:Np]
    phi = u[Np:]
    
    # update ionic conductivity (pore and throat)
    conductivity = models.electrical_double_layer.conductivity
    Kp = conductivity(c=network["pore.concentration_old"],
                      D=network["pore.diffusivity"],
                      T=network["pore.temperature"])
    Kt = conductivity(c=network["throat.concentration_old"],
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
    
    # dcdt and dphidt, note V is missing!
    dcdt = (-Am @ c + bm)
    dphidt = (-Ac @ phi + bc)
    
    # get dudt
    dudt = jnp.concatenate((dcdt, dphidt))
    
    return dudt


def residual(network, u, u_prev, dt):
    
    # get Np
    Np = len(network["pore.coords"])
    
    # get volume
    C = network["pore.capacitance"]
    a = network["pore.surface_area"]
    V = network["pore.effective_volume"]
    CaV = jnp.where(network["pore.micropore"], C*a*V, 0.0)
    
    # get dudt
    dudt = fun(network, u)
    
    # split up dudt
    dcdt = dudt[:Np]
    dphidt = dudt[Np:]
    
    # split up u
    c = u[:Np]
    phi = u[Np:]
    
    # split up u_prev
    c_prev = u_prev[:Np]
    phi_prev = u_prev[Np:]
    
    F_mass = V*(c - c_prev)/dt - dcdt
    F_charge = CaV*(phi - phi_prev)/dt - dphidt
    
    F = jnp.concatenate((F_mass, F_charge))
    
    return F