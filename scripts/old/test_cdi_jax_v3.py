"""

Segregated solves! JITTED!

Got it working on small 10 by 10 network! YAY!

"""
import sys
sys.path.append(r"D:\OneDrive\UW files\code\mypnmlib")

import openpnm as op
import jax.numpy as jnp
from jax import config
import pnmlib as pnm
import collection.jax as co
import properties as prpts
import jax
from jax import lax
import models.jax as models


config.update("jax_disable_jit", False)
config.update("jax_enable_x64", True)

# create network
spacing = 1e-5
shape = [10, 10]
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
mu_att = 1.0  # prpts.properties["mu_att"]
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


def _update_mass_source(network, c, phi):
    
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
    
    # calculate micropore concentration
    model = models.electrical_double_layer.micropore_concentration
    c_mi = model(c,
                 phi_d=network["pore.donnan_potential"],
                 T=network["pore.temperature"],
                 mu_att=network["pore.attraction_term"])
    network["pore.micro_concentration"] = c_mi

    # update mass source
    mass_source = models.electrical_double_layer.mass_source
    Rm = mass_source(c_mi,
                     c_mi_i=network["pore.micro_concentration_old"],
                     V_mi=network["pore.micro_volume"],
                     dt=network["time_step"])
    network["pore.mass_source"] = Rm
    
    
    # update mass source effective
    mass_source_eff = models.electrical_double_layer.mass_source_effective
    Rm_e = mass_source_eff(Rm,
                           c=network["pore.concentration_old"],
                           V=network["pore.volume"],  # FIXME: effective?
                           dt=network["time_step"])
    
    # update mass source term
    network["pore.mass_source_effective"] = Rm_e


def _update_charge_source(network, c, phi):

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

    # update charge source
    charge_source = models.electrical_double_layer.charge_source
    Rc = charge_source(phi_d,
                       phi_d_i=network["pore.donnan_potential_old"],
                       C=network["pore.capacitance"],
                       a=network["pore.surface_area"],
                       V=network["pore.effective_volume"],
                       dt=network["time_step"])

    # update charge source term
    network["pore.charge_source"] = Rc


def _update_charge_conductance(network, c):

    # get conns
    conns = network["throat.conns"]

    # update ionic conductivity (pore and throat)
    conductivity = models.electrical_double_layer.conductivity
    Kp = conductivity(c=c,
                      D=network["pore.diffusivity"],
                      T=network["pore.temperature"])
    Kt = conductivity(c=jnp.average(c[conns], axis=1),
                      D=network["throat.diffusivity"],
                      T=network["throat.temperature"])

    # calculate ionic conductance
    poisson_conductance = models.electrical_double_layer.poisson_conductance
    K = poisson_conductance(network, Kp, Kt,
                            sf=network["throat.diffusive_size_factors"])

    # update ionic conductance
    network["throat.ionic_conductance"] = K
    


def linear_map(A, V, dt):
    def matvec(x):
        return A @ x + V/dt * x
    return matvec


def linear_map2(A, V, dt):
    def matvec(x):
        return A @ x / V + 1/dt * x
    return matvec


def residual(A, b, x, x_prev, V, dt):
    
    # backward euler
    F = V*(x - x_prev)/dt - (-A @ x + b)
    
    return F


# get volume
V = net["pore.effective_volume"]

# get capacitance
C = net["pore.capacitance"]
a = net["pore.surface_area"]
V = net["pore.effective_volume"]
CaV = jnp.where(net["pore.micropore"], C*a*V, 0.0)

# calculate initial mass
c_mi = net["pore.micro_concentration"][net["pore.micropore"]]
V_mi = net["pore.micro_volume"][net["pore.micropore"]]
c = net["pore.concentration"]
V = net["pore.volume"]
m0 = jnp.dot(c, V) + jnp.dot(c_mi/2, V_mi)

# get c, phi, c_mi, phi_d
c = net["pore.concentration"]
phi = net["pore.potential"]
c_mi = net["pore.micro_concentration"]
phi_d = net["pore.donnan_potential"]


def charge_solve(network,
                 phi0,
                 c,
                 phi,
                 V,
                 dt,
                 tol=1e-6,
                 atol=0.0,
                 maxiter=50,
                 p_tol=1e-8,
                 p_max_iter=10,
                 w=1.0):
    """
    
    An implicit solver written in JAX. It is jittable using jax.jit()

    Parameters
    ----------
    network : dict
        Network Dictionary
    fun : callable function
        Function of the form A, b = fun(network)
    phi0 : ndarray
        initial condition of potential field
    c : ndarray
        most recent concentration field
    phi : ndarray
        most recent potential field
    V : ndarray
        volumes used in transient solve
    dt : float
        time step to take in transient solve
    tol : float, optional
        GMRES tolerance. The default is 1e-6.
    atol : float, optional
        GMRES absolute. The default is 0.0.
    maxiter : int, optional
        GMRES max iterations. The default is 50.
    p_tol : float, optional
        picard tolerance. The default is 1e-8.
    p_max_iter : float, optional
        picard max iterations. The default is 10.
    w : float, optional
        damping parameter, value must be between 0 and 1. the default is 1.0.

    Returns
    -------
    phi : ndarray
        Solved potential field

    """
    
    def linear_map(A, V, dt):
        def matvec(x):
            return A @ x + V/dt * x
        return matvec

    
    def body(state):
        
        # get state
        phi, p_iter, p_res = state
        
        # update counter
        p_iter += 1
        
        # update charge source term
        _update_charge_source(network, c, phi)
        
        # get A and b
        Ac, bc = _get_charge_A_and_b(network)
        matvec = linear_map(Ac, V, dt)
        bc_t = bc + V*phi0/dt  # get bc transient
        
        # solve for phi
        phi_new, _ = jax.scipy.sparse.linalg.gmres(
            matvec,
            bc_t,
            x0=phi,
            tol=tol,
            atol=atol,
            maxiter=maxiter
        )
        
        # calculate p_res
        p_res = jnp.linalg.norm(phi_new - phi)
        # print(p_res)
        
        # apply damping
        phi = w * phi_new + (1-w) * phi
        
        return (phi, p_iter, p_res)
    
        
    def cond(state):
        _, p_iter, p_res = state
        return jnp.logical_and(p_res > p_tol,
                               p_iter < p_max_iter)
        
    
    p_iter, p_res = 0, 1.0 
    state = (phi, p_iter, p_res)
    phi, _, _ = lax.while_loop(cond, body, state)
    
    
    return phi


def mass_solve(network,
               c0,
               c,
               phi,
               V,
               dt,
               tol=1e-6,
               atol=0.0,
               maxiter=50,
               p_tol=1e-8,
               p_max_iter=10,
               w=1.0):
    """
    
    An implicit solver written in JAX. It is jittable using jax.jit()

    Parameters
    ----------
    network : dict
        Network Dictionary
    c0 : ndarray
        initial condition of concentration field
    c : ndarray
        most recent concentration field
    phi : ndarray
        most recent potential field
    V : ndarray
        volumes used in transient solve
    dt : float
        time step to take in transient solve
    tol : float, optional
        GMRES tolerance. The default is 1e-6.
    atol : float, optional
        GMRES absolute. The default is 0.0.
    maxiter : int, optional
        GMRES max iterations. The default is 50.
    p_tol : float, optional
        picard tolerance. The default is 1e-8.
    p_max_iter : float, optional
        picard max iterations. The default is 10.
    w : float, optional
        damping parameter, value must be between 0 and 1. the default is 1.0.

    Returns
    -------
    c : ndarray
        Solved concentration field

    """
    
    def linear_map(A, V, dt):
        def matvec(x):
            return A @ x / V + 1/dt * x
        return matvec

    
    def body(state):
        
        # get state
        c, p_iter, p_res = state
        
        # update counter
        p_iter += 1
        
        # update charge source term
        _update_mass_source(network, c, phi)
        
        # get A and b
        Am, bm = _get_mass_A_and_b(network)
        matvec = linear_map(Am, V, dt)
        bm_t = bm / V + c0 / dt
        
        # solve for c
        c_new, _ = jax.scipy.sparse.linalg.gmres(
            matvec,
            bm_t,
            x0=c0,
            tol=tol,
            atol=atol,
            maxiter=maxiter
        )
        
        # calculate p_res
        p_res = jnp.linalg.norm(c_new - c)
        # print(p_res)
        
        # apply damping
        c = w * c_new + (1-w) * c
        
        return (c, p_iter, p_res)
    
        
    def cond(state):
        _, p_iter, p_res = state
        return jnp.logical_and(p_res > p_tol,
                               p_iter < p_max_iter)
        
    
    p_iter, p_res = 0, 1.0 
    state = (c, p_iter, p_res)
    c, _, _ = lax.while_loop(cond, body, state)
    
    
    return c


charge_solve = jax.jit(charge_solve)
mass_solve = jax.jit(mass_solve)


# time stepping
t0 = 0
dt = dt
tf = dt*100
# store solution for first time!
y = jnp.concatenate((c, phi, c_mi, phi_d))
x = jnp.array([t0])
for t in jnp.arange(t0+dt, tf+dt, dt):
    print(f"Time: {t}s")
    # get initial conditions
    c0 = c.copy()
    phi0 = phi.copy()
    # initialize c_old and phi_old
    c_old = c0.copy()
    phi_old = phi0.copy()
    # define gummel convergence
    g_res = jnp.array([100.0, 100.0])
    g_tol = jnp.array([1e-4, 1e-4])
    g_max_iter = 20
    for g_iter in range(g_max_iter):
        print(f"Gummel Iteration No. {g_iter+1}")
        # solve phi, using picard iterations!
        phi = charge_solve(net,
                           phi0=phi0,
                           c=c,
                           phi=phi,
                           V=CaV,
                           dt=dt,
                           tol=1e-6,
                           atol=0.0,
                           maxiter=50,
                           p_tol=1e-8,
                           p_max_iter=10,
                           w=1.0)
        # update potential
        net["pore.potential"] = phi
        # solve c, once!
        c_new = mass_solve(net,
                           c0=c0,
                           c=c,
                           phi=phi,
                           V=V,
                           dt=dt,
                           tol=1e-6,
                           atol=0.0,
                           maxiter=50,
                           p_tol=1e-8,
                           p_max_iter=1,
                           w=1.0)
        # apply damping
        w = 0.5
        c = w * c_new + (1 - w) * c_old
        # update concentration
        net["pore.concentration"] = c
        # update ionic conductance
        _update_charge_conductance(net, c)
        # calculate new residual
        if g_iter > 0:
            g_res = jnp.array([jnp.sum((c - c_old)**2),
                               jnp.sum((phi - phi_old)**2)])
        print(f"  Residual: {g_res}")
        if jnp.all(g_res < g_tol):
            print(f"  Convergence criteria met: {g_res}")
            break
        # break if max no. of iterations reached
        if g_iter == g_max_iter - 1:
            break
        # update old c and phi
        c_old = c.copy()
        phi_old = phi.copy()
    # update source terms on net
    _update_mass_source(net, c, phi)
    _update_charge_source(net, c, phi)
    # calculate effective micro concentrations
    model = models.electrical_double_layer.micro_concentration_eff
    c_mi = model(R=net["pore.mass_source_effective"],
                 c_mi_i=net["pore.micro_concentration_old"],
                 V=net["pore.micro_volume"],
                 dt=net["time_step"])
    net["pore.micro_concentration"] = c_mi
    # update old concentrations
    net["pore.donnan_potential_old"] = net["pore.donnan_potential"].copy()
    net["pore.micro_concentration_old"] = net["pore.micro_concentration"].copy()
    # set "old" concentrations
    net["pore.concentration_old"] = net["pore.concentration"].copy()
    # get c, phi, c_mi, phi_d
    c = net["pore.concentration"]
    phi = net["pore.potential"]
    c_mi = net["pore.micro_concentration"]
    phi_d = net["pore.donnan_potential"]
    # save results as y and x, assume that everything gets saved!
    y = jnp.vstack((y, jnp.concatenate((c, phi, c_mi, phi_d))))
    x = jnp.concatenate((x, jnp.array([t])))

# calculate final mass
c_mi = c_mi[net["pore.micropore"]]
mf = jnp.dot(c, V) + jnp.dot(c_mi/2, V_mi)

# calculate mass balance error
print(f"Mass Balance Error: {abs(mf-m0)/m0*100}%")  # 0.04142848214403494%  @ 1s
print(jnp.average(c))  # 0.11375381915642666
print(jnp.average(abs(phi)))  # 0.016492790763054866