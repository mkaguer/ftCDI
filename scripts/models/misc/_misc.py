import numpy as np
import scipy as sp
import openpnm.models.geometry.conduit_lengths as _conduit_lengths
from openpnm.models.physics._utils import _poisson_conductance
import time

__all__ = ["continuum_size_factor",
           "micropore_size_factor",
           "conductivity",
           "donnan_potential",
           "micropore_concentration",
           "mass_source",
           "charge_source",
           "attraction_term",
           "micropore_concentration_e",
           "inflow",
           "outflow",
           "generic_diffusive",
           "effective_diffusivity",
           "mass_source_effective"]

def continuum_size_factor(
    network,
    throat_diameter="throat.diameter",
    throat_length="throat.length"):
    r"""
    This model calculates the size factor for a conduit assuming one throat
    with square cross-section that is constant across it's length. The pores
    in the conduit are assumed to have negligeable resistance. We can use this
    size factor for both hydraulic and diffusive size factors in a continuum.
    It is simply A/L!
    """
    Nt = len(network['throat.conns'])
    D = network[throat_diameter]
    L = network[throat_length]
    A = D ** 2
    vals = np.vstack([np.ones(Nt)*1e16, A/L, np.ones(Nt)*1e16]).T
    return vals


def micropore_size_factor(
    network,
    pore_diameter="pore.diameter",
    throat_diameter="throat.diameter"):
    r"""
    Calculates the diffusive size factor assuming the micropore has negligible
    resistance!
    """
    D1, Dt, D2 = network.get_conduit_data(pore_diameter.split('.', 1)[-1]).T
    L1, Lt, L2 = _conduit_lengths.cubes_and_cuboids(
        network,
        pore_diameter=pore_diameter,
        throat_diameter=throat_diameter
    ).T
    # Fi is the integral of (1/A) dx, x = [0, Li]
    F1 = L1 / D1**2
    F2 = D2  # micropore is always pore 2!
    Ft = Lt / Dt**2
    # values
    vals = np.vstack([1/F1, 1/Ft, 1/F2]).T
    return vals


def conductivity(
    phase,
    concentration="pore.concentration",
    diffusivity="pore.diffusivity",
    temperature="pore.temperature"):
    
    R = 8.31446262  # J/mol/K
    F = 96485.3399  # C/mol
    c = phase[concentration]
    T = phase[temperature]
    D = phase[diffusivity]
    
    K = 2*F**2*c*D/R/T
    
    return K


def donnan_potential(phase,
                     pore_potential="pore.potential",
                     pore_electrode_potential="pore.electrode_potential",
                     pore_attraction_term="pore.attraction_term",
                     pore_temperature="pore.temperature",
                     pore_capacitance="pore.capacitance",
                     pore_surface_area="pore.surface_area",
                     pore_concentration="pore.concentration"):
    
    Np = len(phase[pore_potential])
    # set constants
    R = 8.31446262  # J/mol/K
    F = 96485.3399  # C/mol
    # retrieve properties from phase
    phi = phase[pore_potential]
    phi_e = phase[pore_electrode_potential]
    mu_att = phase[pore_attraction_term]
    T = phase[pore_temperature]
    C = phase[pore_capacitance]
    a = phase[pore_surface_area]
    c = phase[pore_concentration]
    
    def micropore_charge_concentration(phi_d):
        
        sigma = -2 * c * np.exp(mu_att) * np.sinh(F/R/T * phi_d)
        
        return sigma
    
    def stern_potential(sigma):
        
        phi_st = -sigma * F / C / a
        
        return phi_st
    
    def potential_balance(phi_d):
        
        sigma = micropore_charge_concentration(phi_d)
        phi_st = stern_potential(sigma)
        f = phi_e - phi - phi_d - phi_st
        
        return f
    
    # use fsolve to find the donnan potential
    phi_d0 = phase["pore.donnan_potential"]
    mask = np.isnan(phi_d0)
    phi_d0[mask] = 0
    # start = time.time()
    phi_d = sp.optimize.newton_krylov(potential_balance, phi_d0, f_tol=6e-6)
    # stop = time.time()
    # print(f"Newton krylov time: {stop - start}s")
    
    return phi_d


def micropore_concentration(phase,
                            pore_concentration="pore.concentration",
                            pore_donnan_potential="pore.donnan_potential",
                            pore_temperature="pore.temeprature",
                            pore_attraction_term="pore.attraction_term"):
    
    # set constants
    R = 8.31446262  # J/mol/K
    F = 96485.3399  # C/mol
    # retrieve properties from phase
    c = phase[pore_concentration]
    phi_d = phase[pore_donnan_potential]
    T = phase[pore_temperature]
    mu_att = phase[pore_attraction_term]
    
    c_mi = 2 * c * np.exp(mu_att) * np.cosh(F/R/T*phi_d)
    
    return c_mi

def mass_source(phase,
                pore_micro_concentration="pore.micro_concentration",
                pore_micro_concentration_old="pore.micro_concentration_old",
                pore_micro_volume="pore.micro_volume",
                time_step=0.1):
    
    dt = time_step
    V_mi = phase[pore_micro_volume]
    c_mi_i = phase[pore_micro_concentration_old]
    c_mi_f = phase[pore_micro_concentration]
    
    S1 = 0
    S2 = - 1/2 * V_mi * (c_mi_f - c_mi_i) / dt
    rate = S2
    values = {"S1": S1, "S2": S2, "rate": rate}
    
    return values


def charge_source(phase,
                  pore_donnan_potential="pore.donnan_potential",
                  pore_donnan_potential_old="pore.donnan_potential_old",
                  pore_volume="pore.volume",
                  pore_capacitance="pore.capacitance",
                  pore_surface_area="pore.surface_area",
                  time_step=0.1):
    
    dt = time_step
    phi_d_i = phase[pore_donnan_potential_old]
    phi_d_f = phase[pore_donnan_potential]
    C = phase[pore_capacitance]
    a = phase[pore_surface_area]
    V = phase[pore_volume]
    
    S1 = 0
    S2 = - C * a * V * (phi_d_f - phi_d_i) / dt
    rate = S2
    values = {"S1": S1, "S2": S2, "rate": rate}
    
    return values


def attraction_term(phase,
                    pore_micro_concentration="pore.micro_concentration",
                    pore_attraction_energy="pore.attraction_energy"):
    
    E = phase[pore_attraction_energy]
    c_mi = phase[pore_micro_concentration]
    mu_att = E/c_mi
    
    return mu_att


def micropore_concentration_e(phase,
                              pore_concentration="pore.concentration",
                              pore_attraction_energy="pore.attraction_energy",
                              pore_temperature="pore.temperature",
                              pore_donnan_potential="pore.donnan_potential"):
    """
    This implicit version is necessary when using attraction energy!

    """
    Np = len(phase[pore_concentration])

    # set constants
    R = 8.31446262  # J/mol/K
    F = 96485.3399  # C/mol

    # retrieve properties from phase
    c = phase[pore_concentration]
    E = phase[pore_attraction_energy]
    T = phase[pore_temperature]
    phi_d = phase[pore_donnan_potential]
    
    def fun(c_mi):
        
        f = c_mi - 2*c*np.exp(E/c_mi)*np.cosh(F/R/T*phi_d)
        
        return f
    
    # use fsolve to find the donnan potential
    c_mi0 = np.zeros(Np) # initial guess
    c_mi = sp.optimize.fsolve(fun, c_mi0)
    
    return c_mi
    

def inflow(phase,
           cf=100,
           throat_hydraulic_conductance="throat.hydraulic_conductance",
           pore_pressure="pore.pressure"):
    
    # get network
    network = phase.network
    # get conns
    conns = network["throat.conns"]
    # calculate flow rate for each throat, q12
    P12 = phase[pore_pressure][conns]
    gh = phase[throat_hydraulic_conductance]
    Q12 = -gh * np.diff(P12, axis=1).squeeze()
    # calculate net flow rate for each pore
    Qp = np.zeros(network.Np)
    np.add.at(Qp, conns[:, 0], -Q12)
    np.add.at(Qp, conns[:, 1], Q12)
    # calculate source terms such that S1*X + S2
    S1 = 0
    S2 = -Qp*cf
    r = S2
    values = {"S1": S1, "S2": S2, "rate": r}
    
    return values


def outflow(phase,
            pore_concentration="pore.concentration",
            throat_hydraulic_conductance="throat.hydraulic_conductance",
            pore_pressure="pore.pressure",
            pore_micro_concentration="pore.micro_concentration",
            pore_micro_concentration_old="pore.micro_concentration_old",
            pore_micro_volume="pore.micro_volume",
            time_step=0.1):

    # get network
    network = phase.network
    # get concentration, X
    X = phase[pore_concentration]
    # get properties for S2
    dt = time_step
    V_mi = phase[pore_micro_volume]
    c_mi_i = phase[pore_micro_concentration_old]
    c_mi_f = phase[pore_micro_concentration]
    # get conns
    conns = network["throat.conns"]
    # calculate flow rate for each throat, q12
    P12 = phase[pore_pressure][conns]
    gh = phase[throat_hydraulic_conductance]
    Q12 = -gh * np.diff(P12, axis=1).squeeze()
    # calculate net flow rate for each pore
    Qp = np.zeros(network.Np)
    np.add.at(Qp, conns[:, 0], -Q12)
    np.add.at(Qp, conns[:, 1], Q12)
    # calculate source terms such that S1*X + S2
    S1 = -Qp
    S2 = - 1/2 * V_mi * (c_mi_f - c_mi_i) / dt
    r = S1*X + S2
    values = {"S1": S1, "S2": S2, "rate": r}
    
    return values


def generic_diffusive(phase,
                      pore_diffusivity="pore.diffusivity",
                      throat_diffusivity="throat.diffusivity",
                      size_factors="throat.diffusive_size_factors"):
    r"""
    Calculates the diffusive conductance of conduits in network.

    Parameters
    ----------
    %(phase)s
    pore_diffusivity : str
        %(dict_blurb)s pore diffusivity
    throat_diffusivity : str
        %(dict_blurb)s throat diffusivity
    size_factors : str
        %(dict_blurb)s conduit diffusive size factors

    Returns
    -------
    %(return_arr)s diffusive conductance

    """
    G = _poisson_conductance(phase=phase,
                                pore_conductivity=pore_diffusivity,
                                throat_conductivity=throat_diffusivity,
                                size_factors=size_factors)
    return np.vstack([G, G]).T


def effective_diffusivity(phase, D, epsilon, tau):
    
    return epsilon * D / tau


def mass_source_effective(phase,
                          pore_mass_source="pore.mass_source",
                          pore_concentration="pore.concentration",
                          pore_volume="pore.volume",
                          time_step=0.1):
    
    # get network
    network = phase.network
    
    R = phase[pore_mass_source + ".rate"]   # units: mass / time
    V = network[pore_volume]
    c = phase[pore_concentration]
    
    # Maximum removable mass rate (positive number)
    R_max = c * V / time_step
    
    # Start with original rate
    R_eff = R.copy()
    
    # Identify sinks (negative rates)
    sink = R < 0
    
    # Limit sink magnitude
    R_eff[sink] = np.maximum(R[sink], -R_max[sink])
    
    S1 = 0
    S2 = R_eff
    rate = S2
    values = {"S1": S1, "S2": S2, "rate": rate}
    
    return values