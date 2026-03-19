import jax.numpy as jnp
import pnmlib as pnm

__all__ = ["effective_diffusivity",
           "reshape",
           "donnan_potential",
           "micropore_concentration",
           "mass_source",
           "charge_source",
           "mass_source_effective"]


def effective_diffusivity(phase, D, epsilon, tau):
    
    return epsilon * D / tau


def reshape(phase,
            prop):
    
    prop = phase[prop]
    
    return jnp.array([prop, prop]).T


def donnan_potential(phase,
                     pore_potential="pore.potential",
                     pore_electrode_potential="pore.electrode_potential",
                     pore_attraction_term="pore.attraction_term",
                     pore_temperature="pore.temperature",
                     pore_capacitance="pore.capacitance",
                     pore_surface_area="pore.surface_area",
                     pore_concentration="pore.concentration"):
    
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
    
    # use fsolve to find the donnan potential
    phi_d0 = phase["pore.donnan_potential"]
    # phi_d0 = jnp.where(phi_d0==-1, 0, phi_d0)
    # start = time.time()
    phi_d = pnm.optimize.newton_krylov(potential_balance, phi_d0, tol=6e-6)
    # stop = time.time()
    # print(f"Newton krylov time: {stop - start}s")
    
    return phi_d


def micropore_concentration(phase,
                            pore_concentration="pore.concentration",
                            pore_donnan_potential="pore.donnan_potential",
                            pore_temperature="pore.temperature",
                            pore_attraction_term="pore.attraction_term"):
    
    # set constants
    R = 8.31446262  # J/mol/K
    F = 96485.3399  # C/mol
    # retrieve properties from phase
    c = phase[pore_concentration]
    phi_d = phase[pore_donnan_potential]
    T = phase[pore_temperature]
    mu_att = phase[pore_attraction_term]
    
    c_mi = 2 * c * jnp.exp(mu_att) * jnp.cosh(F/R/T*phi_d)
    
    return c_mi


def mass_source(phase,
                pore_micro_concentration="pore.micro_concentration",
                pore_micro_concentration_old="pore.micro_concentration_old",
                pore_micro_volume="pore.micro_volume",
                time_step="time_step"):
    
    dt = phase[time_step]
    V_mi = phase[pore_micro_volume]
    c_mi_i = phase[pore_micro_concentration_old]
    c_mi_f = phase[pore_micro_concentration]
    
    S1 = jnp.zeros(len(V_mi))
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
                  time_step="time_step"):
    
    dt = phase[time_step]
    phi_d_i = phase[pore_donnan_potential_old]
    phi_d_f = phase[pore_donnan_potential]
    C = phase[pore_capacitance]
    a = phase[pore_surface_area]
    V = phase[pore_volume]
    
    S1 = jnp.zeros(len(V_mi))
    S2 = - C * a * V * (phi_d_f - phi_d_i) / dt
    rate = S2
    values = {"S1": S1, "S2": S2, "rate": rate}
    
    return values


def mass_source_effective(phase,
                          pore_mass_source="pore.mass_source",
                          pore_concentration="pore.concentration",
                          pore_volume="pore.volume",
                          time_step="time_step"):
    
    # get properties
    R = phase[pore_mass_source]["rate"]   # units: mass / time
    V = phase[pore_volume]
    c = phase[pore_concentration]
    dt = phase[time_step]
    
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
