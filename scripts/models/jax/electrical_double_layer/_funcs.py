import jax.numpy as jnp
import pnmlib as pnm

__all__ = ["conductivity",
           "poisson_conductance",
           "donnan_potential",
           "micropore_concentration",
           "mass_source",
           "charge_source",
           "mass_source_effective",
           "mass_source_effective_v2",
           "micro_concentration_eff"]


def conductivity(c, D, T):
    
    R = 8.31446262  # J/mol/K
    F = 96485.3399  # C/mol
    
    K = 2*F**2*c*D/R/T
    
    return K


def poisson_conductance(network, Kp, Kt, sf):
    
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
    phi_d = pnm.optimize.newton_krylov_scan_v2(potential_balance,
                                            phi_d0, tol=1e-8)
    
    return phi_d


def micropore_concentration(c, phi_d, T, mu_att):
    
    # set constants
    R = 8.31446262  # J/mol/K
    F = 96485.3399  # C/mol
    
    c_mi = 2 * c * jnp.exp(mu_att) * jnp.cosh(F/R/T*phi_d)
    
    return c_mi


def mass_source(c_mi_f, c_mi_i, V_mi, dt):
    
    S1 = jnp.zeros(len(V_mi))
    S2 = - 1/2 * V_mi * (c_mi_f - c_mi_i) / dt
    rate = S2
    values = {"S1": S1, "S2": S2, "rate": rate}
    
    return values


def charge_source(phi_d_f, phi_d_i, C, a, V, dt):
    
    S1 = jnp.zeros(len(V))
    S2 = - C * a * V * (phi_d_f - phi_d_i) / dt
    rate = S2
    values = {"S1": S1, "S2": S2, "rate": rate}
    
    return values


def mass_source_effective(R, c, V, dt):
    
    # get rate
    R = R["rate"]
    
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


def micro_concentration_eff(R, c_mi_i, V, dt):
    
    R = R["rate"]
    c_mi = -2 * dt * R / V + c_mi_i
    
    return c_mi