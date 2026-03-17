import numpy as np
import models.numpy.misc as mods
import openpnm as op
from scipy.integrate import solve_ivp
from scipy_dae.integrate import solve_dae
import time

class CapacitiveDeionization:
    
    def __init__(self,
                 network,
                 phase,
                 time_step,
                 cell_voltage, 
                 mu_att,
                 capacitance,
                 surface_area,
                 feed_concentration
        ):
        self.network = network
        self.phase = phase
        self.time_step = time_step
        self.cell_voltage = cell_voltage
        self.mu_att = mu_att
        self.capacitance = capacitance
        self.surface_area = surface_area
        self.feed_concentration = feed_concentration


    def set_properties(self):
        
        # get phase
        phase = self.phase
        # retrieve cell voltage
        V_cell = self.cell_voltage
        # retrieve mu_att
        mu_att = self.mu_att
        # retrieve capacitance
        C = self.capacitance
        # retrieve surface area
        a = self.surface_area
        # set potential
        phase["pore.potential@cathode"] = -V_cell/2
        phase["pore.potential@anode"] = V_cell/2
        phase["pore.potential@separator"] = 0
        # set electrode potential
        phase["pore.electrode_potential@cathode"] = -V_cell/2
        phase["pore.electrode_potential@anode"] = V_cell/2
        phase["pore.electrode_potential@separator"] = 0
        # set attraction term
        phase["pore.attraction_term@macropore"] = 0
        phase["pore.attraction_term@micropore"] = mu_att
        phase["pore.attraction_term@separator"] = 0
        phase["pore.attraction_term@perforated"] = 0
        # set capacitance
        phase["pore.capacitance@macropore"] = 1
        phase["pore.capacitance@micropore"] = C
        phase["pore.capacitance@separator"] = 1 
        phase["pore.capacitance@perforated"] = 1
        # set surface area
        phase["pore.surface_area@macropore"] = 1
        phase["pore.surface_area@micropore"] = a
        phase["pore.surface_area@separator"] = 1 
        phase["pore.surface_area@perforated"] = 1
        
        
    def apply_sources(self):
        
        # get phase
        phase = self.phase
        # retrieve time step
        dt = self.time_step
        # set 
        
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
                        pore_micro_volume="pore.effective_volume",
                        time_step=dt)
        phase["pore.donnan_potential_old"] = phase["pore.donnan_potential"].copy()
        phase.add_model(propname="pore.charge_source",
                        model=mods.charge_source,
                        domain="micropore",
                        pore_donnan_potential="pore.donnan_potential",
                        pore_donnan_potential_old="pore.donnan_potential_old",
                        pore_volume="pore.effective_volume",
                        pore_capacitance="pore.capacitance",
                        pore_surface_area="pore.surface_area",
                        time_step=dt)


    def run_stokes_flow(self, P_in=1, P_out=1):
        
        # retrieve network
        network = self.network
        # retrieve phase
        phase = self.phase
        # define stokes flow algorithm object
        sf = op.algorithms.StokesFlow(network=network, phase=phase)
        # set BCs
        sf.set_BC(pores=network.pores('inlet'),
                  bctype="value",
                  bcvalues=P_in)
        sf.set_BC(pores=network.pores('outlet'),
                  bctype="value",
                  bcvalues=P_out)
        # run stokes flow
        sf.run()  # This will update ad-dif conductance
        
    
    def build_mass_algorithm(self):
        
        # retrieve network
        network = self.network
        # retrieve phase
        phase = self.phase
        # retrieve feed concentration
        cf = self.feed_concentration
        # create mass transport algorithm
        mt = op.algorithms.TransientAdvectionDiffusion(network=network,
                                                       phase=phase)
        # set settings
        mt.settings["conductance"] = "throat.mass_conductance"
        mt.settings["quantity"] = "pore.concentration"
        mt.settings["pore_volume"] = "pore.effective_volume"
        # set inflow BC
        phase.add_model(propname="pore.inflow",
                        model=mods.inflow,
                        cf=cf,
                        throat_hydraulic_conductance="throat.hydraulic_conductance",
                        pore_pressure="pore.pressure")
        phase.regenerate_models()
        mt.set_source(pores=network.pores("inlet"), propname="pore.inflow")
        # set outflow BC
        mt.set_outflow_BC(pores=network.pores("outlet"))
        # set source terms
        mt.set_source(pores=network.pores("micropore"),
                      propname="pore.mass_source")
        # Finally, apply BCs and source terms to instantiate A and b
        mt._apply_BCs()
        mt._apply_sources()
        
        return mt
        
    
    def build_charge_algorithm(self):
        
        # retrieve network
        network = self.network
        # retrieve phase
        phase = self.phase
        # create charge transport algorithm
        ct = op.algorithms.TransientReactiveTransport(network=network,
                                                      phase=phase)
        # set settings
        ct.settings["conductance"] = "throat.ionic_conductance"
        ct.settings["quantity"] = "pore.potential"
        ct.settings["pore_volume"] = "pore.effective_volume"
        # set source terms
        ct.set_source(pores=network.pores("micropore"),
                      propname="pore.charge_source")
        # Finally, apply BCs and source terms to instantiate A and b
        ct._apply_BCs()
        ct._apply_sources()
        
        return ct


    def rhs_mass(self, t, c):
        
        # retrieve network
        network = self.network
        # retrieve algorithm
        alg = self.mass_alg
        # get volume
        V = network[alg.settings["pore_volume"]]
        # get A and b, assume BCs/sources already applied
        A = alg.A
        b = alg.b
        # calculate dcdt
        dcdt = (-A.dot(c) + b)/V
        
        return dcdt


    def rhs_charge(self, t, phi):
        
        # retrieve network
        network = self.network
        # retrieve phase
        phase = self.phase
        # retrieve algorithm
        alg = self.charge_alg
        # get volume
        V = network[alg.settings["pore_volume"]]
        # get capacitance, surface area
        C = phase["pore.capacitance"]
        a = phase["pore.surface_area"]
        # get A and b
        A = alg.A
        b = alg.b
        # calculate dphidt
        dphidt = (-A.dot(phi) + b)/C/a/V
        
        return dphidt


    def F(self, t, phi, phip):
        
        # retrieve network
        network = self.network
        # get dphidt
        F = self.rhs_charge(t, phi)  # F = dphidt
        # get mask
        mask = network["pore.micropore"]
        F[mask] -= phip[mask]
        
        return F


    def run(self, t0, tf, dt, t_save):
        
        # retrieve network object
        network = self.network
        # retrieve phase object
        phase = self.phase
        # get initial condition of ALL properties to be solved
        c = phase["pore.concentration"]
        phi = phase["pore.potential"]
        c_mi = phase["pore.micro_concentration"]
        phi_d = phase["pore.donnan_potential"]
        # store solution for first time!
        y = np.concatenate((c, phi, c_mi, phi_d))
        x = np.array([t0])
        # initialize current
        I = []
        # perform time stepping
        for t in np.arange(t0+dt, tf+dt, dt):
            print(f"Time: {t}s")
            # define initial condition for this time step
            c0 = c.copy()
            phi0 = phi.copy()
            phip0 = self.rhs_charge(t, phi0)
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
                start = time.time()
                # solve mass balance first
                sol_m = solve_ivp(self.rhs_mass, 
                                  t_span=(0, dt), 
                                  y0=c0, 
                                  t_eval=[dt])
                end = time.time()
                print(f"mass balance took {end - start}s")
                c = sol_m.y[:, -1]
                start = time.time()
                # solve for potential second, use solve_dae
                sol_c = solve_dae(self.F,
                                  t_span=(0, dt),
                                  y0=phi0,
                                  yp0=phip0,
                                  t_eval=[dt])
                end = time.time()
                print(f"charge balance took {end - start}s")
                phi = sol_c.y[:, -1]
                # update concentration and potential
                phase["pore.concentration"] = c
                phase["pore.potential"] = phi
                # regenerate physics based on c and phi
                start = time.time()
                phase.regenerate_models()
                end = time.time()
                print(f"regenerate physics took {end - start}s")
                # update transport algs
                self.mass_alg["pore.concentration"] = c
                self.charge_alg["pore.potential"] = phi
                self.mass_alg._update_A_and_b()
                self.charge_alg._update_A_and_b()
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
            if np.any(np.isclose(t, t_save, atol=1e-10)):
                # get phi_d and c_mi
                c_mi = phase["pore.micro_concentration"]
                phi_d = phase["pore.donnan_potential"]
                # save results as y and x, assume that everything gets saved!
                y = np.vstack((y, np.concatenate((c, phi, c_mi, phi_d))))
                x = np.concatenate((x, np.array([t])))
                # calculate the current
                # divide by two because there are two times the number of throats
                # FIXME: watch this when you scale, I think it works but be careful
                throats = network.throats("separator")
                current = self.charge_alg.rate(throats=throats, mode="group")/2
                I.append(current[0])
                print(f"Outlet Concentration: {c[network.pores('outlet')]}")
            self.current = np.array(I)

        return x, y
                
    def _measure_total_mass(self):
        
        # retrieve phase
        phase = self.phase
        # retrieve network
        network = self.network
        # get pore volume
        V = network["pore.effective_volume"]
        # get concentrations
        c_mi = phase["pore.micro_concentration"]
        c = phase["pore.concentration"]
        # set micropore concentrations to 
        c_mi[~network["pore.micropore"]] = 0
        # calcualte total mass
        total_mass = 0.5*np.dot(c_mi, V) + np.dot(c, V)
        
        return total_mass