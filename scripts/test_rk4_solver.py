import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import openpnm as op
import time

op.visualization.set_mpl_style()

np.random.seed(1)

#%% Write custom RK4 solver

def my_RK4(fun, t_span, y0, h, t_eval=None):

    # get t0 and tf
    t0 = t_span[0]
    tf = t_span[1]
    # initialize ys and ts
    ys = [y0]
    ts = [t0]
    # initialize yn
    yn = y0
    # time marching
    for tn in np.arange(t0+h, tf+h, h):
        # get ks
        k1 = fun(tn, yn)
        k2 = fun(tn + h/2, yn + h*k1/2)
        k3 = fun(tn + h/2, yn + h*k2/2)
        k4 = fun(tn + h, yn + h*k3)
        # calculate new yn
        yn = yn + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        # append tn and yn to ts and ys
        if t_eval is None:
            ts.append(tn)
            ys.append(yn)
        else:
            if np.any(np.isclose(tn, t_eval, atol=1e-10)):
                ts.append(tn)
                ys.append(yn)
    # turn lists into arrays
    ts = np.array(ts)
    ys = np.array(ys)
    
    return ts, ys


#%% Simple test case

def dydt(t, y):
    
    return y + 5


# select initial condition
t0 = 0
y0 = 1

# select parameters
h = 0.01
tf = 1

# solve using my_RK4
ts, ys = my_RK4(dydt, t_span=(t0, tf), y0=y0, h=h)

# solve using solve_ivp
sol = solve_ivp(dydt,
                t_span=(t0, tf),
                y0=np.array([y0]),
                method="RK45",
                t_eval=np.arange(t0, tf+h, h))

# plot results
plt.figure(1)
plt.plot(ts, ys, label="My RK4")
plt.plot(sol.t, sol.y[0], "--", label="RK45")
plt.ylabel("y")
plt.xlabel("x")
plt.legend()
plt.show()

#%% Solve PNM

# intialize network
net = op.network.Cubic(shape=[10, 10, 10], spacing=1e-5)
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
net.add_model_collection(models=geo_mods)
net.regenerate_models()

# initialize phase
phase = op.phase.Water(network=net)
phase["throat.diffusivity"] = 1.68e-9
phys_mods = op.models.collections.physics.basic.copy()
phase.add_model_collection(models=phys_mods)
phase.regenerate_models()

# set initial concentration
c0 = np.zeros(net.Np)
c0[net.pores("xmin")] = 1.0
net["pore.concentration"] = c0

# initialize mass transport
mt = op.algorithms.TransientFickianDiffusion(network=net,
                                             phase=phase)
mt.set_value_BC(pores=net.pores("xmin"), values=1)

# Finally, apply BCs and source terms to instantiate A and b
mt._apply_BCs()
mt._apply_sources()

def rhs_mass(t, c):

    # FIXME: use alg volume
    # retrieve properties
    V = phase[mt.settings["pore_volume"]]
    # get A and b for mass
    Am = mt.A
    bm = mt.b
    # calculate dcdt
    dcdt = (-Am.dot(c) + bm)/V

    return dcdt

# select initial condition
t0 = 0
y0 = c0

# select parameters
h = 0.001
tf = 10

# solve using solve_ivp
start = time.time()
sol = solve_ivp(rhs_mass,
                t_span=(t0, tf),
                y0=y0,
                method="RK45",
                t_eval=np.arange(t0, tf+h, h))
stop = time.time()
print(f"RK45 Time: {stop - start}s")

# solve using my_RK4
start = time.time()
ts, ys = my_RK4(rhs_mass, t_span=(t0, tf), y0=y0, h=h,
                t_eval=np.arange(t0, tf+h*10, h*10))
stop = time.time()
print(f"RK4 Time: {stop - start}s")

# average concentration
c_avg = np.average(sol.y, axis=0)
y_avg = np.average(ys, axis=1)

# plot
plt.figure(2)
plt.plot(sol.t, c_avg, label="RK45")
plt.plot(ts, y_avg, "--", label="My RK4")
plt.ylabel("Average Concentration")
plt.xlabel("time (s)")
plt.legend()
plt.show()
