"""
v2 does not take jax.jvp. It uses cg to solve Ax=b directly!
"""
import matplotlib.pyplot as plt
import numpy as np
import openpnm as op

op.visualization.set_mpl_style()

np.random.seed(1)

# intialize network
net = op.network.Cubic(shape=[100, 100, 50], spacing=1e-5)
geo_mods = op.models.collections.geometry.spheres_and_cylinders.copy()
del geo_mods['pore.diameter']
net['pore.diameter'] = 5e-6
net.add_model_collection(models=geo_mods)
net.regenerate_models()

# initialize phase
phase = op.phase.Water(network=net)
phase["throat.diffusivity"] = 1e-9  # 1.68e-9
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
mt.set_value_BC(pores=net.pores("xmax"), values=0)

# Finally, apply BCs and source terms to instantiate A and b
mt._apply_BCs()
mt._apply_sources()

# get A and b
A = mt.A
b = mt.b

# get volume
V = net["pore.volume"]
