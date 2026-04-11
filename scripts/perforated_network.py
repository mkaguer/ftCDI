"""
This script creates a perforated network from the scaled network

Created by: Mike McKague
Date: January 13, 2026
"""

import openpnm as op
import numpy as np
import network

test = False

# set dimensions
if test:
    length = 4e-5
    d = 3e-5
else:
    length = 3e-4  # already sliced
    d = 2e-4

# load network
if test:
    data = np.load('../networks/scale_network_test.npz')
else:
    data = np.load('../networks/scale_network.npz')
data = {key: np.array(data[key]) for key in data.files}

# retrieve diameters
Dp = data["pore.diameter"]
Dt = data["throat.diameter"]

# get coords and conns
conns = data["throat.conns"]
coords = data["pore.coords"]

# assign new pore indices b/c of stitching
perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
inv_perm = np.empty_like(perm)
inv_perm[perm] = np.arange(len(perm))  # maps old indices
coords = coords[perm]
conns = inv_perm[conns]

# re-order Dp, Dt already getting re-ordered
Dp = Dp[perm]

# get spacing
spacing = np.max(np.abs(coords[conns[0, 0]] - coords[conns[0, 1]]))
spacing = np.round(spacing, 10)

# get shape
Lx = np.round((np.max(coords[:, 0]) - np.min(coords[:, 0]))/spacing, 10) + 1
Ly = np.round((np.max(coords[:, 1]) - np.min(coords[:, 1]))/spacing, 10) + 1
Lz = np.round((np.max(coords[:, 2]) - np.min(coords[:, 2]))/spacing, 10) + 1
shape = [int(Lx), int(Ly), int(Lz)]

# make my bcc
net = network.my_bcc(shape, spacing)

# add pore diameters
net["pore.diameter@macropore"] = Dp
net["pore.diameter@micropore"] = 1e-16  # can't be 1e-32!

# Get indices for assigning throat diameters
mapping = {tuple(row): i for i, row in enumerate(conns)}
idx = np.array([mapping[tuple(row)] for row in net["throat.conns@macropore"]])

# assign throat diameters
net["throat.diameter@macropore"] = Dt[idx]
D = net["pore.diameter"]
throat_conns = net["throat.conns@micropore"]
D_ma = np.where(net["pore.macropore"][throat_conns[:, 0]],
                net["pore.diameter"][throat_conns[:, 0]],
                net["pore.diameter"][throat_conns[:, 1]])
net["throat.diameter@micropore"] = D_ma
# net["throat.diameter@micropore"] = np.sqrt((1/2*spacing)**2 + (1/2*spacing)**2)

# slice network
if test:
    net = network.slice_network(net, length=length, axis=0)
else:
    pass

# cut hole
net = network.cut_hole(net, d=d, axis=0)

# add perforated pores
network = network.add_perforated_pores(net, axis=0)
network["pore.diameter@perforated"] = d
D = net["pore.diameter"]
throat_conns = net["throat.conns@perforated"]
network["throat.diameter@perforated"] = 1.0 * np.min(D[throat_conns], axis=1)

# export to paraview
network["throat.radius"] = network["throat.diameter"]/2
if test:
    op.io.project_to_xdmf(project=net.project,
                          filename="../paraview/perforated_network_test" + f"_{d}")
else:
    op.io.project_to_xdmf(project=net.project,
                          filename="../paraview/perforated_network" + f"_{d}")

# export to .npz file
if test:
    np.savez_compressed("../networks/perforated_network_test" + f"_{d}" + ".npz", **net)
else:
    np.savez_compressed("../networks/perforated_network" + f"_{d}" + ".npz", **net)
