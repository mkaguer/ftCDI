"""
This script creates a perforated network from the scaled network

Created by: Mike McKague
Date: January 13, 2026
"""

import openpnm as op
import numpy as np
import network

# set dimensions
length = 5e-5
d = 5e-5

# load network
data = np.load('../networks/scale_network_1a.npz')
data = {key: np.array(data[key]) for key in data.files}

# retrieve diameters
Dp = data["pore.diameter"]
Dt = data["throat.diameter"]

# get coords and conns
conns = data["throat.conns"]
coords = data["pore.coords"]

# get spacing
spacing = np.max(np.abs(coords[conns[0, 0]] - coords[conns[0, 1]]))
spacing = np.round(spacing, 10)

# get shape
Lx = np.round((np.max(coords[:, 0]) - np.min(coords[:, 0]))/spacing, 10) + 1
Ly = np.round((np.max(coords[:, 1]) - np.min(coords[:, 1]))/spacing, 10) + 1
Lz = np.round((np.max(coords[:, 2]) - np.min(coords[:, 2]))/spacing, 10) + 1
shape = [int(Lx), int(Ly), int(Lz)]
shape = [10, 10, 10]

# make my bcc
net = network.my_bcc(shape, spacing)

# add pore diameters
net["pore.diameter@macropore"] = Dp
net["pore.diameter@micropore"] = 1e-16  # can't be 1e-32!

# assign throat diameters
net["throat.diameter@macropore"] = Dt
D = net["pore.diameter"]
throat_conns = net["throat.conns@micropore"]
net["throat.diameter@micropore"] = 1.0 * np.max(D[throat_conns], axis=1)

# slice network
net = network.slice_network(net, length=length, axis=0)

# cut hole
net = network.cut_hole(net, d=d, axis=0)

# add perforated pores
network = network.add_perforated_pores(net, axis=0)
network["pore.diameter@perforated"] = d
D = net["pore.diameter"]
throat_conns = net["throat.conns@perforated"]
network["throat.diameter@perforated"] = 1.0 * np.min(D[throat_conns], axis=1)

# export to paraview
op.io.project_to_xdmf(project=net.project,
                      filename="../paraview/perforated_network_1a")

# export to .npz file
np.savez_compressed("../networks/perforated_network_1a.npz", **net)
