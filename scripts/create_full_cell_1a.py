import numpy as np
import openpnm as op
import network




test = False

# set dimensions
if test:
    l_sep = 2.714e-5
    d = 3e-5
else:
    l_sep = 1.9e-4  # should be 190um (Guyes)
    d = 2e-4

# load network
if test:
    data = np.load('../networks/perforated_network_1a_test.npz')
else:
    data = np.load('../networks/perforated_network_1a.npz')
data = {key: np.array(data[key]) for key in data.files}

# convert to openpnm object
net = op.io.network_from_porespy(data)

# infer spacing
conns = net["throat.conns"]
coords = net["pore.coords"]
spacing = np.max(np.abs(coords[conns[0, 0]] - coords[conns[0, 1]]))
spacing = np.round(spacing, 10)

# create full cell
net = network.create_full_cell_v2(net, l_separator=l_sep)

# set pore/throat diameter of separator
net["pore.diameter@separator_channel"] = d
net["throat.diameter@separator_channel"] = d
net["throat.diameter@separator"] = spacing

# add separator_channel pore/throats to separator label
throats = net["throat.separator_channel"]
pores = net["pore.separator_channel"]
net["throat.separator"][throats] = True
mask = np.zeros(net.Np, dtype=bool)
mask[pores] = True
net["pore.separator"] = mask

# re-label perforated throats connected to a macropore as "macropore"
throats = op.topotools.find_interface_throats(net,
                                              P1=net.pores("macropore"),
                                              P2=net.pores("perforated"))
net["throat.perforated"][throats] = False
net["throat.macropore"][throats] = True

# remove macropore label from perforated throats
throats = net.throats("perforated")
net["throat.macropore"][throats] = False

# export to paraview
if test:
    op.io.project_to_xdmf(project=net.project,
                          filename="../paraview/create_full_cell_1a_test")
else:
    op.io.project_to_xdmf(project=net.project,
                          filename="../paraview/create_full_cell_1a")

if test:
    # export to .npz file
    np.savez_compressed("../networks/create_full_cell_1a_test.npz", **net)
else:
    # export to .npz file
    np.savez_compressed("../networks/create_full_cell_1a.npz", **net)
