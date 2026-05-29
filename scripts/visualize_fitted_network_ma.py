import numpy as np
import porespy as ps
import openpnm as op

# import network (full cell with labels)
data = np.load("../networks/fit_network_macro.npz")
data = {key: np.array(data[key]) for key in data.files}

# convert to openpnm
net = op.io.network_from_porespy(data)

# convert to voxel image
im = ps.networks.generate_voxel_image(net, rtol=0.05)

# save image as stl
im = im != 0
res = 1e-5 * 9 / im.shape[0]
im = np.pad(im, pad_width=((3,0),(3,0),(3,0)), mode="constant", constant_values=True)
ps.io.to_stl(im, filename="../paraview/fit_network_macro.stl", voxel_size=res)

# save network
# spacing = 1e-5
# net["pore.coords"] += spacing
# op.io.project_to_xdmf(net.project, filename="../paraview/visualize_fitted_network_max.xdmf")