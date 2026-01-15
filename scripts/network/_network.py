import openpnm as op
import numpy as np

__all__ = ["my_bcc",
           "slice_network",
           "cut_hole",
           "add_perforated_pores",
           "create_full_cell"]


def my_bcc(shape, spacing=1):

    # wrap openpnm bcc network
    network = op.network.BodyCenteredCubic(shape=shape, spacing=spacing)
    # trim body to body throats
    op.topotools.trim(network=network, throats=network.throats("body_to_body"))
    # get list of labels to remove at end
    labels = network.labels()
    # add custom labels
    network.set_label(label="micropore",
                      pores=network.pores("body"),
                      throats=network.throats("corner_to_body"))
    network.set_label(label="macropore",
                      pores=network.pores("corner"),
                      throats=network.throats("corner_to_corner"))
    # clean-up labels
    for label in labels:
        del network[label]
    # If 2d, micropores get offset by half of spacing, here is a fix:
    if len(shape) == 2:
        network["pore.coords"][:, 2][network.pores("micropore")] -= spacing/2

    return network


def slice_network(network, length, axis=0):

    # get coords
    coords = network["pore.coords"]
    # get pores to trim
    pores = coords[:, axis] >= length
    # trim pores
    op.topotools.trim(network=network, pores=pores)

    return network


def cut_hole(network, d, axis=0):

    # get coords and conns
    coords = network["pore.coords"]
    conns = network["throat.conns"]
    # get spacing
    spacing = np.max(np.abs(coords[conns[0, 0]] - coords[conns[0, 1]]))
    spacing = np.round(spacing, 10)
    # get "y" and "z" depending on axis
    if axis == 0:
        y = coords[:, 1]
        z = coords[:, 2]
    if axis == 1:
        y = coords[:, 0]
        z = coords[:, 2]
    if axis == 2:
        y = coords[:, 0]
        z = coords[:, 1]
    # get centre point
    ct_y = (np.max(y) - np.min(y) + spacing)/2
    ct_z = (np.max(z) - np.min(z) + spacing)/2
    # calculate r for ALL points
    r = np.sqrt((y-ct_y)**2 + (z-ct_z)**2)
    # find all points with r < d/2
    pores = r < d/2
    # add label
    mask1 = r < d/2 + spacing
    mask2 = network["pore.macropore"]
    network["pore.hole"] = mask1 * mask2
    # trim pores
    op.topotools.trim(network=network, pores=pores)

    return network


def add_perforated_pores(network, axis=0):

    # get coords and conns
    coords = network["pore.coords"]
    conns = network["throat.conns"]
    # get lattice spacing
    spacing = np.max(np.abs(coords[conns[0, 0]] - coords[conns[0, 1]]))
    spacing = np.round(spacing, 10)
    # determine transverse axes
    transverse_axes = [0, 1, 2]
    transverse_axes.remove(axis)
    t1, t2 = transverse_axes
    # compute transverse center
    y = coords[:, t1]
    z = coords[:, t2]
    ct_y = (y.max() - y.min() + spacing) / 2
    ct_z = (z.max() - z.min() + spacing) / 2
    # determine donor extent
    ax_min = coords[:, axis].min()
    ax_max = coords[:, axis].max()
    num_pores = int((ax_max - ax_min) / spacing + 1)
    # create donor network
    shape = np.ones(3, dtype=int)
    shape[axis] = num_pores
    donor = op.network.Cubic(shape, spacing=spacing)
    donor_coords = donor["pore.coords"]
    # center donor pores transversely
    donor_coords[:, t1] = ct_y
    donor_coords[:, t2] = ct_z
    # connect donor pores to hole pores
    hole_mask = network["pore.hole"]
    new_conns = []
    for i, coord in enumerate(donor_coords[:, axis]):
        same_plane = coords[:, axis] == coord
        P_existing = np.where(same_plane & hole_mask)[0]
        P_donor = np.full(len(P_existing), network.Np + i)
        new_conns.append(np.column_stack((P_existing, P_donor)))
    # stack donor–hole connections
    if new_conns:
        new_conns = np.vstack(new_conns)
    else:
        new_conns = np.empty((0, 2), dtype=int)
    # connect donor pores to each other
    donor_indices = np.arange(network.Np, network.Np + num_pores)
    donor_chain = np.column_stack((donor_indices[:-1], donor_indices[1:]))
    new_conns = np.vstack((new_conns, donor_chain)).astype(int)
    # extend network
    op.topotools.extend(
        network,
        coords=donor_coords,
        conns=new_conns,
        labels=["perforated"],
    )
    # assign inlet and outlet labels
    pore_coords = network["pore.coords"][:, axis]
    network.set_label(
        label="inlet",
        pores=network["pore.perforated"] & (pore_coords == ax_min),
    )
    network.set_label(
        label="outlet",
        pores=network["pore.perforated"] & (pore_coords == ax_max),
    )
    # update macropore throats
    network["throat.macropore"] = (
        network["throat.macropore"]
        + network["throat.perforated"]
    )
    # remove temporary labels
    for label in ["pore.all", "throat.all", "pore.hole"]:
        del network[label]

    return network


def create_full_cell(network, l_separator=1e-4):

    # get Nt
    Nt = network.Nt
    # get coords
    coords = network["pore.coords"]
    # get plane
    x_plane = np.max(coords[:, 0]) + l_separator/2
    # get distance to plane
    dist = x_plane - coords[:, 0]
    # mirror network to get donor network
    donor = network.copy()
    donor = op.io.network_from_porespy(donor)
    coords = donor["pore.coords"].copy()
    coords[:, 0] += 2*dist
    donor["pore.coords"] = coords
    try:
        # get throat coords
        t_coords = network["throat.coords"]
        # get plane
        x_plane_t = np.max(t_coords[:, 0]) + l_separator/2
        # get distance to plane
        dist_t = x_plane_t - t_coords[:, 0]
        # mirror throats in donor network
        t_coords = donor["throat.coords"].copy()
        t_coords[:, 0] += 2*dist_t
        donor["throat.coords"] = t_coords
    except:
        pass
    # stitch donor to network
    P_network = network.pores("outlet")
    P_donor = network.pores("outlet")  # reflected
    op.topotools.stitch(network,
                        donor,
                        P_network,
                        P_donor,
                        method="nearest",
                        label_stitches="separator")
    # add back donor throat coords
    try:
        network["throat.coords"][Nt:2*Nt, :] = donor["throat.coords"]
    except:
        pass
    # add separator nodes
    throats = network.throats("separator")
    network = _add_throat_nodes(network, throats, label="separator")
    # add cathode and anode labels
    coords = network["pore.coords"]
    network["pore.cathode"] = coords[:, 0] < x_plane
    network["pore.anode"] = coords[:, 0] > x_plane
    # remove and reassign inlet/outlet labels to macropores
    mask = network["pore.perforated"]
    del network["pore.inlet"]
    del network["pore.outlet"]
    network["pore.inlet"] = (coords[:, 0] == np.min(coords[:, 0])) * mask
    network["pore.outlet"] = (coords[:, 0] == np.max(coords[:, 0])) * mask

    return network


def _add_throat_nodes(network, throats, label="node"):

    # find coords of new nodes
    conns = network["throat.conns"]
    coords = network["pore.coords"]
    temp = coords[conns][throats]  # Nt by 2 by 3
    node_coords = temp[:, 0, :] + np.diff(temp, axis=1).squeeze()/2
    # find conns of new nodes
    Np_max = network.Np + len(node_coords)
    P_nodes = np.arange(network.Np, Np_max)
    P1 = conns[throats][:, 0]
    P2 = conns[throats][:, 1]
    node_conns = np.vstack((np.array([P1, P_nodes]).T,
                            np.array([P2, P_nodes]).T))
    # delete throats
    op.topotools.trim(network, throats=throats)
    # add throat nodes
    op.topotools.extend(network,
                        coords=node_coords,
                        conns=node_conns,
                        labels=[label])

    return network
