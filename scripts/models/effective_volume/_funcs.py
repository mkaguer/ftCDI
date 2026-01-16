import numpy as np


def cubic(network, rho_sep=1, rho_mi=1, separator=True):

    # initialize effective volume
    eff_vol = network["pore.volume"].copy()
    if separator is True:
        # add volume of separator throats to separator pores
        conns = network["throat.conns@separator"]
        P1 = conns[:, 0]
        P2 = conns[:, 1]
        mask1 = np.isin(P1, network.pores("separator"))
        mask2 = np.isin(P2, network.pores("separator"))
        np.add.at(eff_vol, P1, network["throat.volume@separator"]*mask1)
        np.add.at(eff_vol, P2, network["throat.volume@separator"]*mask2)
        # subtract 1/2 of pore volume from pores connected to separator
        Ps = np.concatenate((P1[~mask1], P2[~mask2]))
        eff_vol[Ps] = 1/2*network["pore.volume"][Ps]
        # correct volume for porosity
        eff_vol[network.pores("separator")] *= rho_sep
    # add volume of micropore throats to micropore pores
    conns = network["throat.conns@micropore"]
    P1 = conns[:, 0]
    P2 = conns[:, 1]
    mask1 = np.isin(P1, network.pores("micropore"))
    mask2 = np.isin(P2, network.pores("micropore"))
    np.add.at(eff_vol, P1, network["throat.volume@micropore"]*mask1)
    np.add.at(eff_vol, P2, network["throat.volume@micropore"]*mask2)
    # add volume of macropore throats to macropores, 1/2 to each neighbour
    conns = network["throat.conns@macropore"]
    P1 = conns[:, 0]
    P2 = conns[:, 1]
    np.add.at(eff_vol, P1, 1/2*network["throat.volume@macropore"])
    np.add.at(eff_vol, P2, 1/2*network["throat.volume@macropore"])
    # correct volume for porosity
    eff_vol[network.pores("micropore")] *= rho_mi

    return eff_vol


def bcc(network, rho_sep=1, rho_mi=1, separator=True):

    # get spacing
    coords = network["pore.coords"]
    conns = network["throat.conns"]
    spacing = np.max(np.abs(coords[conns[0][0]] - coords[conns[0][1]]))
    # initialize effective volume
    eff_vol = network["pore.volume"].copy()
    if separator is True:
        # add volume of separator throats to separator pores
        conns = network["throat.conns@separator"]
        P1 = conns[:, 0]
        P2 = conns[:, 1]
        mask1 = np.isin(P1, network.pores("separator"))
        mask2 = np.isin(P2, network.pores("separator"))
        np.add.at(eff_vol, P1, network["throat.volume@separator"]*mask1)
        np.add.at(eff_vol, P2, network["throat.volume@separator"]*mask2)
        # subtract 1/2 of pore volume from pores connected to separator
        Ps = np.concatenate((P1[~mask1], P2[~mask2]))
        eff_vol[Ps] = 1/2*network["pore.volume"][Ps]
        # correct volume for porosity
        eff_vol[network.pores("separator")] *= rho_sep
    # use total unresolved micropore volume as micropore volume
    pores = network.pores("micropore")
    Ps = network.find_neighbor_pores(pores=pores, flatten=False)
    f = network.find_neighbor_throats
    Ts = [f(pores=p, mode="xnor") for p in Ps]
    # FIXME: the 8 and 4 assume 3D!
    Vp = np.array([np.sum(network["pore.volume"][p])/8 for p in Ps])
    Vt = np.array([np.sum(network["throat.volume"][t])/4 for t in Ts])
    eff_vol[pores] = spacing**3 - Vp - Vt
    # add volume of macropore throats to macropores, 1/2 to each neighbour
    conns = network["throat.conns@macropore"]
    P1 = conns[:, 0]
    P2 = conns[:, 1]
    np.add.at(eff_vol, P1, 1/2*network["throat.volume@macropore"])
    np.add.at(eff_vol, P2, 1/2*network["throat.volume@macropore"])
    # correct volume for porosity
    eff_vol[network.pores("micropore")] *= rho_mi

    return eff_vol
