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


def bcc_fast(network, rho_sep=1.0, rho_mi=1.0, separator=True):

    # --- Pre-fetch arrays (important!) ---
    pore_vol = network["pore.volume"]
    throat_vol = network["throat.volume"]
    eff_vol = pore_vol.copy()

    coords = network["pore.coords"]
    conns_all = network["throat.conns"]

    # spacing from first throat (unchanged logic)
    spacing = np.max(np.abs(coords[conns_all[0, 0]] - coords[conns_all[0, 1]]))

    # --- Separator handling ---
    if separator:
        sep_pores = network.pores("separator")
        sep_mask = np.zeros(network.Np, dtype=bool)
        sep_mask[sep_pores] = True

        conns = network["throat.conns@separator"]
        tvol = network["throat.volume@separator"]

        P1 = conns[:, 0]
        P2 = conns[:, 1]

        m1 = sep_mask[P1]
        m2 = sep_mask[P2]

        np.add.at(eff_vol, P1[m1], tvol[m1])
        np.add.at(eff_vol, P2[m2], tvol[m2])

        # pores connected to separator (non-separator side)
        Ps = np.concatenate((P1[~m1], P2[~m2]))
        eff_vol[Ps] = 0.5 * pore_vol[Ps]

        eff_vol[sep_pores] *= rho_sep

    # --- Micropore unresolved volume (correct shared-throat logic) ---
    mi_pores = network.pores("micropore")
    all_pores = network.pores("all")
    
    pore_vol = network["pore.volume"]
    throat_vol = network["throat.volume"]
    
    # Neighbor pores for each micropore
    nbr_pores = network.find_neighbor_pores(pores=mi_pores,
                                            flatten=False)
    
    # Pre-fetch throat neighbors for *all* pores once
    all_pore_throats = network.find_neighbor_throats(pores=all_pores,
                                                     flatten=False)
    
    Vp = np.empty(len(mi_pores))
    Vt = np.empty(len(mi_pores))
    
    for i, Ps in enumerate(nbr_pores):
        # pore volume term
        Vp[i] = pore_vol[Ps].sum() / 8.0
    
        # intersection of throat sets
        shared = np.array([], dtype=int)
        array1 = all_pore_throats[Ps[0]]
        for p in Ps[1:]:
            array2 = all_pore_throats[p]
            keep, idx1, idx2 = np.intersect1d(array1, array2,
                                             assume_unique=True,
                                             return_indices=True)
            shared = np.concatenate((shared, keep))
            # remove shared values from array 1
            mask1 = np.ones(len(array1), dtype=bool)
            mask1[idx1] = False
            array1 = array1[mask1]
            # remove shared values from array 2
            mask2 = np.ones(len(array2), dtype=bool)
            mask2[idx2] = False
            array2 = array2[mask2]
            # concatenate array 1 and 2
            array1 = np.concatenate((array1, array2))
        
        # throat volume
        Vt[i] = throat_vol[shared].sum() / 4.0
    
    eff_vol[mi_pores] = spacing**3 - Vp - Vt

    # --- Macropore throats ---
    conns = network["throat.conns@macropore"]
    tvol = network["throat.volume@macropore"]

    P1 = conns[:, 0]
    P2 = conns[:, 1]

    np.add.at(eff_vol, P1, 0.5 * tvol)
    np.add.at(eff_vol, P2, 0.5 * tvol)

    eff_vol[mi_pores] *= rho_mi

    return eff_vol
