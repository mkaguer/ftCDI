import jax.numpy as jnp
from jax import lax


def bcc(network,
        spacing,
        rho_sep,
        rho_mi,
):
    """
    
    Work in progress, not currently in use!

    """
    # retrieve volumes
    pore_vol = network["pore.volume"]
    throat_vol = network["throat.volume"]

    # retrieve throat conns
    throat_conns = network["throat.conns"]

    # get Np and Nt
    Np = pore_vol.shape[0]
    Nt = throat_vol.shape[0]
    
    # find micropores
    mask = network["pore.micropore"]
    mi_pores = jnp.arange(Np)[mask]
    
    # get micropore throats
    mi_throats = network["throat.micropore"]
    
    # get P1 and P2
    P1 = throat_conns[:, 0]
    P2 = throat_conns[:, 1]
    
    # organize thoat_conns_mi into [mi, ma]
    throat_conns_mi = throat_conns[mi_throats]
    mask = jnp.isin(throat_conns_mi, mi_pores)
    true_col = jnp.argmax(mask, axis=1)
    rows = jnp.arange(len(mask))
    throat_conns_mi = jnp.stack([throat_conns_mi[rows, true_col], 
                                 throat_conns_mi[rows, 1-true_col]],
                                axis=1)
    
    # get pore volumes to subtract
    Vp = jnp.zeros(Np)
    indices = throat_conns_mi[:, 0]
    values = pore_vol[throat_conns_mi[:, 1]]/8
    Vp = jnp.add.at(Vp, indices, values, inplace=False)
    
    # get throat volumes to subtract
    _, a = jnp.unique(throat_conns_mi[:, 0], return_inverse=True)
    arr = jnp.concatenate((jnp.array([a]).T, throat_conns_mi), axis=1)
    idx = jnp.argsort(arr[:, 0])
    arr = arr[idx, :]
    arr = jnp.sort(arr, axis=0)
    
    # loop through arr to get throat volumes
    Vt = jnp.zeros(Np)
    for i in range(len(mi_pores)):
        mask = arr[:, 0] == i
        ma = arr[:, 2][mask]
        ma_throats = jnp.isin(P1, ma) & jnp.isin(P2, ma)
        idx = jnp.arange(Nt)[ma_throats]
        Vt = Vt.at[i].set(jnp.sum(throat_vol[idx])/4)
        
    # calculate effective volume
    eff_vol = jnp.where(network["pore.micropore"], (spacing**3 - Vp - Vt)*rho_mi, pore_vol)
    
    # add volume of separator throats to separator pores
    sep_throats = network["throat.separator"]
    throat_conns_sep = network["throat.conns"][sep_throats]
    P1 = throat_conns_sep[:, 0]
    P2 = throat_conns_sep[:, 1]
    mask1 = jnp.isin(P1, network["pore.separator"])
    mask2 = jnp.isin(P2, network["pore.separator"])
    eff_vol = jnp.add.at(eff_vol, P1, rho_sep*throat_vol[sep_throats]*mask1,
                         inplace=False)
    eff_vol = jnp.add.at(eff_vol, P2, rho_sep*throat_vol[sep_throats]*mask2,
                         inplace=False)
    
    # subtract 1/2 of pore volume from pores connected to separator
    Ps = jnp.concatenate((P1[~mask1], P2[~mask2]))
    eff_vol.at[Ps].set(1/2*pore_vol[Ps])
    
    # add volume of macropore throats to macropores, 1/2 to each neighbour
    ma_throats = network["throat.macropore"]
    throat_conns_ma = network["throat.conns"][ma_throats]
    P1 = throat_conns_ma[:, 0]
    P2 = throat_conns_ma[:, 1] 
    eff_vol = jnp.add.at(eff_vol, P1, 1/2*throat_vol[ma_throats],
                         inplace=False)
    eff_vol = jnp.add.at(eff_vol, P2, 1/2*throat_vol[ma_throats],
                         inplace=False)
    
    return eff_vol
