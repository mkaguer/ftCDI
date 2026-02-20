import jax.numpy as jnp
import numpy as np

__all__ = ["numpy_to_jax",
           "jax_to_numpy"]

def numpy_to_jax(network):
    
    # create new network
    network_new = {}
    # move data from old network to new network
    for key in network.keys():
        network_new[key] = jnp.asarray(network[key])
        
    return network_new


def jax_to_numpy(network):
    
    # create new network
    network_new = {}
    # move data from old network to new network
    for key in network.keys():
        network_new[key] = np.asarray(network[key])
        
    return network_new

