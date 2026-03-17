import jax.numpy as jnp

__all__ = ["apply_models"]


def apply_models(network, models, domain=None):
    
    propnames = models.keys()
    for propname in propnames:
        model = models[propname]["model"]
        args = {k: v for k, v in models[propname].items() if k != "model"}
        if domain is None:
            network[propname] = model(network, **args)
        else:
            idx = propname.index(".")
            element = propname[0:idx]
            mask = network[element + "." + domain]
            values = model(network, **args)
            try:
                if jnp.array(values).ndim == 2:
                    mask = mask[:, jnp.newaxis]
            except:
                # for dicts
                network[propname] = values
                continue
            # FIXME: line below causes a tracer
            # prop = network.get(propname, jnp.full_like(values, -1))
            if propname in network:
                prop = network[propname]
            else:
                prop = jnp.full_like(values, -1)
            values = jnp.where(mask, values, prop)
            network[propname] = values
