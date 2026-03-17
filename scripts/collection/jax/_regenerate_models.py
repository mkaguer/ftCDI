import pnmlib as pnm
import collection.jax as co

def regenerate_models(proj, phase):
    """
    this is a custom 'regenerate_models' for CDI_simulation, models are
    regenerated in the order they were applied!
    """
    # get network
    network = proj["network"]
    # regenerate geometry models
    pnm.models.apply_models(network,
                            models=co.geometry.macropore.spheres_and_cylinders,
                            domain="macropore")
    pnm.models.apply_models(network,
                            models=co.geometry.micropore.spheres_and_cylinders,
                            domain="micropore")
    pnm.models.apply_models(network,
                            models=co.geometry.separator.continuum,
                            domain="separator")
    pnm.models.apply_models(network,
                            models=co.geometry.macropore.intersecting_cylinders,
                            domain="perforated")
    # regenerate phase models
    pnm.models.apply_models(phase,
                            models=co.phase.macropore,
                            domain="macropore")
    pnm.models.apply_models(phase,
                            models=co.phase.micropore,
                            domain="micropore")
    pnm.models.apply_models(phase,
                            models=co.phase.separator,
                            domain="separator")
    pnm.models.apply_models(phase,
                            models=co.phase.macropore,
                            domain="perforated")
    # regenerate physics models
    pnm.models.apply_models(phase,
                            models=co.physics.perforated,
                            domain="perforated")
    pnm.models.apply_models(phase,
                            models=co.physics.macropore,
                            domain="macropore")
    pnm.models.apply_models(phase,
                            models=co.physics.micropore,
                            domain="micropore")
    pnm.models.apply_models(phase,
                            models=co.physics.separator,
                            domain="separator")
    # regenerate phi_d and c_mi properties
    pnm.models.apply_models(phase,
                            models=co.physics.properties,
                            domain="micropore")
    # add source term models
    pnm.models.apply_models(phase,
                            models=co.physics.sources,
                            domain="micropore")


