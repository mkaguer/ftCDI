from openpnm.algorithms import TransientReactiveTransport
import scipy as sp

__all__ = ['CustomTransientReactiveTransport']


class CustomTransientReactiveTransportSettings:
    r"""

    Parameters
    ----------
    %(ReactiveTransportSettings.parameters)s

    """
    pore_volume = 'pore.volume'


class CustomTransientReactiveTransport(TransientReactiveTransport):
    r"""
    A subclass of TransientReactiveTransport for custom simulations.

    Parameters
    ----------
    network : Network
        The Network with which this algorithm is associated.

    Notes
    -----
    Either a Network or a Project must be specified.

    """

    def __init__(self, phase, name='cust_trans_react_?', **kwargs):
        super().__init__(phase=phase, name=name, **kwargs)
        self.settings._update(CustomTransientReactiveTransportSettings())

    def _build_A(self):
        """
        Builds the coefficient matrix based on throat conductance values.

        Notes
        -----
        The conductance to use is specified in stored in the algorithm's
        settings under ``alg.settings['conductance']``.

        """
        gvals = self.settings['conductance']
        if gvals in self.iterative_props:
            self.settings.cache = False
        if not self.settings['cache']:
            self._pure_A = None
        if self._pure_A is None:
            phase = self.project[self.settings.phase]
            g = phase[gvals]
            am = self.network.create_adjacency_matrix(weights=g, fmt='csr')
            # build laplacian
            deg = am.sum(axis=1).A.ravel()
            D = sp.sparse.diags(deg, format='csr')
            self._pure_A = (D - am).astype(float)
        self.A = self._pure_A.copy()
        
    def _apply_BCs(self):
        
        return None
