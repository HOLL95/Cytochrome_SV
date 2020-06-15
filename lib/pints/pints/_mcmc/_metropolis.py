#
# Random-walk Metropolis MCMC
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np


class MetropolisRandomWalkMCMC(pints.SingleChainMCMC):
    """
    Metropolis Random Walk MCMC, as described in [1]_.

    Standard Metropolis using multivariate Gaussian distribution as proposal
    step, also known as Metropolis Random Walk MCMC.

    Extends :class:`SingleChainMCMC`.

    References
    ----------
    .. [1] "Equation of state calculations by fast computing machines".
           Metropolis, N., Rosenbluth, A.W., Rosenbluth, M.N., Teller, A.H. and
           Teller, E. (1953) The journal of chemical physics, 21(6),
           pp.1087-1092
           https://doi.org/10.1063/1.1699114
    """

    def __init__(self, x0, sigma0=None):
        super(MetropolisRandomWalkMCMC, self).__init__(x0, sigma0)

        # Set initial state
        self._running = False

        # Current point and proposed point
        self._current = None
        self._current_log_pdf = None
        self._proposed = None

    def acceptance_rate(self):
        """
        Returns the current (measured) acceptance rate.
        """
        return self._acceptance

    def ask(self):
        """ See :meth:`SingleChainMCMC.ask()`. """
        # Initialise on first call
        if not self._running:
            self._initialise()

        # Propose new point
        if self._proposed is None:
            # Note: Gaussian distribution is symmetric
            #  N(x|y, sigma) = N(y|x, sigma) so that we can drop the proposal
            #  distribution term from the acceptance criterion
            self._proposed = np.random.multivariate_normal(
                self._current, self._sigma0)

            # Set as read-only
            self._proposed.setflags(write=False)

        # Return proposed point
        return self._proposed

    def current_log_pdf(self):
        """ See :meth:`SingleChainMCMC.current_log_pdf()`. """
        return self._current_log_pdf

    def _initialise(self):
        """
        Initialises the routine before the first iteration.
        """
        if self._running:
            raise RuntimeError('Already initialised.')

        # Propose x0 as first point
        self._current = None
        self._current_log_pdf = None
        self._proposed = self._x0

        # Acceptance rate monitoring
        self._iterations = 0
        self._acceptance = 0

        # Update sampler state
        self._running = True

    def _log_init(self, logger):
        """ See :meth:`Loggable._log_init()`. """
        logger.add_float('Accept.')

    def _log_write(self, logger):
        """ See :meth:`Loggable._log_write()`. """
        logger.log(self._acceptance)

    def name(self):
        """ See :meth:`pints.MCMCSampler.name()`. """
        return 'Metropolis random walk MCMC'

    def tell(self, fx):
        """ See :meth:`pints.SingleChainMCMC.tell()`. """
        # Check if we had a proposal
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')

        # Ensure fx is a float
        fx = float(fx)

        # First point?
        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError(
                    'Initial point for MCMC must have finite log_pdf.')

            # Accept
            self._current = self._proposed
            self._current_log_pdf = fx

            # Increase iteration count
            self._iterations += 1

            # Clear proposal
            self._proposed = None

            # Return first point for chain
            return self._current

        # Check if the proposed point can be accepted
        accepted = 0
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < fx - self._current_log_pdf:
                accepted = 1
                self._current = self._proposed
                self._current_log_pdf = fx

        # Clear proposal
        self._proposed = None

        # Update acceptance rate (only used for output!)
        self._acceptance = ((self._iterations * self._acceptance + accepted) /
                            (self._iterations + 1))

        # Increase iteration count
        self._iterations += 1

        # Return new point for chain
        return self._current

    def replace(self, current=None, current_log_pdf=None, proposed=None):
        """ See :meth:`pints.SingleChainMCMC.replace()`. """

        # At least one round of ask-and-tell must have been run
        if (not self._running) or self._current_log_pdf is None:
            raise RuntimeError(
                'Replace can only be used when already running.')

        # Check values
        current = pints.vector(current)
        if not len(current) == self._n_parameters:
            raise ValueError('Point `current` has the wrong dimensions.')
        current_log_pdf = float(current_log_pdf)
        if proposed is not None:
            proposed = pints.vector(proposed)
            if not len(proposed) == self._n_parameters:
                raise ValueError('Point `proposed` has the wrong dimensions.')

        # Store
        self._current = current
        self._current_log_pdf = current_log_pdf
        self._proposed = proposed

