#
# Toy base classes.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import pints
from scipy.integrate import odeint


class ToyLogPDF(pints.LogPDF):
    """
    Abstract base class for toy distributions.

    Extends :class:`pints.LogPDF`.
    """

    def distance(self, samples):
        """
        Calculates a measure of distance from ``samples`` to some
        characteristic of the underlying distribution.
        """
        raise NotImplementedError

    def sample(self, n_samples):
        """
        Generates independent samples from the underlying distribution.
        """
        raise NotImplementedError

    def suggested_bounds(self):
        """
        Returns suggested boundaries for prior.
        """
        raise NotImplementedError


class ToyModel(object):
    """
    Defines an interface for toy problems.

    Note that toy models should extend both ``ToyModel`` and one of the forward
    model classes, e.g. :class:`pints.ForwardModel`.
    """
    def suggested_parameters(self):
        """
        Returns an numpy array of the parameter values that are representative
        of the model.

        For example, these parameters might reproduce a particular result that
        the model is famous for.
        """
        raise NotImplementedError

    def suggested_times(self):
        """
        Returns an numpy array of time points that is representative of the
        model
        """
        raise NotImplementedError


class ToyODEModel(ToyModel):
    """
    Defines an interface for toy problems where the underlying model is an
    ordinary differential equation (ODE) that describes some time-series
    generating model.

    Note that toy ODE models should extend both :class:`pints.ToyODEModel` and
    one of the forward model classes, e.g. :class:`pints.ForwardModel` or
    :class:`pints.ForwardModelS1`.

    To use this class as the basis for a :class:`pints.ForwardModel`, the
    method :meth:`_rhs()` should be reimplemented.

    Models implementing :meth:`_rhs()`, :meth:`jacobian()` and :meth:`_dfdp()`
    can be used to create a :class:`pints.ForwardModelS1`.
    """
    def _dfdp(self, y, t, p):
        """
        Returns the derivative of the ODE RHS at time ``t``, with respect to
        model parameters ``p``.

        Parameters
        ----------
        y
            The state vector at time ``t`` (with length ``n_outputs``).
        t
            The time to evaluate at (as a scalar).
        p
            A vector of model parameters (of length ``n_parameters``).

        Returns
        -------
        A matrix of dimensions ``n_parameters`` by ``n_parameters``.
        """
        raise NotImplementedError

    def jacobian(self, y, t, p):
        """
        Returns the Jacobian (the derivative of the RHS ODE with respect to the
        outputs) at time ``t``.

        Parameters
        ----------
        y
            The state vector at time ``t`` (with length ``n_outputs``).
        t
            The time to evaluate at (as a scalar).
        p
            A vector of model parameters (of length ``n_parameters``).

        Returns
        -------
        A matrix of dimensions ``n_outputs`` by ``n_outputs``.
        """
        raise NotImplementedError

    def _rhs(self, y, t, p):
        """
        Returns the evaluated RHS (``dy/dt``) for a given state vector ``y``,
        time ``t``, and parameter vector ``p``.

        Parameters
        ----------
        y
            The state vector at time ``t`` (with length ``n_outputs``).
        t
            The time to evaluate at (as a scalar).
        p
            A vector of model parameters (of length ``n_parameters``).

        Returns
        -------
        A vector of length ``n_outputs``.
        """
        raise NotImplementedError

    def _rhs_S1(self, y_and_dydp, t, p):
        """
        Forms the RHS of ODE for numerical integration to obtain both outputs
        and sensitivities.

        Parameters
        ----------
        y_and_dydp
            A combined vector of states (elements ``0`` to ``n_outputs - 1``)
            and sensitivities (elements ``n_outputs`` onwards).
        t
            The time to evaluate at (as a scalar).
        p
            A vector of model parameters (of length ``n_parameters``).

        Returns
        -------
        A vector of length ``n_outputs + n_parameters``.
        """
        y = y_and_dydp[0:self.n_outputs()]
        dydp = y_and_dydp[self.n_outputs():].reshape((self.n_parameters(),
                                                      self.n_outputs()))
        dydt = self._rhs(y, t, p)
        d_dydp_dt = (
            np.matmul(dydp, np.transpose(self.jacobian(y, t, p))) +
            np.transpose(self._dfdp(y, t, p)))
        return np.concatenate((dydt, d_dydp_dt.reshape(-1)))

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        return self._simulate(parameters, times, False)

    def _simulate(self, parameters, times, sensitivities):
        """
        Private helper function that uses ``scipy.integrate.odeint`` to
        simulate a model (with or without sensitivities).

        Parameters
        ----------
        parameters
            With dimensions ``n_parameters``.
        times
            The times at which to calculate the model output / sensitivities.
        sensitivities
            If set to ``True`` the function returns the model outputs and
            sensitivities ``(values, sensitivities)``. If set to ``False`` the
            function only returns the model outputs ``values``. See
            :meth:`pints.ForwardModel.simulate()` and
            :meth:`pints.ForwardModel.simulate_with_sensitivities()` for
            details.
        """
        times = pints.vector(times)
        if np.any(times < 0):
            raise ValueError('Negative times are not allowed.')

        # Scipy odeint requires the first element in ``times`` to be the
        # initial point, which ForwardModel says _has to be_ t=0
        offset = 0
        if len(times) < 1 or times[0] != 0:
            times = np.concatenate(([0], times))
            offset = 1

        if sensitivities:
            n_params = self.n_parameters()
            n_outputs = self.n_outputs()
            y0 = np.zeros(n_params * n_outputs + n_outputs)
            y0[0:n_outputs] = self._y0
            result = odeint(self._rhs_S1, y0, times, (parameters,))
            values = result[:, 0:n_outputs]
            dvalues_dp = (result[:, n_outputs:].reshape(
                (len(times), n_outputs, n_params), order="F"))
            return values[offset:], dvalues_dp[offset:]
        else:
            values = odeint(self._rhs, self._y0, times, (parameters,))
            return values[offset:]

    def simulateS1(self, parameters, times):
        """ See :meth:`pints.ForwardModelS1.simulateS1()`. """
        return self._simulate(parameters, times, True)
