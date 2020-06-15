#
# Root of the pints module.
# Provides access to all shared functionality (optimisation, mcmc, etc.).
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
"""
Pints: Probabilistic Inference on Noisy Time Series.

This module provides several optimisation and sampling methods that can be
applied to find the parameters of a model (typically a time series model) that
are most likely, given an experimental data set.
"""


from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import sys

#
# Version info
#
def _load_version_int():
    try:
        import os
        root = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(root, 'version'), 'r') as f:
            version = f.read().strip().split(',')
        major, minor, revision = [int(x) for x in version]
        return major, minor, revision
    except Exception as e:
        raise RuntimeError('Unable to read version number (' + str(e) + ').')

__version_int__ = _load_version_int()
__version__ = '.'.join([str(x) for x in __version_int__])
if sys.version_info[0] < 3:
    del(x)  # Before Python3, list comprehension iterators leaked

#
# Expose pints version
#
def version(formatted=False):
    """
    Returns the version number, as a 3-part integer (major, minor, revision).
    If ``formatted=True``, it returns a string formatted version (for example
    "Pints 1.0.0").
    """
    if formatted:
        return 'Pints ' + __version__
    else:
        return __version_int__


#
# Constants
#
# Float format: a float can be converted to a 17 digit decimal and back without
# loss of information
FLOAT_FORMAT = '{: .17e}'

#
# Core classes
#
from ._core import ForwardModel, ForwardModelS1
from ._core import TunableMethod
from ._core import SingleOutputProblem, MultiOutputProblem

#
# Utility classes and methods
#
from ._util import strfloat, vector, matrix2d
from ._util import Timer
from ._logger import Logger, Loggable

#
# Logs of probability density functions (not necessarily normalised)
#
from ._log_pdfs import (
    LogPDF,
    LogPrior,
    LogPosterior,
    ProblemLogLikelihood,
    SumOfIndependentLogPDFs,
)

#
# Log-priors
#
from ._log_priors import (
    BetaLogPrior,
    CauchyLogPrior,
    ComposedLogPrior,
    ExponentialLogPrior,
    GammaLogPrior,
    GaussianLogPrior,
    HalfCauchyLogPrior,
    InverseGammaLogPrior,
    LogNormalLogPrior,
    MultivariateGaussianLogPrior,
    NormalLogPrior,
    StudentTLogPrior,
    UniformLogPrior,
)

#
# Log-likelihoods
#
from ._log_likelihoods import (
    AR1LogLikelihood,
    ARMA11LogLikelihood,
    CauchyLogLikelihood,
    GaussianIntegratedUniformLogLikelihood,
    GaussianKnownSigmaLogLikelihood,
    GaussianLogLikelihood,
    KnownNoiseLogLikelihood,
    MultiplicativeGaussianLogLikelihood,
    ScaledLogLikelihood,
    StudentTLogLikelihood,
    UnknownNoiseLogLikelihood,
)

#
# Boundaries
#
from ._boundaries import (
    Boundaries,
    LogPDFBoundaries,
    RectangularBoundaries,
)

#
# Error measures
#
from ._error_measures import (
    ErrorMeasure,
    ProblemErrorMeasure,
    ProbabilityBasedError,
    SumOfErrors,
    MeanSquaredError,
    RootMeanSquaredError,
    SumOfSquaresError,
)

#
# Parallel function evaluation
#
from ._evaluation import (
    evaluate,
    Evaluator,
    ParallelEvaluator,
    SequentialEvaluator,
)


#
# Optimisation
#
from ._optimisers import (
    curve_fit,
    fmin,
    Optimisation,
    OptimisationController,
    optimise,
    Optimiser,
    PopulationBasedOptimiser,
    TriangleWaveTransform,
)
from ._optimisers._cmaes import CMAES
from ._optimisers._nelder_mead import NelderMead
from ._optimisers._pso import PSO
from ._optimisers._snes import SNES
from ._optimisers._xnes import XNES


#
# Diagnostics
#
from ._diagnostics import (
    effective_sample_size,
    rhat,
    rhat_all_params,
)


#
#  MCMC
#
from ._mcmc import (
    mcmc_sample,
    MCMCController,
    MCMCSampler,
    MCMCSampling,
    MultiChainMCMC,
    SingleChainMCMC,
)
from ._mcmc._adaptive_covariance import AdaptiveCovarianceMC
from ._mcmc._differential_evolution import DifferentialEvolutionMCMC
from ._mcmc._dream import DreamMCMC
from ._mcmc._emcee_hammer import EmceeHammerMCMC
from ._mcmc._haario_ac import HaarioACMC
from ._mcmc._haario_bardenet_ac import HaarioBardenetACMC
from ._mcmc._haario_bardenet_ac import AdaptiveCovarianceMCMC
from ._mcmc._hamiltonian import HamiltonianMCMC
from ._mcmc._mala import MALAMCMC
from ._mcmc._metropolis import MetropolisRandomWalkMCMC
from ._mcmc._monomial_gamma_hamiltonian import MonomialGammaHamiltonianMCMC
from ._mcmc._population import PopulationMCMC
from ._mcmc._rao_blackwell_ac import RaoBlackwellACMC
from ._mcmc._relativistic import RelativisticMCMC
from ._mcmc._slice_stepout import SliceStepoutMCMC
from ._mcmc._slice_doubling import SliceDoublingMCMC
from ._mcmc._summary import MCMCSummary


#
# Nested samplers
#
from ._nested import NestedSampler
from ._nested import NestedController
from ._nested._rejection import NestedRejectionSampler
from ._nested._ellipsoid import NestedEllipsoidSampler


#
# Noise generators (always import!)
#
from . import noise

#
# Remove any imported modules, so we don't expose them as part of pints
#
del(sys)
