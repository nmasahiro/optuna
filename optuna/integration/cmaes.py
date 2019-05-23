import math
import numpy
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers import BaseSampler
from optuna.structs import FrozenTrial
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna import types

try:
    import cma
    _available = True
except ImportError as e:
    _import_error = e
    # CmaEsSampler is disabled because cma is not available.
    _available = False


if types.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA


class CmaEsSampler(BaseSampler):
    """
    A Sampler which suggests values based on CMA-ES algorithm.
    We used cma library as an implementation of CMA-ES.
    Note that cma does not support dynamic search space.
    """

    def __init__(self, popsize, sigma0, seed=None):
        # type: (int, float, Optional[int]) -> None

        _check_cma_availability()

        self.popsize = popsize
        self.sigma0 = sigma0
        self.seed = seed

        self.es = None  # type: Optional[cma.CMAEvolutionStrategy]
        self.search_space = None
        self.known_trials = set()  # type: Set[int]
        self.param_names = []
        self.suggested = []
        self.cma_params = {}
        self.rng = numpy.random.RandomState(seed)
        self.logger = optuna.logging.get_logger(__name__)

    def sample_relative(self, study, trial, search_space):
        # type: (RunningStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]

        if len(search_space) == 0:
            # Empty search space.
            return {}

        if self.es is None:
            # initialize CMAEvolutionStrategy.
            self._initialize_cma(search_space)
            self.search_space = search_space
            self.known_trials = set()
            self.cma_params = {}

        if search_space != self.search_space:
            raise ValueError('CMA-ES does not support dynamic search space.')

        self._import_trials(study)

        if len(self.suggested) == 0:
            self.suggested.extend(self.es.ask())

        cur = self.suggested.pop()
        # self.cma_params[trial.number] = cur

        ret_val = {}
        for param_name, value in zip(self.param_names, cur):
            ret_val[param_name] = self._to_optuna_observation(search_space, param_name, value)

        return ret_val

    def _initialize_cma(self, search_space):
        # type: (Dict[str, BaseDistribution]) -> None

        self.param_names = list(sorted(search_space.keys()))
        self.logger.info("CMA-ES handles following parameters: {}".format(self.param_names))

        initial_params = []
        lows = []
        highs = []
        for param_name in self.param_names:
            dist = search_space[param_name]
            if isinstance(dist, CategoricalDistribution):
                # TODO(Yanase): Support Categorical
                raise NotImplementedError()
            if isinstance(dist, UniformDistribution) or \
                    isinstance(dist, LogUniformDistribution):
                lows.append(self._to_cma_observation(search_space, param_name, dist.low))
                highs.append(self._to_cma_observation(search_space, param_name, dist.high))
            elif isinstance(dist, DiscreteUniformDistribution):
                r = dist.high - dist.low
                lows.append(0 - 0.5 * dist.q)
                highs.append(r + 0.5 * dist.q)
            elif isinstance(dist, IntUniformDistribution):
                lows.append(dist.low - 0.5)
                highs.append(dist.high + 0.5)
            else:
                raise ValueError('Incompatible distribution is given: {}.'.format(dist))
            # TODO(Yanase): Set initial_param from outside.
            initial_params.append(self.rng.uniform(lows[-1], highs[-1]))

        cma_option = {
            'BoundaryHandler': cma.BoundTransform,
            'bounds': [lows, highs],
            'popsize': self.popsize
        }
        if self.seed is not None:
            cma_option['seed'] = self.seed

        self.logger.info("Initial parameters for CMA-ES: {}".format(initial_params))
        self.logger.info("Options of CMA-ES: {}".format(cma_option))
        self.es = cma.CMAEvolutionStrategy(initial_params, self.sigma0, cma_option)

    def _import_trials(self, study):
        # type: (RunningStudy) -> None
        matched_trials = []
        for trial in study.trials:
            if trial.state != TrialState.COMPLETE:
                continue
            if trial.distributions != self.search_space:
                continue
            if trial.number in self.known_trials:
                continue
            matched_trials.append(trial)

        for i in range(len(matched_trials) // self.popsize):
            self.es.ask()
            xs = []
            for t in matched_trials[i * self.popsize:(i + 1) * self.popsize]:
                x = []
                if t.number in self.cma_params:
                    xs.append(self.cma_params[t.number])
                else:
                    for param_name in self.param_names:
                        value = t.params[param_name]
                        x.append(self._to_cma_observation(self.search_space, param_name, value))
                    xs.append(x)
                self.known_trials.add(t.number)
            ys = [t.value for t in matched_trials[i * self.popsize:(i + 1) * self.popsize]]
            if study.direction == StudyDirection.MAXIMIZE:
                ys = [-y for y in ys]
            self.es.tell(xs, ys)

    def _to_cma_observation(self, search_space, param_name, optuna_observation):
        # type: (Dict[str, BaseDistribution], str, float) -> float

        dist = search_space[param_name]
        if isinstance(dist, LogUniformDistribution):
            return math.log(optuna_observation)
        if isinstance(dist, DiscreteUniformDistribution):
            return optuna_observation - dist.low
        if isinstance(dist, CategoricalDistribution):
            # TODO(Yanase): Support Categorical
            raise NotImplementedError()
        return optuna_observation

    def _to_optuna_observation(self, search_space, param_name, cma_observation):
        # type: (Dict[str, BaseDistribution], str, float) -> float

        dist = search_space[param_name]
        if isinstance(dist, LogUniformDistribution):
            return math.exp(cma_observation)
        if isinstance(dist, DiscreteUniformDistribution):
            v = numpy.round(cma_observation / dist.q) * dist.q + dist.low
            # v may slightly exceed range due to round-off errors.
            return float(min(max(v, dist.low), dist.high))
        if isinstance(dist, IntUniformDistribution):
            v = numpy.round(cma_observation)
            # v may slightly exceed range due to round-off errors.
            return int(min(max(v, dist.low), dist.high))
        if isinstance(dist, CategoricalDistribution):
            # TODO(Yanase): Support Categorical
            raise NotImplementedError()
        return cma_observation


def _check_cma_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'cma library is not available. Please install cma to use this feature. '
            'cma can be installed by executing `$ pip install cma`. '
            'For further information, please refer to the installation guide of cma. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
