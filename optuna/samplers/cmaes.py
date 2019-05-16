import cma
import math
import numpy
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers import BaseSampler
from optuna.structs import FrozenTrial
from optuna import types

if types.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA


class CMASampler(BaseSampler):
    """
    A Sampler which suggests values based on CMA-ES algorithm.
    We used cma library as an implementation of CMA-ES.
    Note that cma does not support dynamic search space.
    """

    def __init__(self, popsize, sigma0, import_trials=False, seed=None):
        # type: (int, float, bool, Optional[int]) -> None

        self.popsize = popsize
        self.sigma0 = sigma0
        self.logger = optuna.logging.get_logger(__name__)
        self.import_trials = import_trials
        self.es = None  # type: Optional[cma.CMAEvolutionStrategy]
        self.param_names = []
        self.completed = {}
        self.suggested = []
        self.running = {}
        self.rng = numpy.random.RandomState(seed)

    def sample_relative(self, study, trial, search_space):
        # type: (RunningStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]

        if len(search_space) == 0:
            # Empty search space.
            return {}

        moved = []
        for tid in self.running.keys():
            t = study.storage.get_trial(tid)
            if t.state.is_finished():
                self.completed[tid] = self.running[tid]
                moved.append(tid)

        for tid in moved:
            del self.running[tid]

        if self.es is None:
            # initialize CMAEvolutionStrategy.

            lows = []
            highs = []

            self.param_names = list(search_space.keys())
            self.param_names.sort()
            initial_params = []
            self.logger.info(self.param_names)

            for param_name in self.param_names:
                dist = search_space[param_name]
                if isinstance(dist, UniformDistribution) or \
                        isinstance(dist, DiscreteUniformDistribution) or \
                        isinstance(dist, IntUniformDistribution):
                    lows.append(dist.low)
                    highs.append(dist.high)
                elif isinstance(dist, LogUniformDistribution):
                    lows.append(math.log(dist.low))
                    highs.append(math.log(dist.high))
                else:
                    # TODO(Yanase): Support Categorical
                    raise NotImplementedError()
                # TODO(Yanase): Set initial_param from outside.
                initial_params.append(self.rng.uniform(lows[-1], highs[-1]))

            cma_option = {
                'BoundaryHandler': cma.BoundTransform,
                'bounds': [lows, highs],
                'popsize': self.popsize
            }

            self.logger.info("Options of CMA-ES: {}".format(cma_option))
            self.es = cma.CMAEvolutionStrategy(initial_params, self.sigma0, cma_option)

            if self.import_trials:
                matched_trials = []
                for trial in study.trials:
                    if trial.distributions == search_space:
                        matched_trials.append(trial)

                matched_trials.sort(key=lambda x: x.value, reverse=True)
                rest = len(matched_trials) % self.popsize
                matched_trials = matched_trials[rest:]
                assert len(matched_trials) % self.popsize == 0

                for i in range(len(matched_trials) // self.popsize):
                    _ = self.es.ask()
                    xs = []
                    for t in matched_trials[i * self.popsize:(i + 1) * self.popsize]:
                        x = []
                        for param_name in self.param_names:
                            value = t.params[param_name]
                            dist = search_space[param_name]
                            if isinstance(dist, UniformDistribution) or \
                                    isinstance(dist, DiscreteUniformDistribution) or \
                                    isinstance(dist, IntUniformDistribution):
                                x.append(value)
                            elif isinstance(dist, LogUniformDistribution):
                                x.append(math.log(value))
                            else:
                                # TODO(Yanase): Support Categorical
                                raise NotImplementedError()
                        xs.append(x)
                    ys = [t.value for t in matched_trials[i * self.popsize:(i + 1) * self.popsize]]
                    self.es.tell(xs, ys)
                pass

        if len(self.completed) >= self.popsize:
            solutions = []
            values = []
            for i, cma_params in self.completed.items():
                t = study.storage.get_trial(i)
                solutions.append(cma_params)
                values.append(t.value)
            self.es.tell(solutions, values)
            self.completed.clear()

        if len(self.suggested) == 0:
            self.suggested.extend(self.es.ask())

        cur = self.suggested.pop()
        self.running[trial.trial_id] = cur

        ret_val = {}
        for param_name, value in zip(self.param_names, cur):
            if isinstance(search_space[param_name], LogUniformDistribution):
                value = math.exp(value)
            ret_val[param_name] = value

        return ret_val
