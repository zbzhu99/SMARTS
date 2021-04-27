from logging import Logger
from typing import List, Sequence

from checker import Checker, Result


class CheckerConfig:
    def __init__(self, checker, end_early=True) -> None:
        self.checker: Checker = checker
        self.end_early = end_early


class CheckerHost:
    def __init__(self, sim, logger: Logger) -> None:
        self.checker_configs: List[CheckerConfig] = []
        self.sim = sim
        self._has_checker = False
        self._logger = logger

    @property
    def done(self):
        return self._has_checker and len(self.checker_configs) < 1

    def add_checkers(self, *checker_configs: CheckerConfig):
        self.checker_configs.extend(checker_configs)
        self._has_checker = True

    def record_step(self, observations, rewards, dones, infos) -> List[Result]:
        results = []
        ccs = list(enumerate(self.checker_configs))
        for i, checker_config in reversed(ccs):
            result = checker_config.checker.evaluate(
                self.sim, observations, rewards, dones, infos
            )

            if checker_config.end_early and result.checker_success != Result.TBD:
                self.checker_configs.pop(i)

            results.append(result)
        print(results)
        self._logger.info(results)

        return results

    def conclude(self):
        results = []
        for checker_config in self.checker_configs:
            result = checker_config.checker.timeout()
            results.append(result)
        self._logger.info(results)

        self.checker_configs.clear()
        self._has_checker = False
