from typing import List, Dict
from copy import deepcopy
from prompt_breeder.evolution.fitness import FitnessScorer
from prompt_breeder.types import TaskPrompt, UnitOfEvolution, Population
from prompt_breeder.mutators.base import Mutator


class AddElite(Mutator):
    """Return a new unit with the current elite added to the elites set"""

    fitness_scorer: FitnessScorer
    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return ["unit"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def max_fitness_prompt(
        self, task_prompt_set: List[TaskPrompt], run_manager=None, **kwargs
    ) -> TaskPrompt:
        # sort by fitness
        cb = run_manager.get_child() if run_manager else None
        fitnesses = [
            self.fitness_scorer.score(prompt, callbacks=cb)
            for prompt in task_prompt_set
        ]
        pairs = list(zip(task_prompt_set, fitnesses))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[0][0]

    def mutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        return self.run({"unit": unit}, **kwargs)

    def _call(
        self, inputs: Dict[str, UnitOfEvolution], run_manager=None, **kwargs
    ) -> Dict[str, UnitOfEvolution]:
        unit = deepcopy(inputs["unit"])
        unit.elites += [self.max_fitness_prompt(unit.task_prompt_set)]
        return {self.output_key: unit}
