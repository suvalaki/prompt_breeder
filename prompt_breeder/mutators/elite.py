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
        if str(unit) not in [str(x) for x in unit.elites]:
            best_prompt = self.max_fitness_prompt(
                unit.task_prompt_set, run_manager=run_manager
            )
            best_fit = self.fitness_scorer.score(
                best_prompt, callbacks=run_manager.get_child() if run_manager else None
            )

            if len(unit.elites) > 0:
                best_prompt_old = self.max_fitness_prompt(
                    unit.elites, run_manager=run_manager
                )
                best_fit_old = self.fitness_scorer.score(
                    best_prompt_old,
                    callbacks=run_manager.get_child() if run_manager else None,
                )
                if best_fit >= best_fit_old:
                    unit.elites += [best_prompt]
            else:
                unit.elites += [best_prompt]
        return {self.output_key: unit}
