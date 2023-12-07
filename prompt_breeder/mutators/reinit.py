from typing import List, Dict, Any
from prompt_breeder.evolution.fitness import PopulationFitnessScorer
from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.mutators.base import Mutator


class ReplaceWithInit(Mutator):
    """Return a new unit with the current elite added to the elites set"""

    fitness_scorer: PopulationFitnessScorer
    value: Any
    initializer: Any
    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return ["unit"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def mutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        return self.run({"unit": unit}, **kwargs)

    def _call(
        self, inputs: Dict[str, UnitOfEvolution], run_manager=None, **kwargs
    ) -> Dict[str, UnitOfEvolution]:
        cb = run_manager.get_child() if run_manager else None
        if (
            self.fitness_scorer.score(
                inputs["unit"].task_prompt_set, callbacks=cb, **kwargs
            )
            <= self.value
        ):
            return {
                self.output_key: self.initializer.initialize(
                    str(inputs["unit"].problem_description),
                    callbacks=cb,
                    **kwargs,
                )
            }

        return {self.output_key: inputs["unit"]}

    async def amutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        return await self.arun({"unit": unit}, **kwargs)
