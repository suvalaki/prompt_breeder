from typing import List, Any, Dict, Callable
from tqdm import tqdm

from langchain.chains.base import Chain

from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.mutators.base import Mutator
from prompt_breeder.evolution.fitness import (
    PopulationFitnessScorer,
)


class EvolutionStep(Chain):
    fitness_scorer: PopulationFitnessScorer
    pre_step_modifiers: List[Mutator]
    mutators: List[Mutator]
    post_step_modifiers: List[Mutator]
    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return ["population"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _pre_step(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        for mutator in self.pre_step_modifiers:
            unit = mutator.mutate(population, unit)
        return unit

    def _post_step(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        for mutator in self.post_step_modifiers:
            unit = mutator.mutate(population, unit)
        return unit


class EvolutionExecutor(Chain):
    """Run the evolution over a number of generations"""

    step: EvolutionStep
    return_intermediate_steps: bool = False
    post_step_callback: Callable[[Population], None] = lambda x, **k: None
    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return ["population", "generations"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any], run_manager=None, **kwargs):
        cb = run_manager.get_child() if run_manager else None

        if self.return_intermediate_steps:
            intermediate_steps = []

        population = inputs["population"]
        for i in tqdm(range(inputs["generations"])):
            population = self.step.run(
                {"population": population}, callbacks=cb, **kwargs
            )

            if self.return_intermediate_steps:
                intermediate_steps += [population]

            self._post_step(population, callback=cb, **kwargs)

        if self.return_intermediate_steps:
            return {self.output_key: intermediate_steps}
        return {self.output_key: population}

    async def _acall(self, inputs: Dict[str, Any], run_manager=None, **kwargs):
        cb = run_manager.get_child() if run_manager else None

        if self.return_intermediate_steps:
            intermediate_steps = []

        population = inputs["population"]
        for i in tqdm(range(inputs["generations"])):
            population = await self.step.arun(
                {"population": population}, callbacks=cb, **kwargs
            )

            if self.return_intermediate_steps:
                intermediate_steps += [population]

            self._post_step(population, callback=cb, **kwargs)

        if self.return_intermediate_steps:
            return {self.output_key: intermediate_steps}
        return {self.output_key: population}

    def _post_step(self, population: Population, **kwargs):
        self.post_step_callback(population, **kwargs)
