from typing import List, Any, Dict, Callable
from tqdm import tqdm
from abc import abstractmethod

from pydantic import BaseModel, ConfigDict
from langchain.chains.base import Chain

from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.mutators.base import Mutator
from prompt_breeder.evolution.fitness import (
    PopulationFitnessScorer,
)


class EvolutionTransition(BaseModel):
    unit: UnitOfEvolution
    mutator: str | None
    mutant: UnitOfEvolution

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class EvolutionStep(Chain):
    fitness_scorer: PopulationFitnessScorer
    pre_step_modifiers: List[Mutator]
    mutators: List[Mutator]
    post_step_modifiers: List[Mutator]
    post_mutation_callback: None | Callable[[EvolutionTransition], None] = None
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

    @abstractmethod
    def _get_mutator(self) -> Mutator:
        pass

    def mutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> EvolutionTransition:
        mutator = self._get_mutator()
        return EvolutionTransition(
            unit=unit,
            mutator=mutator.__class__.__name__,
            mutant=self._mutate(mutator, population, unit, **kwargs),
        )

    async def amutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> EvolutionTransition:
        mutator = self._get_mutator()
        return EvolutionTransition(
            unit=unit,
            mutator=mutator.__class__.__name__,
            mutant=await self._amutate(mutator, population, unit, **kwargs),
        )

    @abstractmethod
    def _mutate(
        self, mutator: Mutator, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        ...

    @abstractmethod
    async def _amutate(
        self, mutator: Mutator, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        ...


class EvolutionExecutor(Chain):
    """Run the evolution over a number of generations"""

    step: EvolutionStep
    return_intermediate_steps: bool = False
    post_step_callback: List[Callable[[Population], None]] | Callable[
        [Population], None
    ] | None = None
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
        if not self.post_step_callback:
            return
        if isinstance(self.post_step_callback, list):
            for cb in self.post_step_callback:
                cb(population, **kwargs)
        else:
            self.post_step_callback(population, **kwargs)
