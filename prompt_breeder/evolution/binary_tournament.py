from typing import Tuple, List, Dict
import asyncio
import tqdm
import tqdm.asyncio
import random
from copy import deepcopy

from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.mutators.base import Mutator
from prompt_breeder.evolution.base import EvolutionStep


class BinaryEvolution(EvolutionStep):
    def _assign_pairs(self, population: Population) -> List[Tuple[int, int]]:
        """Get the index for pairs which will complete against one another"""
        members_idx = list(range(len(population.members)))
        elements = 2 * (len(members_idx) // 2)
        random_members = random.sample(members_idx[:elements], k=elements)
        pairs = list(zip(random_members[0::2], random_members[1::2]))
        return pairs

    def _get_mutator(self) -> Mutator:
        mutator = random.choice(self.mutators)
        print("MUTATOR")
        print(type(mutator))
        return mutator

    def _mutate(
        self, mutator: Mutator, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        mutant = mutator.mutate(population, unit, **kwargs)
        return mutant

    async def _amutate(
        self, mutator: Mutator, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        mutant = await mutator.amutate(population, unit, **kwargs)
        return mutant

    def _single_match(
        self,
        population: Population,
        unit0: UnitOfEvolution,
        unit1: UnitOfEvolution,
        **kwargs
    ) -> Tuple[UnitOfEvolution, UnitOfEvolution]:
        """Have the two units compare fitness and mutate the loser"""

        unit0_new = deepcopy(unit0)
        unit1_new = deepcopy(unit1)

        unit0 = self._pre_step(population, unit0_new, **kwargs)
        unit1 = self._pre_step(population, unit1_new, **kwargs)

        fitness0 = self.fitness_scorer.score(unit0_new.task_prompt_set, **kwargs)
        fitness1 = self.fitness_scorer.score(unit1_new.task_prompt_set, **kwargs)

        if fitness0 < fitness1:
            unit0_new = self.mutate(population, unit1_new, **kwargs).mutant
            unit0_new = self._post_step(population, unit0_new, **kwargs)
            unit1_new = self._post_step(population, unit1_new, **kwargs)
        else:
            unit1_new = self.mutate(population, unit0_new, **kwargs).mutant
            unit0_new = self._post_step(population, unit0_new, **kwargs)
            unit1_new = self._post_step(population, unit1_new, **kwargs)

        return unit0_new, unit1_new

    async def _asingle_match(
        self,
        population: Population,
        unit0: UnitOfEvolution,
        unit1: UnitOfEvolution,
        **kwargs
    ) -> Tuple[UnitOfEvolution, UnitOfEvolution]:
        """Have the two units compare fitness and mutate the loser"""

        unit0_new = deepcopy(unit0)
        unit1_new = deepcopy(unit1)

        unit0 = self._pre_step(population, unit0_new, **kwargs)
        unit1 = self._pre_step(population, unit1_new, **kwargs)

        fitness0 = await self.fitness_scorer.ascore(unit0_new.task_prompt_set, **kwargs)
        fitness1 = await self.fitness_scorer.ascore(unit1_new.task_prompt_set, **kwargs)

        if fitness0 < fitness1:
            unit0_new = (await self.amutate(population, unit1_new, **kwargs)).mutant
            unit0_new = self._post_step(population, unit0_new, **kwargs)
        else:
            unit1_new = (await self.amutate(population, unit0_new, **kwargs)).mutant
            unit1_new = self._post_step(population, unit1_new, **kwargs)

        return unit0_new, unit1_new

    def _call(self, inputs: Dict[str, Population], run_manager=None, **kwargs):
        cb = run_manager.get_child() if run_manager else None
        population = deepcopy(inputs["population"])
        pairs = self._assign_pairs(population)

        for i, j in tqdm.tqdm(pairs, leave=False, position=1):
            population.members[i], population.members[j] = self._single_match(
                inputs["population"],
                population.members[i],
                population.members[j],
                callbacks=cb,
                **kwargs
            )

        return {self.output_key: population}

    async def _inplace_single_match(
        self,
        i: int,
        j: int,
        inputs: Dict[str, Population],
        population: Population,
        run_manager=None,
        **kwargs
    ):
        cb = run_manager.get_child() if run_manager else None
        population.members[i], population.members[j] = await self._asingle_match(
            inputs["population"],
            population.members[i],
            population.members[j],
            callbacks=cb,
            **kwargs
        )

    async def _acall(self, inputs: Dict[str, Population], run_manager=None, **kwargs):
        population = deepcopy(inputs["population"])
        pairs = self._assign_pairs(population)
        await asyncio.gather(
            *[
                self._inplace_single_match(
                    i, j, inputs, population, run_manager, **kwargs
                )
                for i, j in pairs
            ]
        )
        return {self.output_key: population}
