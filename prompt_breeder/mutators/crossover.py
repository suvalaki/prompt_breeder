from typing import List, Tuple, Any, Dict
from random import random, randint, choices
from itertools import chain
from math import exp
from copy import deepcopy

from prompt_breeder.types import (
    TaskPrompt,
    UnitOfEvolution,
    Population,
)
from prompt_breeder.mutators.base import Mutator
from prompt_breeder.evolution.fitness import FitnessScorer


def softmax(x: List[float]) -> List[float]:
    denom = sum([exp(float(y)) for y in x])
    return [exp(float(y)) / denom for y in x]


# This is to occur after fitness evaluation
class PromptCrossover(Mutator):
    """
    After a mutation operator is applied, with 10% chance a task-prompt is replaced
    with a randomly chosen task-prompt from another member of the population. This
    member is chosen according to fitness proportionate selection. Crossover is not
    applied to mutation-prompts, only to the task-prompts."""

    fitness_scorer: FitnessScorer
    probability_of_replacement: float = 0.1
    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return ["population", "unit"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _get_probability_map(
        self, population: Population, unit: UnitOfEvolution
    ) -> Tuple[List[TaskPrompt], List[float]]:
        members = list(
            chain(*[x.task_prompt_set for x in population.members if x != unit])
        )
        fitnesses = [self.fitness_scorer.score(member) for member in members]
        probabilities = softmax([float(f) for f in fitnesses])
        return members, probabilities

    def mutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        return self.run({"population": population, "unit": unit}, **kwargs)

    def _call(
        self, inputs: Dict[str, Any], run_manager=None, **kwargs
    ) -> Dict[str, UnitOfEvolution]:
        new_unit = deepcopy(inputs["unit"])
        if random() < self.probability_of_replacement:
            idx = randint(0, len(new_unit.task_prompt_set) - 1)
            members, probs = self._get_probability_map(
                inputs["population"], inputs["unit"]
            )
            replacement = deepcopy(choices(members, probs)[0])
            new_unit.task_prompt_set[idx] = replacement
        return {self.output_key: new_unit}
