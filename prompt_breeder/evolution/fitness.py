from __future__ import annotations
import asyncio
from typing import List, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict

from prompt_breeder.types import TaskPrompt


@runtime_checkable
class Fitness(Protocol):
    def __lt__(self, rhs: Fitness) -> bool:
        pass

    def __le__(self, rhs: Fitness) -> bool:
        pass

    def __float__(self):
        pass


@runtime_checkable
class FitnessScorer(Protocol):
    def score(self, prompt: TaskPrompt, **kwargs) -> Fitness:
        pass

    async def ascore(self, prompt: TaskPrompt, **kwargs) -> Fitness:
        pass


class PopulationFitnessScorer(ABC, BaseModel):
    scorer: FitnessScorer
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def score(self, population: List[TaskPrompt], **kwargs) -> Fitness:
        pass

    @abstractmethod
    async def ascore(self, population: List[TaskPrompt], **kwargs) -> Fitness:
        pass


class WorstMemberFitness(PopulationFitnessScorer):
    def score(self, population: List[TaskPrompt], **kwargs) -> Fitness:
        return min([self.scorer.score(member, **kwargs) for member in population])

    async def ascore(self, population: List[TaskPrompt], **kwargs) -> Fitness:
        return min(
            await asyncio.gather(
                *[self.scorer.ascore(member, **kwargs) for member in population]
            )
        )


class BestMemberFitness(PopulationFitnessScorer):
    def score(self, population: List[TaskPrompt], **kwargs) -> Fitness:
        return max([self.scorer.score(member, **kwargs) for member in population])

    async def ascore(self, population: List[TaskPrompt], **kwargs) -> Fitness:
        return max(
            await asyncio.gather(
                *[self.scorer.ascore(member, **kwargs) for member in population]
            )
        )
