import asyncio
from typing import List, Any, Dict, Callable
from langchain.chains.base import Chain
from prompt_breeder.types import ProblemDescription, UnitOfEvolution, Population
from prompt_breeder.evolution.fitness import PopulationFitnessScorer
from tqdm import tqdm


class UnitInitialization(Chain):
    problem_description_factory: Callable[[str], ProblemDescription]
    n_members_per_unit: int
    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return ["problem_description"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def initialize(self, problem_description: str, **kwargs) -> UnitOfEvolution:
        return self.run({"problem_description": problem_description}, **kwargs)

    async def ainitialize(self, problem_description: str, **kwargs) -> UnitOfEvolution:
        return await self.arun({"problem_description": problem_description}, **kwargs)


class PopulationInitialization(Chain):
    initializer: UnitInitialization
    n_units: int
    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return ["problem_description"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def initialize(self, problem_description: str, **kwargs) -> Population:
        return self.run({"problem_description": problem_description}, **kwargs)

    async def ainitialize(self, problem_description: str, **kwargs) -> Population:
        return await self.arun({"problem_description": problem_description}, **kwargs)

    def _call(self, inputs: Dict[str, Any], run_manager=None, **kwargs):
        return {
            self.output_key: Population(
                members=[
                    self.initializer.initialize(
                        inputs["problem_description"],
                        callbacks=run_manager.get_child() if run_manager else None,
                        **kwargs
                    )
                    for i in range(self.n_units)
                ]
            )
        }

    async def _acall(self, inputs: Dict[str, Any], run_manager=None, **kwargs):
        return {
            self.output_key: Population(
                members=await asyncio.gather(
                    *[
                        self.initializer.ainitialize(
                            inputs["problem_description"],
                            callbacks=run_manager.get_child() if run_manager else None,
                            **kwargs
                        )
                        for i in range(self.n_units)
                    ]
                )
            )
        }


class PositivePopulationInitialization(Chain):
    initializer: UnitInitialization
    n_units: int
    output_key: str = "output"
    fitness_scorer: PopulationFitnessScorer
    value: Any

    @property
    def input_keys(self) -> List[str]:
        return ["problem_description"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def initialize(self, problem_description: str, **kwargs) -> Population:
        return self.run({"problem_description": problem_description}, **kwargs)

    async def ainitialize(self, problem_description: str, **kwargs) -> Population:
        return await self.arun({"problem_description": problem_description}, **kwargs)

    def _call(self, inputs: Dict[str, Any], run_manager=None, **kwargs):
        members = []
        for i in tqdm(range(self.n_units)):
            fitness = self.value
            while fitness <= self.value:
                member = self.initializer.initialize(
                    inputs["problem_description"],
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs
                )
                fitness = self.fitness_scorer.score(member.task_prompt_set, **kwargs)
            members += [member]

        return {self.output_key: Population(members=members)}

    async def _acreate_one(
        self, inputs: Dict[str, Any], run_manager=None, **kwargs
    ) -> UnitOfEvolution:
        fitness = self.value
        while fitness <= self.value:
            member = await self.initializer.ainitialize(
                inputs["problem_description"],
                callbacks=run_manager.get_child() if run_manager else None,
                **kwargs
            )
            fitness = await self.fitness_scorer.ascore(member.task_prompt_set, **kwargs)
        return member

    async def _acall(self, inputs: Dict[str, Any], run_manager=None, **kwargs):
        members = await asyncio.gather(
            *[
                self._acreate_one(inputs, run_manager, **kwargs)
                for i in range(self.n_units)
            ]
        )
        return {self.output_key: Population(members=members)}
