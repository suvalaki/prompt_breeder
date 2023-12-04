import asyncio
from typing import Callable, Iterator, List
from abc import abstractmethod
from langchain.chains.base import Chain
from prompt_breeder.types import (
    UnitOfEvolution,
    ThinkingStyle,
    MutationPrompt,
    TaskPrompt,
    Population,
)


class Mutator(Chain):
    task_prompt_factory: Callable[[str], TaskPrompt]
    mutation_prompt_factory: Callable[[str], MutationPrompt]

    @abstractmethod
    def mutate(self, population: Population, unit: UnitOfEvolution, **kwargs):
        pass

    @abstractmethod
    async def amutate(self, population: Population, unit: UnitOfEvolution, **kwargs):
        pass


class DirectMutator(Mutator):
    def mutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        # Only mutate the task using the task and mutation prompt
        mutant_population = [
            self._wrapped_task_run(population, unit, member, **kwargs)
            for member in unit.task_prompt_set
        ]
        return UnitOfEvolution(
            problem_description=unit.problem_description,
            task_prompt_set=mutant_population,
            mutation_prompt=unit.mutation_prompt,
            elites=unit.elites,
        )

    async def amutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        # Only mutate the task using the task and mutation prompt
        mutant_population = await asyncio.gather(
            *[
                self._awrapped_task_run(population, unit, member, **kwargs)
                for member in unit.task_prompt_set
            ]
        )
        return UnitOfEvolution(
            problem_description=unit.problem_description,
            task_prompt_set=mutant_population,
            mutation_prompt=unit.mutation_prompt,
            elites=unit.elites,
        )

    def _wrapped_task_run(
        self,
        population: Population,
        unit: UnitOfEvolution,
        member: TaskPrompt,
        **kwargs
    ) -> TaskPrompt:
        return self.task_prompt_factory(
            self.run(
                {
                    "population": population,
                    "problem_description": str(unit.problem_description),
                    "task_prompt": str(member),
                    "mutation_prompt": str(unit.mutation_prompt),
                },
                **kwargs
            )
        )

    async def _awrapped_task_run(
        self,
        population: Population,
        unit: UnitOfEvolution,
        member: TaskPrompt,
        **kwargs
    ) -> TaskPrompt:
        return self.task_prompt_factory(
            await self.arun(
                {
                    "population": population,
                    "problem_description": str(unit.problem_description),
                    "task_prompt": str(member),
                    "mutation_prompt": str(unit.mutation_prompt),
                },
                **kwargs
            )
        )


class DistributionEstimationMutator(DirectMutator):
    def _wrapped_task_run(
        self,
        population: Population,
        unit: UnitOfEvolution,
        member: TaskPrompt,
        **kwargs
    ) -> TaskPrompt:
        return self.task_prompt_factory(
            self.run(
                {
                    "population": population,
                    "problem_description": str(unit.problem_description),
                    "task_prompt": str(member),
                    "mutation_prompt": str(unit.mutation_prompt),
                    "task_prompt_set": [str(x) for x in unit.task_prompt_set],
                    "elites": [str(x) for x in unit.elites],
                },
                **kwargs
            )
        )

    async def _awrapped_task_run(
        self,
        population: Population,
        unit: UnitOfEvolution,
        member: TaskPrompt,
        **kwargs
    ) -> TaskPrompt:
        return self.task_prompt_factory(
            await self.arun(
                {
                    "population": population,
                    "problem_description": str(unit.problem_description),
                    "task_prompt": str(member),
                    "mutation_prompt": str(unit.mutation_prompt),
                    "task_prompt_set": [str(x) for x in unit.task_prompt_set],
                    "elites": [str(x) for x in unit.elites],
                },
                **kwargs
            )
        )


class Hypermutation(DistributionEstimationMutator):
    """Hyper mutations first apply a mutator to the mutation prompt (to generate
    a new mutation prompt). The new mutation prompt is then used to mutate the
    task prompt using direct FirstOrderMutation.
    """

    thinking_style_provider: Iterator[ThinkingStyle]
    mutate_mutator_chain: Chain
    mutate_task_prompt_chain: Chain
    output_key: str = "output"

    @property
    def input_keys(self):
        return list(UnitOfEvolution.model_fields.keys())

    @property
    def output_keys(self):
        return [self.output_key]

    def _call(self, inputs, run_manager=None, **kwargs):
        cb = run_manager.get_child() if run_manager else None

        thinking_style = next(self.thinking_style_provider)

        # mutate the task and the mutation using all the task prompts in the unit
        # population
        meta_mutator: str = self.mutate_mutator_chain.run(
            {"thinking_style": str(thinking_style), **inputs}, callbacks=cb, **kwargs
        )

        mutant_population: List[str] = [
            self.mutate_task_prompt_chain.run(
                {
                    "population": inputs["population"],
                    "thinking_style": str(thinking_style),
                    "task_prompt": member,
                    "mutation_prompt": meta_mutator,
                    **{k: v for k, v in inputs.items() if k != "mutation_prompt"},
                },
                callbacks=cb,
                **kwargs
            )
            for member in inputs["task_prompt_set"]
        ]
        return {
            self.output_key: {
                "task_prompt_set": mutant_population,
                "mutation_prompt": meta_mutator,
            }
        }

    async def _acall(self, inputs, run_manager=None, **kwargs):
        cb = run_manager.get_child() if run_manager else None

        thinking_style = next(self.thinking_style_provider)

        # mutate the task and the mutation using all the task prompts in the unit
        # population
        meta_mutator: str = await self.mutate_mutator_chain.arun(
            {"thinking_style": str(thinking_style), **inputs}, callbacks=cb, **kwargs
        )

        mutant_population: List[str] = await asyncio.gather(
            *[
                self.mutate_task_prompt_chain.arun(
                    {
                        "population": inputs["population"],
                        "thinking_style": str(thinking_style),
                        "task_prompt": member,
                        "mutation_prompt": meta_mutator,
                        **{k: v for k, v in inputs.items() if k != "mutation_prompt"},
                    },
                    callbacks=cb,
                    **kwargs
                )
                for member in inputs["task_prompt_set"]
            ]
        )
        return {
            self.output_key: {
                "task_prompt_set": mutant_population,
                "mutation_prompt": meta_mutator,
            }
        }

    def mutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        response = self.run(
            {
                "population": population,
                "problem_description": str(unit.problem_description),
                "mutation_prompt": str(unit.mutation_prompt),
                "task_prompt_set": [str(x) for x in unit.task_prompt_set],
                "elites": [str(x) for x in unit.elites],
            },
            **kwargs
        )
        return UnitOfEvolution(
            problem_description=unit.problem_description,
            mutation_prompt=self.mutation_prompt_factory(response["mutation_prompt"]),
            task_prompt_set=[
                self.task_prompt_factory(mutant)
                for mutant in response["task_prompt_set"]
            ],
            elites=unit.elites,
        )

    async def amutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        response = await self.arun(
            {
                "population": population,
                "problem_description": str(unit.problem_description),
                "mutation_prompt": str(unit.mutation_prompt),
                "task_prompt_set": [str(x) for x in unit.task_prompt_set],
                "elites": [str(x) for x in unit.elites],
            },
            **kwargs
        )
        return UnitOfEvolution(
            problem_description=unit.problem_description,
            mutation_prompt=self.mutation_prompt_factory(response["mutation_prompt"]),
            task_prompt_set=[
                self.task_prompt_factory(mutant)
                for mutant in response["task_prompt_set"]
            ],
            elites=unit.elites,
        )
