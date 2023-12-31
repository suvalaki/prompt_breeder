import pytest  # noqa: F401
import asyncio
from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
)
from prompt_breeder.evolution.fitness import Fitness
from prompt_breeder.mutators.elite import (
    AddElite,
)

# Lets make a custom fitness that is just the prompt length


class StringLengthFitness(Fitness):
    def score(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return len(str(prompt))

    async def ascore(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return len(str(prompt))


def test_runs_over_unit():
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    unit = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )
    mutator = AddElite(
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=StringLengthFitness(),
        verbose=1,
    )
    population = Population(members=[unit])
    ans = mutator.mutate(population, unit)
    assert len(ans.elites) == 1


def test_async_runs_over_unit():
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    unit = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )
    mutator = AddElite(
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=StringLengthFitness(),
        verbose=1,
    )
    population = Population(members=[unit])
    ans = asyncio.run(mutator.amutate(population, unit))
    assert len(ans.elites) == 1


def test_runs_non_empty_empty_unit():
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    unit = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[StringTaskPrompt(text="Shorter string")],
    )
    mutator = AddElite(
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=StringLengthFitness(),
        verbose=1,
    )
    population = Population(members=[unit])
    ans = mutator.mutate(population, unit)
    assert len(ans.elites) == 2


def test_async_runs_non_empty_empty_unit():
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    unit = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[StringTaskPrompt(text="Shorter string")],
    )
    mutator = AddElite(
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=StringLengthFitness(),
        verbose=1,
    )
    population = Population(members=[unit])
    ans = asyncio.run(mutator.amutate(population, unit))
    assert len(ans.elites) == 2


def test_runs_non_empty_empty_unit_skips_lower_fitness():
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    unit = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[
            StringTaskPrompt(
                text="longer_string"
                "Solve the math word problem, giving your answer as an arabic numeral"
            )
        ],
    )
    mutator = AddElite(
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=StringLengthFitness(),
        verbose=1,
    )
    population = Population(members=[unit])
    ans = mutator.mutate(population, unit)
    assert len(ans.elites) == 1


def test_async_runs_non_empty_empty_unit_skips_lower_fitness():
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    unit = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[
            StringTaskPrompt(
                text="longer_string"
                "Solve the math word problem, giving your answer as an arabic numeral"
            )
        ],
    )
    mutator = AddElite(
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=StringLengthFitness(),
        verbose=1,
    )
    population = Population(members=[unit])
    ans = asyncio.run(mutator.amutate(population, unit))
    assert len(ans.elites) == 1
