import pytest  # noqa: F401
import asyncio
from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
)
from prompt_breeder.mutators.crossover import (
    PromptCrossover,
)
from prompt_breeder.evolution.fitness import Fitness

# Lets make a custom fitness that is just the prompt length


class StringLengthFitness(Fitness):
    def score(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return len(str(prompt))

    async def ascore(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return len(str(prompt))


def test_probability_map_includes_only_other_units():
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    unit0 = UnitOfEvolution(
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

    prompt2 = StringTaskPrompt(
        text="Therefore Solve the math word problem, show your workings."
    )
    prompt3 = StringTaskPrompt(text="Therefore Solve the math word problem.")
    unit1 = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt2,
            prompt3,
        ],
        mutation_prompt=StringMutationPrompt(text="therefore make the task better."),
        elites=[],
    )

    population = Population(members=[unit0, unit1])

    mutator = PromptCrossover(
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=StringLengthFitness(),
        probability_of_replacement=1.0,  # always mutate
        verbose=1,
    )

    members0, probs0 = mutator._get_probability_map(population, unit0)
    assert len(members0) == 2
    assert len(probs0) == 2
    assert all([el not in unit0.task_prompt_set for el in members0])
    assert sum(probs0) == pytest.approx(1)

    members1, probs1 = mutator._get_probability_map(population, unit1)
    assert len(members1) == 2
    assert len(probs1) == 2
    assert all([el not in unit1.task_prompt_set for el in members1])
    assert sum(probs1) == pytest.approx(1)


def test_mutation_isnt_inplace():
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    unit0 = UnitOfEvolution(
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

    prompt2 = StringTaskPrompt(
        text="Therefore Solve the math word problem, show your workings."
    )
    prompt3 = StringTaskPrompt(text="Therefore Solve the math word problem.")
    unit1 = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt2,
            prompt3,
        ],
        mutation_prompt=StringMutationPrompt(text="therefore make the task better."),
        elites=[],
    )

    population = Population(members=[unit0, unit1])

    mutator = PromptCrossover(
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=StringLengthFitness(),
        probability_of_replacement=1.0,  # always mutate
        verbose=1,
    )

    mutant = mutator.mutate(population, unit0)
    assert all([mutant is not unit0, mutant is not unit1])
    assert all([mutant is not x for x in population.members])

    # because probability of replacement is 1 it myst be the case that
    # at least one string prompt inside our mutant is diffierent.
    # In this case it should come from unit1.
    assert any(
        [
            task.text in [task1.text for task1 in unit1.task_prompt_set]
            for task in mutant.task_prompt_set
        ]
    )
    assert (
        sum(
            [
                task.text in [task1.text for task1 in unit1.task_prompt_set]
                for task in mutant.task_prompt_set
            ]
        )
        == 1
    )
    assert (
        sum(
            [
                task.text in [task1.text for task1 in unit0.task_prompt_set]
                for task in mutant.task_prompt_set
            ]
        )
        == len(unit0.task_prompt_set) - 1
    )

    mutant = asyncio.run(mutator.amutate(population, unit0))
    assert all([mutant is not unit0, mutant is not unit1])
    assert all([mutant is not x for x in population.members])

    # Check setting probability to zero correctly means the mutator
    # does nothing

    mutator = PromptCrossover(
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=StringLengthFitness(),
        probability_of_replacement=0.0,  # always mutate
        verbose=1,
    )

    mutant = mutator.mutate(population, unit0)
    # Check references are different - new object created
    assert all([mutant is not unit0, mutant is not unit1])
    assert all([mutant is not x for x in population.members])

    # because probability of replacement is 1 it myst be the case that
    # at least one string prompt inside our mutant is diffierent.
    # In this case it should come from unit1.
    #
    assert not any(
        [
            task.text in [task1.text for task1 in unit1.task_prompt_set]
            for task in mutant.task_prompt_set
        ]
    )
    assert (
        not sum(
            [
                task.text in [task1.text for task1 in unit1.task_prompt_set]
                for task in mutant.task_prompt_set
            ]
        )
        == 1
    )
    assert (
        not sum(
            [
                task.text in [task1.text for task1 in unit0.task_prompt_set]
                for task in mutant.task_prompt_set
            ]
        )
        == len(unit0.task_prompt_set) - 1
    )
