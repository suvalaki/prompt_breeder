import pytest  # noqa: F401
import os  # noqa: F401

from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
)
from prompt_breeder.evolution.callbacks.save_population import (
    IncrementAge,
    SavePopulation,
    SaveEachPopulation,
)


def create_pop():
    prompt00 = StringTaskPrompt(
        text="one Solve the math word problem, show your workings.     "
    )
    prompt01 = StringTaskPrompt(text="one Solve the math word problem.      ")
    unit0 = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral."
        ),
        task_prompt_set=[
            prompt00,
            prompt01,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )

    prompt10 = StringTaskPrompt(
        text="two Solve the math word problem, show your workings."
    )
    prompt11 = StringTaskPrompt(text="two Solve the math word problem.")
    unit1 = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral."
        ),
        task_prompt_set=[
            prompt10,
            prompt11,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )

    population = Population(members=[unit0, unit1])
    return population


def test_age_increases_inplace():
    population = create_pop()
    age0 = population.age
    incrementer = IncrementAge()
    incrementer(population)
    age1 = population.age
    assert age1 == age0 + 1


def test_save_population(tmp_path_factory):
    population = create_pop()
    pth = tmp_path_factory.mktemp("data") / "population.json"
    saver = SavePopulation(fp=str(pth))
    saver(population)
    assert pth.exists()


def test_save_each_population(tmp_path_factory):
    population = create_pop()
    pth = tmp_path_factory.mktemp("data")
    saver = SaveEachPopulation(fp=str(pth))
    filepth = pth / "population_0.json"
    saver(population)
    assert filepth.exists()
