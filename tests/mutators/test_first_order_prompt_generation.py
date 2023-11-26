import pytest  # noqa: F401
from langchain.llms import Ollama
from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
)
from prompt_breeder.mutators.first_order_prompt_generation import (
    FirstOrderPromptGeneration,
)


def test_runs_over_unit():
    llm = Ollama(model="mistral")
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
    mutator = FirstOrderPromptGeneration(
        llm=llm,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        verbose=1,
    )
    population = Population(members=[unit])
    _ = mutator.mutate(population, unit)
