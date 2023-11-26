import pytest  # noqa: F401
from langchain.llms import Ollama
from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
    StringThinkingStyle,
)
from prompt_breeder.mutators.zero_order_hypermutation import (
    ZeroOrderHypermutation,
)


class MockThinkingStyleProvider:
    def __iter__(self):
        return self

    def __next__(self):
        return StringThinkingStyle(text="Lets think about this.")


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
    mutator = ZeroOrderHypermutation.from_llm(
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        thinking_style_provider=MockThinkingStyleProvider(),
        llm=llm,
        verbose=1,
    )
    population = Population(members=[unit])
    _ = mutator.mutate(population, unit)
