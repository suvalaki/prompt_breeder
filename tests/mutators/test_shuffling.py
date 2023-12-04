import pytest  # noqa: F401
import asyncio
from prompt_breeder.types import FewShowUnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
    StringPhenotype,
)
from prompt_breeder.mutators.shuffling import (
    ContextShuffling,
)


class MockContextProvider:
    def __init__(self):
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self) -> StringPhenotype:
        self.i += 1
        return StringPhenotype(text=f"Phenotype {self.i}")


def test_adds_to_full_if_not_exhausted():
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    correct_working_out_provider = MockContextProvider()
    unit = FewShowUnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        contexts=[next(correct_working_out_provider) for i in range(2)],
        elites=[],
    )
    mutator = ContextShuffling(
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        probability_of_refresh_full_list=0,  # Will never call a full refresh.
        # we put this here to validate not full context gets filled anyway
        correct_working_out_provider=correct_working_out_provider,
        verbose=1,
    )
    population = Population(members=[unit])

    context0 = unit.contexts
    ans = mutator.mutate(population, unit)
    context1 = ans.contexts

    assert len(context0) < 10
    assert len(context1) == 10


def test_complete_refresh():
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    correct_working_out_provider = MockContextProvider()
    unit = FewShowUnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        contexts=[next(correct_working_out_provider) for i in range(10)],
        elites=[],
    )
    mutator = ContextShuffling(
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        probability_of_refresh_full_list=1,
        correct_working_out_provider=correct_working_out_provider,
        verbose=1,
    )
    population = Population(members=[unit])

    context0 = unit.contexts
    ans = mutator.mutate(population, unit)
    context1 = ans.contexts

    assert all([x not in y for x in context0 for y in context1])


def test_replace_one():
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    correct_working_out_provider = MockContextProvider()
    unit = FewShowUnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        contexts=[next(correct_working_out_provider) for i in range(1)],
        elites=[],
    )
    mutator = ContextShuffling(
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        probability_of_refresh_full_list=0,
        max_context_size=1,
        correct_working_out_provider=correct_working_out_provider,
        verbose=1,
    )
    population = Population(members=[unit])

    # Setting max context size to 1 means the inverse is 1 so the prob of replacement
    # is 1. setting full replcamenet prob to zero we can validate a change

    context0 = unit.contexts
    ans = mutator.mutate(population, unit)
    context1 = ans.contexts

    assert all([x not in y for x in context0 for y in context1])

    context0 = unit.contexts
    ans = asyncio.run(mutator.amutate(population, unit))
    context1 = ans.contexts

    assert all([x not in y for x in context0 for y in context1])
