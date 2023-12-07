import pytest  # noqa: F401

from prompt_breeder.mutators.reinit import ReplaceWithInit
from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
)
from prompt_breeder.evolution.fitness import BestMemberFitness


class MockScorerAndInitializer:
    score_mapping = {
        "0": 0,
        "1": 0,
        "2": 3,
        "3": 1,
    }

    def __init__(self, i=0):
        self.i = i

    def initialize(self, problem_description: str, **kwargs) -> UnitOfEvolution:
        self.i += 1
        val = StringTaskPrompt(text=str(self.i))

        unit = UnitOfEvolution(
            problem_description=StringPrompt(text=""),
            task_prompt_set=[
                val,
            ],
            mutation_prompt=StringMutationPrompt(text=""),
            elites=[],
        )

        return unit

    def score(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return self.score_mapping[prompt.text]

    def ascore(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return self.score_mapping[prompt.text]

    def get_all(self, prompt: StringTaskPrompt, **kwargs):
        return [None]


def test_reinit_on_zero_score():
    scorer = MockScorerAndInitializer(i=0)
    multi = BestMemberFitness(scorer=scorer)
    init = ReplaceWithInit(
        fitness_scorer=multi,
        value=0,
        initializer=scorer,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
    )

    val = StringTaskPrompt(text=str(0))
    unit = UnitOfEvolution(
        problem_description=StringPrompt(text=""),
        task_prompt_set=[
            val,
        ],
        mutation_prompt=StringMutationPrompt(text=""),
        elites=[],
    )
    ans = init.mutate(Population(members=[]), unit)

    assert ans.task_prompt_set[0].text == "1"


def test_no_init_on_positive_until_positive_score():
    scorer = MockScorerAndInitializer(i=2)
    multi = BestMemberFitness(scorer=scorer)
    init = ReplaceWithInit(
        fitness_scorer=multi,
        value=0,
        initializer=scorer,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
    )

    val = StringTaskPrompt(text=str(0))
    unit = UnitOfEvolution(
        problem_description=StringPrompt(text=""),
        task_prompt_set=[
            val,
        ],
        mutation_prompt=StringMutationPrompt(text=""),
        elites=[],
    )
    ans = init.mutate(Population(members=[]), unit)

    assert ans.task_prompt_set[0].text == "3"
