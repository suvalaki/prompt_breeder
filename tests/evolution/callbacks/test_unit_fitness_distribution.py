import pytest  # noqa: F401

from prompt_breeder.prompts.string import (
    StringTaskPrompt,
)
from prompt_breeder.evolution.fitness import BestMemberFitness
from prompt_breeder.evolution.callbacks.unit_fitness_distribution import (
    UnitFitnessSummary,
)

from .test_save_population import create_pop


class StringLengthFitness:
    def score(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return len(str(prompt))

    def ascore(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return len(str(prompt))

    def get_all(self, prompt: StringTaskPrompt, **kwargs):
        return [None]


def test_saves_unit_fitness_summary(tmp_path_factory):
    pop = create_pop()
    pth = tmp_path_factory.mktemp("data") / "distribution_output.csv"
    scorer = StringLengthFitness()
    multi_scorer = BestMemberFitness(scorer=scorer)
    saver = UnitFitnessSummary(
        fp=str(pth), fitness_scorer=multi_scorer, val_fitness_scorer=multi_scorer
    )
    saver(pop)
    assert pth.exists()
