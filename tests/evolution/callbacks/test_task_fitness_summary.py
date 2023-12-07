import pytest  # noqa: F401

from prompt_breeder.prompts.string import (
    StringTaskPrompt,
)
from prompt_breeder.evolution.callbacks.task_fitness_summary import TaskPromptSummary

from .test_save_population import create_pop


class StringLengthFitness:
    def score(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return len(str(prompt))

    def ascore(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return len(str(prompt))

    def get_all(self, prompt: StringTaskPrompt, **kwargs):
        return [None]


def test_saves_task_prompt_summary(tmp_path_factory):
    pop = create_pop()
    pth = tmp_path_factory.mktemp("data") / "detailed_output.csv"
    scorer = StringLengthFitness()
    saver = TaskPromptSummary(
        fp=str(pth), fitness_scorer=scorer, val_fitness_scorer=scorer
    )
    saver(pop)
    assert pth.exists()
