from pydantic import BaseModel
from ..types import (
    Prompt,
    TaskPrompt,
    MutationPrompt,
    HyperMutationPrompt,
    ProblemDescription,
    ThinkingStyle,
    Phenotype,
)


class StringPrompt(Prompt, BaseModel):
    text: str

    def __str__(self):
        return self.text

    def replace(self, x: str):
        self.text = x


class StringTaskPrompt(TaskPrompt, StringPrompt):
    pass


class StringMutationPrompt(MutationPrompt, StringPrompt):
    pass


class StringHyperMutationPrompt(HyperMutationPrompt, StringPrompt):
    pass


class StringProblemDescription(ProblemDescription, StringPrompt):
    pass


class StringThinkingStyle(ThinkingStyle, StringPrompt):
    pass


class StringPhenotype(Phenotype, StringPrompt):
    pass
