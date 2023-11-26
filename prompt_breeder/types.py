from typing import List
from pydantic import BaseModel, ConfigDict

from abc import ABC, abstractmethod


class Prompt(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def replace(self, *args, **kwargs):
        ...


class TaskPrompt(Prompt):
    """We are interested in evolving task prompts. A task-prompt P is a string used to
    condition the context of an LLM in advance of some further input Q, intended to
    ensure a better response than if Q had been presented in the absence of P .
    """

    ...


class MutationPrompt(Prompt):
    """The text prompt which when concatenated to the task-prompt is intended to
    produce a continuation which is an improved task-prompt.

    a mutated task prompt P ′ is defined by P ′ = LLM(M + P ) where ‘+‘ corresponds
    to string concatenation.
    """

    ...


class HyperMutationPrompt(MutationPrompt):
    """The text prompt which when concatenated to the task-prompt is intended to
    produce a continuation which is an improved task-prompt.

    we obtain a mutated mutation-prompt M ′ via M ′ = LLM(H + M )
    """

    ...


class ProblemDescription(Prompt):
    """The initial text description of the problem which could be used as the ini-
    tial task-prompt. The user can make their best attempt to produce an effective
    problem description, which is the starting point of Promptbreeder."""

    ...


class ThinkingStyle(Prompt):
    """"""

    ...


class ApplicationRule:
    pass


class PromptStrategy(BaseModel):
    """A set of task-prompts and rules for their application at inference time during a
    fitness evaluation. In the minimal case the prompt strategy is just a single
    task-prompt. Typically our prompt strategies consisted of two sequentially applied
    task-prompts.
    """

    task_prompts: List[TaskPrompt]
    rules: List[ApplicationRule]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Phenotype(Prompt):
    """(Phenotype/Workings out/Context/Reasoning Path) Used interchangeably to mean
    their output of the LLM on a specific question or problem when prompted with
    the task-prompt concatenated to the question."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class UnitOfEvolution(BaseModel):
    """The informational structure that is being evolved, here consisting of a task-
    prompt set (typically 2), a mutation-prompt"""

    problem_description: Prompt
    task_prompt_set: List[TaskPrompt]
    mutation_prompt: MutationPrompt
    # For each unit of evolution, we store a history of the individuals in its lin-
    # eage that were the best in the population
    elites: List[TaskPrompt] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FewShowUnitOfEvolution(UnitOfEvolution):
    """The informational structure that is being evolved, here consisting of a task-
    prompt set (typically 2), a mutation-prompt, and in the few-shot case a set of
    2-3 contexts (workings out)."""

    contexts: List[Phenotype]


class Population(BaseModel):
    """The set of units of evolution (e.g. 50)."""

    members: List[UnitOfEvolution]
    elites: List[TaskPrompt] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)
