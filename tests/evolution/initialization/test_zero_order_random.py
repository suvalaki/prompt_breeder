import pytest  # noqa: F401
from langchain.llms import Ollama
from prompt_breeder.prompts.string import (
    StringTaskPrompt,
    StringMutationPrompt,
    StringThinkingStyle,
    StringProblemDescription,
)
from prompt_breeder.evolution.initialization.base import PopulationInitialization
from prompt_breeder.evolution.initialization.zero_order_random import (
    ZeroOrderInitialization,
)


class MockThinkingStyleProvider:
    static_styles = [
        "What are the potential risks and drawbacks of each solution?",
        "What are the alternative perspectives or viewpoints on this problem",
        "What are the long-term implications of this problem and its solutions?",
    ]

    def __init__(self):
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        self.i += 1
        if self.i >= len(self.static_styles):
            self.i = 0
        return StringThinkingStyle(text=self.static_styles[self.i])


class MockMutationPromptProvider:
    static_styles = [
        "What errors are there in the solution?",
        "How could you improve the working out of the problem?",
        "Look carefully to see what you did wrong, how could you fix the problem?",
    ]

    def __init__(self):
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i >= len(self.static_styles):
            self.i = 0
        return StringMutationPrompt(text=self.static_styles[self.i])


def test_creates_requuired_different_initial_thinking_styles():
    llm = Ollama(model="mistral")
    initializer = ZeroOrderInitialization.from_llm(
        problem_description_factory=lambda x: StringProblemDescription(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        thinking_style_provider=MockThinkingStyleProvider(),
        mutation_prompt_provider=MockMutationPromptProvider(),
        llm=llm,
        n_members_per_unit=3,
        verbose=1,
    )
    ans = initializer.initialize(problem_description="Solve the math word problem")
    assert len(ans.task_prompt_set) == 3
    assert all(
        [x != y for x in ans.task_prompt_set for y in ans.task_prompt_set if x is not y]
    )


def test_population():
    llm = Ollama(model="mistral")
    initializer = ZeroOrderInitialization.from_llm(
        problem_description_factory=lambda x: StringProblemDescription(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        thinking_style_provider=MockThinkingStyleProvider(),
        mutation_prompt_provider=MockMutationPromptProvider(),
        llm=llm,
        n_members_per_unit=3,
        verbose=1,
    )
    pop_initializer = PopulationInitialization(
        initializer=initializer,
        n_units=2,
    )
    ans = pop_initializer.initialize(problem_description="Solve the math word problem")
    assert len(ans.members) == 2
    assert all([x != y for x in ans.members for y in ans.members if x is not y])
