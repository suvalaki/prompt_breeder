from typing import Callable, Iterator
from langchain.llms.base import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from prompt_breeder.types import TaskPrompt, MutationPrompt, ThinkingStyle
from prompt_breeder.mutators.base import Mutator, Hypermutation
from prompt_breeder.mutators.first_order_prompt_generation import (
    FirstOrderPromptGeneration,
)


class FirstOrderMutation(LLMChain, Mutator):
    prompt = PromptTemplate.from_template(
        "Please summarize and improve the following instruction: {mutation_prompt} "
    )


class FirstOrderHypermutation(Hypermutation):
    """Concatenate the hyper-mutation-prompt "Please summarize
    and improve the following instruction:" to a mutation-prompt so that the LLM gener-
    ates a new mutation-prompt. This newly generated mutation-prompt is then applied to
    the taskprompt of that unit (see First-Order Prompt Generation)."""

    mutate_mutator_chain: FirstOrderMutation
    mutate_task_prompt_chain: FirstOrderPromptGeneration

    @classmethod
    def from_llm(
        cls,
        mutation_prompt_factory: Callable[[str], MutationPrompt],
        task_prompt_factory: Callable[[str], TaskPrompt],
        thinking_style_provider: Iterator[ThinkingStyle],
        llm: BaseLanguageModel,
        **kwargs
    ):
        return cls(
            task_prompt_factory=task_prompt_factory,
            mutation_prompt_factory=mutation_prompt_factory,
            thinking_style_provider=thinking_style_provider,
            mutate_mutator_chain=FirstOrderMutation(
                llm=llm,
                task_prompt_factory=task_prompt_factory,
                mutation_prompt_factory=mutation_prompt_factory,
                **kwargs,
            ),
            mutate_task_prompt_chain=FirstOrderPromptGeneration(
                llm=llm,
                task_prompt_factory=task_prompt_factory,
                mutation_prompt_factory=mutation_prompt_factory,
                **kwargs,
            ),
            **kwargs,
        )
