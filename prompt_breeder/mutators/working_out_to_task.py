import asyncio
from typing import Iterator, Callable, List
from copy import deepcopy
from langchain.llms.base import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model

from prompt_breeder.types import (
    Phenotype,
    UnitOfEvolution,
    Population,
)
from prompt_breeder.mutators.base import Mutator
from prompt_breeder.types import MutationPrompt, TaskPrompt

# from langchain.prompts.example_selector.base import BaseExampleSelector

BASE_TEMPLATE = PromptTemplate.from_template(
    "I gave a friend an instruction and some advice. "
    "Here are the correct examples of his workings out: \n{context}\n"
    "The instruction was: "
)
CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are assisting in the creation of a generic problem description "
            "to be applied over multiple questions.\n\n"
            "A student was provided with a probelm description and some advice. "
            "Here are several examples of correct workings out: \n{context}\n\n"
            "What was the generic problem description and advice which is "
            "applicable to all examples?"
        ),
    ]
)
PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=BASE_TEMPLATE,
    conditionals=[(is_chat_model, CHAT_TEMPLATE)],
)


# Lamarkin Operator
class WorkingOutToTask(LLMChain, Mutator):
    """We give an LLM a previously generated working out that led to a correct answer
    via the following prompt: "I gave a friend an instruction and some advice. Here
    are the correct examples of his workings out + <<correct working out>> +
    The instruction was:". This is effectively reverse-engineering the task-prompt from
    a given working out."""

    # Get new correct workings
    correct_working_out_provider: Iterator[Phenotype]
    max_context_size: int = 5

    @classmethod
    def from_llm(
        cls,
        mutation_prompt_factory: Callable[[str], MutationPrompt],
        task_prompt_factory: Callable[[str], TaskPrompt],
        correct_working_out_provider: Iterator[Phenotype],
        llm: BaseLanguageModel,
        **kwargs
    ):
        return cls(
            llm=llm,
            prompt=PROMPT_SELECTOR.get_prompt(llm),
            mutation_prompt_factory=mutation_prompt_factory,
            task_prompt_factory=task_prompt_factory,
            correct_working_out_provider=correct_working_out_provider,
            **kwargs
        )

    def mutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        unit = deepcopy(unit)
        examples = [
            next(self.correct_working_out_provider)
            for i in range(self.max_context_size)
        ]

        unit.task_prompt_set = [
            self.task_prompt_factory(
                self.run(
                    {
                        "context": "\n\n".join(
                            ["EXAMPLE: " + str(example) for example in examples]
                        ),
                    },
                    **kwargs
                )
            )
            for member in unit.task_prompt_set
        ]
        return unit

    async def _asingleton_task_prompt(self, examples: List[Phenotype], **kwargs):
        return self.task_prompt_factory(
            await self.arun(
                {
                    "context": "\n\n".join(
                        ["EXAMPLE: " + str(example) for example in examples]
                    ),
                },
                **kwargs
            )
        )

    async def amutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        unit = deepcopy(unit)
        examples = [
            next(self.correct_working_out_provider)
            for i in range(self.max_context_size)
        ]
        unit.task_prompt_set = await asyncio.gather(
            *[self._asingleton_task_prompt(examples) for member in unit.task_prompt_set]
        )
        return unit
