from typing import List, Callable
from langchain.llms.base import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.output_parsers.regex import RegexParser

from prompt_breeder.types import MutationPrompt, TaskPrompt
from prompt_breeder.mutators.base import DirectMutator


BASE_TEMPLATE = PromptTemplate.from_template(
    "{problem_description} An ordered list of 100 hints: "
)
CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Act as a expert improving a problem description.\n"
            "Generate a new improved hint to complete the task. "
            "Reply only with an ordered list of 10 hints. "
            "For example: '1. ..., 2. ..., 3. ...'.\n\n"
            "TASK: {problem_description}"
        ),
    ]
)
PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=BASE_TEMPLATE,
    conditionals=[(is_chat_model, CHAT_TEMPLATE)],
)


class SingleRegexParser(RegexParser):
    """A passthrough output parser to prevent Regex outputs
    comming in a dict."""

    output_key: str = "output"
    output_keys: List[str] = ["_0", "_1", "output"]

    def parse(self, text: str) -> str:  # type: ignore
        try:
            parsed = super().parse(text)
            val = parsed[self.output_key]
            while val.startswith(" "):
                val = val[1:]
            while val.endswith("\n"):
                val = val[:-2]
            return val
        except Exception:
            return ""


class ZeroOrderPromptGeneration(LLMChain, DirectMutator):
    """Generate a new task-prompt by concatenating the problem description D
    (e.g. "Solve the math word problem, giving your answer as an arabic numeral")
    with the prompt "A list of 100 hints:", which invites the LLM to come up with
    a new hint that could help solve a problem in the given problem domain."""

    output_parser: SingleRegexParser = SingleRegexParser(
        regex=r"((.|\n)*1\.)((.|\n)*?)2\."
    )

    @classmethod
    def from_llm(
        cls,
        mutation_prompt_factory: Callable[[str], MutationPrompt],
        task_prompt_factory: Callable[[str], TaskPrompt],
        llm: BaseLanguageModel,
        **kwargs
    ):
        return cls(
            llm=llm,
            prompt=PROMPT_SELECTOR.get_prompt(llm),
            mutation_prompt_factory=mutation_prompt_factory,
            task_prompt_factory=task_prompt_factory,
            **kwargs
        )
