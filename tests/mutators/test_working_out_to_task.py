import pytest  # noqa: F401
from typing import List, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
    StringPhenotype,
)
from prompt_breeder.mutators.working_out_to_task import (
    WorkingOutToTask,
)


class MockPassthroughLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "passthrough"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return prompt


class MockContextProvider:
    def __init__(self):
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self) -> StringPhenotype:
        self.i += 1
        return StringPhenotype(text=f"Phenotype {self.i}")


def test_runs_over_unit():
    llm = MockPassthroughLLM()
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    unit = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )
    mutator = WorkingOutToTask(
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        correct_working_out_provider=MockContextProvider(),
        llm=llm,
        verbose=1,
    )
    population = Population(members=[unit])
    _ = mutator.mutate(population, unit)
