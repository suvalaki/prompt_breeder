import pytest  # noqa: F401
from typing import Dict, List, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
)
from prompt_breeder.mutators.lineage_based_mutation import (
    LineageBasedMutation,
)


KEY0 = "key0"
KEY1 = "key1"

# "{task_prompt_set}  INSTRUCTION MUTATNT: "
FIXED_PROMPT_REPLY: Dict[str, str] = {
    KEY0: KEY0,
    KEY1: KEY1,
}


class MockLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom_lineage"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if KEY0 in prompt and KEY1 in prompt:
            return "both"
        elif KEY0 in prompt:
            return KEY0
        elif KEY1 in prompt:
            return KEY1
        else:
            return "niether"


def test_runs_over_unit():
    llm = MockLLM()
    prompt0 = StringTaskPrompt(text=KEY0)
    prompt1 = StringTaskPrompt(text=KEY1)
    unit = UnitOfEvolution(
        problem_description=StringPrompt(text="ignored"),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="ignored"),
        elites=[prompt0, prompt1],
    )
    mutator = LineageBasedMutation.from_llm(
        llm=llm,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        verbose=1,
    )
    population = Population(members=[unit])
    ans = mutator.mutate(population, unit)
    assert all([str(x) == "both" for x in ans.task_prompt_set])
