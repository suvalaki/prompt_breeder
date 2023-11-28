from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from prompt_breeder.mutators.base import DirectMutator


class FirstOrderPromptGeneration(LLMChain, DirectMutator):
    """We concatenate the mutation-prompt (red), to the parent
    task-prompt (blue), and pass it to the LLM to produce the mutated task-prompt.

    This procedure is identical to the initialization method, except that a randomly
    sampled thinking-style string is not used. First-order prompt generation is
    Promptbreeder’s standard asexual mutation operat
    """

    prompt: PromptTemplate = PromptTemplate.from_template(
        "{mutation_prompt}  INSTRUCTION: {task_prompt}  INSTRUCTION MUTATNT: "
    )
