import pytest  # noqa: F401
from typing import Dict
from langchain.evaluation import load_evaluator
from langchain.evaluation.embedding_distance.base import (
    EmbeddingDistance,
)

from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
)
from prompt_breeder.evolution.fitness import Fitness
from prompt_breeder.mutators.eda_rank_and_index_mutation import (
    EdaRankAndIndexMutation,
)

from tests.mutators.test_estimation_of_distribution_mutation import (
    KEY0,
    KEY1,
    MockFixedEmbeddings,
    MockLLM,
)


FIXED_SCORES: Dict[str, float] = {
    KEY0: 1.0,
    KEY1: 2.0,
}


# Lets make a custom fitness that is just the prompt length
class FixedScoreFitness(Fitness):
    def score(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return int(FIXED_SCORES[str(prompt)])


def test_population_sorts_by_fitness():
    llm = MockLLM()
    embed_model = MockFixedEmbeddings()
    prompt0 = StringTaskPrompt(text=KEY0)
    prompt1 = StringTaskPrompt(text=KEY1)
    unit = UnitOfEvolution(  # noqa: F841
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral."
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )
    embed_scorer = load_evaluator(
        "embedding_distance",
        distance_metric=EmbeddingDistance.COSINE,
        embeddings=embed_model,
    )
    mutator = EdaRankAndIndexMutation(
        llm=llm,
        embed_scorer=embed_scorer,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        threshold=0,
        fitness_scorer=FixedScoreFitness(),
        verbose=1,
    )
    pop = [prompt0, prompt1]
    mutator.sort_population(pop)

    # Prompt 0 and 1 should be sufficiently different (as the threshold is zero)
    # Because promp1 is larger than prompt0 it should have a higher fitness
    assert len(pop) == 2
    assert pop[0] == prompt1
    assert pop[1] == prompt0
    # we have reversed the order


def test_runs_over_unit():
    llm = MockLLM()
    embed_model = MockFixedEmbeddings()
    prompt0 = StringTaskPrompt(text=KEY0)
    prompt1 = StringTaskPrompt(text=KEY1)
    unit = UnitOfEvolution(
        problem_description=StringPrompt(text="ignored by ED mutation"),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="ignored by ED mutation"),
        elites=[],
    )
    population = Population(members=[unit])
    mutator = EdaRankAndIndexMutation.from_llm(
        llm=llm,
        embeddings=embed_model,
        distance_metric=EmbeddingDistance.COSINE,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=FixedScoreFitness(),
        verbose=1,
    )
    _ = mutator.mutate(population, unit)  # noqa: F841
