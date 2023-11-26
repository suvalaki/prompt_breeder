import pytest  # noqa: F401
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
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


# Lets make a custom fitness that is just the prompt length
class StringLengthFitness(Fitness):
    def score(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return len(str(prompt))


def test_population_sorts_by_fitness():
    llm = Ollama(model="mistral", temperature=1.0)
    embed_model = OllamaEmbeddings(
        model="mistral",
    )
    prompt0 = StringTaskPrompt(text="Solve the math word problem.")
    prompt1 = StringTaskPrompt(
        text="Lets take the time to solve the math word problem.   "
    )
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
        fitness_scorer=StringLengthFitness(),
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
    llm = Ollama(model="mistral", temperature=1.0)
    embed_model = OllamaEmbeddings(
        model="mistral",
    )
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    unit = UnitOfEvolution(
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
    population = Population(members=[unit])
    mutator = EdaRankAndIndexMutation.from_llm(
        llm=llm,
        embeddings=embed_model,
        distance_metric=EmbeddingDistance.COSINE,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=StringLengthFitness(),
        verbose=1,
    )
    ans = mutator.mutate(population, unit)  # noqa: F841
