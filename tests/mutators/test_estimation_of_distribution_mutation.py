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
from prompt_breeder.mutators.estimation_of_distribution_mutation import (
    EstimationOfDistributionMutation,
)


def test_same_string_is_filtered():
    llm = Ollama(model="mistral", temperature=1.0)
    embed_model = OllamaEmbeddings(
        model="mistral",
    )
    prompt0 = StringTaskPrompt(text="Solve the math word problem.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
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
    scorer = load_evaluator(
        "embedding_distance",
        distance_metric=EmbeddingDistance.EUCLIDEAN,
        embeddings=embed_model,
    )
    _ = scorer.evaluate_strings(
        prediction=str(prompt0),
        reference=str(prompt1),
    )
    mutator = EstimationOfDistributionMutation(
        llm=llm,
        embed_scorer=scorer,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        verbose=1,
    )
    ans = mutator.filter_population([prompt0, prompt1])
    assert len(ans) == 1


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
    scorer = load_evaluator(
        "embedding_distance",
        distance_metric=EmbeddingDistance.COSINE,
        embeddings=embed_model,
    )
    _ = scorer._evaluate_strings(
        prediction=str(prompt0),
        prediction_b=str(prompt1),
    )
    mutator = EstimationOfDistributionMutation(
        llm=llm,
        embed_scorer=scorer,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        verbose=1,
    )
    population = Population(members=[unit])
    _ = mutator.mutate(population, unit)
