import os
import pandas as pd
from pydantic import BaseModel, ConfigDict
from langchain.llms import Ollama
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain.storage import InMemoryStore
from langchain.embeddings import OllamaEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.evaluation.embedding_distance.base import (
    EmbeddingDistance,
    EmbeddingDistanceEvalChain,
)
from prompt_breeder.prompts.string import (
    StringTaskPrompt,
    StringMutationPrompt,
    StringThinkingStyle,
    StringProblemDescription,
)
from prompt_breeder.provider.json_file import RandomJsonListLoad

from prompt_breeder.evolution.fitness import BestMemberFitness
from prompt_breeder.evolution.base import EvolutionExecutor
from prompt_breeder.evolution.binary_tournament import BinaryEvolution
from prompt_breeder.evolution.initialization.base import PopulationInitialization
from prompt_breeder.evolution.initialization.zero_order_random import (
    ZeroOrderInitialization,
)

from prompt_breeder.data import FP_BASE_MUTATION_PROMPTS, FP_BASE_THINKING_STYLES

from prompt_breeder.mutators.zero_order_prompt_generation import (
    ZeroOrderPromptGeneration,
)
from prompt_breeder.mutators.first_order_prompt_generation import (
    FirstOrderPromptGeneration,
)
from prompt_breeder.mutators.estimation_of_distribution_mutation import (
    EstimationOfDistributionMutation,
)
from prompt_breeder.mutators.eda_rank_and_index_mutation import (
    EdaRankAndIndexMutation,
)
from prompt_breeder.mutators.lineage_based_mutation import (
    LineageBasedMutation,
)
from prompt_breeder.mutators.zero_order_hypermutation import (
    ZeroOrderHypermutation,
)
from prompt_breeder.mutators.first_order_hypermutation import (
    FirstOrderHypermutation,
)
from prompt_breeder.mutators.crossover import (
    PromptCrossover,
)
from prompt_breeder.mutators.elite import (
    AddElite,
)

from experiments.gsm8k.fitness import create_gsm8k_fitness


class PostStepEvaluationCallback(BaseModel):
    fitness_scorer: BestMemberFitness
    val_fitness_scorer: BestMemberFitness
    fp: str = "./output.csv"
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __call__(self, population, callbacks=None, **kwargs):
        fitnesses = [
            self.fitness_scorer.score(unit.task_prompt_set)
            for unit in population.members
        ]
        val_fitnesses = [
            self.val_fitness_scorer.score(unit.task_prompt_set)
            for unit in population.members
        ]

        train_metrics = pd.Series(fitnesses).describe()
        val_metrics = pd.Series(val_fitnesses).describe()
        val_metrics.index = ["val_" + n for n in val_metrics.index]
        df = pd.DataFrame(pd.concat([train_metrics, val_metrics])).transpose()
        df.to_csv(self.fp, mode="a", header=not os.path.exists(self.fp))

        print("iteration complete")
        print(df)


def str_task_prompt_factory(x):
    return StringTaskPrompt(text=x)


def str_mutation_prompt_factory(x):
    return StringMutationPrompt(text=x)


def str_thinkingstype_prompt_factory(x):
    return StringThinkingStyle(text=x)


def str_problem_desc_prompt_factory(x):
    return StringProblemDescription(text=x)


def create_experiment(
    cached_llm: Ollama,
    llm: Ollama,
    embed_model: CacheBackedEmbeddings,
    n_members_per_unit: int = 3,
    n_units: int = 20,
    ed_threshold: float = 0.05,
    crossover_prob=0.1,
    num_predict: int = 100,
):
    fitness_scorer = create_gsm8k_fitness(cached_llm)
    val_fitness_scorer = create_gsm8k_fitness(cached_llm, "test")
    multiple_scorer = BestMemberFitness(scorer=fitness_scorer)
    val_multiple_scorer = BestMemberFitness(scorer=val_fitness_scorer)

    step_logger = PostStepEvaluationCallback(
        fitness_scorer=multiple_scorer,
        val_fitness_scorer=val_multiple_scorer,
    )

    thinking_style_provider = RandomJsonListLoad(
        factory=str_task_prompt_factory, repeating=True
    ).load(fp=str(FP_BASE_THINKING_STYLES))
    mutation_prompt_provider = RandomJsonListLoad(
        factory=str_mutation_prompt_factory, repeating=True
    ).load(fp=str(FP_BASE_MUTATION_PROMPTS))

    # Diresct Mutators
    mutator_zero_order_prompt_gen = ZeroOrderPromptGeneration(
        llm=llm,
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        llm_kwargs={"num_predict": num_predict},
    )
    mutator_first_order_prompt_gen = FirstOrderPromptGeneration(
        llm=llm,
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        llm_kwargs={"num_predict": num_predict},
    )

    # Distribution Mutators
    embed_distance_eval = EmbeddingDistanceEvalChain(
        embeddings=embed_model, distance_metric=EmbeddingDistance.COSINE
    )
    mutator_estimation_of_distribution = EstimationOfDistributionMutation(
        llm=llm,
        embed_scorer=embed_distance_eval,
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        threshold=ed_threshold,
        llm_kwargs={"num_predict": num_predict},
    )
    mutator_eda_rank = EdaRankAndIndexMutation(
        llm=llm,
        embed_scorer=embed_distance_eval,
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        threshold=ed_threshold,
        fitness_scorer=fitness_scorer,
        llm_kwargs={"num_predict": num_predict},
    )
    mutator_lineage = LineageBasedMutation(
        llm=llm,
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        llm_kwargs={"num_predict": num_predict},
    )

    # Hypermutations
    mutator_zero_order_hyper = ZeroOrderHypermutation.from_llm(
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        thinking_style_provider=thinking_style_provider,
        llm=llm,
        llm_kwargs={"num_predict": num_predict},
    )
    mutator_first_order_hyper = FirstOrderHypermutation.from_llm(
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        thinking_style_provider=thinking_style_provider,
        llm=llm,
        llm_kwargs={"num_predict": num_predict},
    )

    # Modifiers
    mutator_prompt_corssover = PromptCrossover(
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        fitness_scorer=fitness_scorer,
        probability_of_replacement=crossover_prob,
    )
    mutator_elite = AddElite(
        task_prompt_factory=str_task_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        fitness_scorer=fitness_scorer,
    )
    # Since no COT yet. No context shuffling

    evolution_step = BinaryEvolution(
        fitness_scorer=multiple_scorer,
        pre_step_modifiers=[],
        mutators=[
            mutator_zero_order_prompt_gen,
            mutator_first_order_prompt_gen,
            mutator_estimation_of_distribution,
            mutator_eda_rank,
            mutator_lineage,
            mutator_zero_order_hyper,
            mutator_first_order_hyper,
        ],
        post_step_modifiers=[mutator_prompt_corssover, mutator_elite],
    )
    evolution = EvolutionExecutor(step=evolution_step, post_step_callback=step_logger)

    # Initialize
    initializer = ZeroOrderInitialization.from_llm(
        problem_description_factory=str_problem_desc_prompt_factory,
        mutation_prompt_factory=str_mutation_prompt_factory,
        task_prompt_factory=str_task_prompt_factory,
        thinking_style_provider=thinking_style_provider,
        mutation_prompt_provider=mutation_prompt_provider,
        llm=llm,
        n_members_per_unit=n_members_per_unit,
        llm_kwargs={"num_predict": num_predict},
    )
    pop_initializer = PopulationInitialization(
        initializer=initializer,
        n_units=n_units,
    )

    return pop_initializer, evolution


if __name__ == "__main__":
    llm_cache = InMemoryCache()
    set_llm_cache(llm_cache)
    cached_llm = Ollama(model="mistral", cache=True)
    llm = Ollama(model="mistral", cache=False)
    underlying_embeddings = OllamaEmbeddings(model="mistral")
    store = InMemoryStore()
    embed_model = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model
    )

    pop_initializer, evolution = create_experiment(cached_llm, llm, embed_model)

    # Run the algorithm
    initial_population = pop_initializer.initialize(
        problem_description="Solve the math word problem"
    )

    final_population = evolution.run(
        {"population": initial_population, "generations": 20}
    )
