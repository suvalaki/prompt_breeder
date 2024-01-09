import asyncio
import os
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain.storage import InMemoryStore
from langchain.embeddings import OllamaEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from prompt_breeder.provider.json_file import load_string_population
from experiments.gsm8k.define_experiment import create_experiment  # noqa: F401


if __name__ == "__main__":
    # from langchain.llms.ollama import Ollama
    from langchain.chat_models.ollama import ChatOllama

    fp_population: str = "./population.json"

    llm_cache = InMemoryCache()
    set_llm_cache(llm_cache)
    store = InMemoryStore()

    cached_llm = ChatOllama(model="mistral", temperature=0.0, cache=True)
    llm = ChatOllama(model="mistral", temperature=1.0, cache=False)
    underlying_embeddings = OllamaEmbeddings(model="mistral")

    # cached_llm = Ollama(model="orca-mini:3b", temperature=0.0, cache=True)
    # llm = Ollama(model="orca-mini:3b", temperature=1.0, cache=False)
    # underlying_embeddings = OllamaEmbeddings(model="orca-mini:3b")

    # cached_llm = Ollama(model="mistral:text", temperature=0.0, cache=True)
    # llm = Ollama(model="mistral:text", temperature=1.0, cache=False)
    # underlying_embeddings = OllamaEmbeddings(model="mistral:text")

    embed_model = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model
    )

    pop_initializer, evolution = create_experiment(
        cached_llm,
        llm,
        embed_model,
        n_members_per_unit=2,
        n_units=10,
        num_predict=200,
        samples=50,
        fp_population=fp_population,
    )

    ASYNC = False

    if os.path.exists(fp_population):
        initial_population = load_string_population(fp_population)

    # Run the algorithm
    else:
        if ASYNC:
            initial_population = asyncio.run(
                pop_initializer.ainitialize(
                    problem_description="Solve the math word problem"
                )
            )
        else:
            initial_population = pop_initializer.initialize(
                problem_description="Solve the math word problem"
            )

        evolution._post_step(initial_population)

    if ASYNC:
        final_population = asyncio.run(
            evolution.arun({"population": initial_population, "generations": 500})
        )
    else:
        final_population = evolution.run(
            {"population": initial_population, "generations": 500}
        )
