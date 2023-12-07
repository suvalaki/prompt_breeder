import os
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain.storage import InMemoryStore
from langchain.embeddings import OllamaEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from prompt_breeder.provider.json_file import load_string_population
from experiments.gsm8k.define_experiment import create_experiment


if __name__ == "__main__":
    from langchain.chat_models.ollama import ChatOllama

    fp_population: str = "./population.json"

    llm_cache = InMemoryCache()
    set_llm_cache(llm_cache)
    store = InMemoryStore()

    cached_llm = ChatOllama(model="mistral", temperature=0.0, cache=True)
    llm = ChatOllama(model="mistral", temperature=1.0, cache=False)
    underlying_embeddings = OllamaEmbeddings(model="mistral")
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
        samples=25,
        fp_population=fp_population,
    )

    if os.path.exists(fp_population):
        initial_population = load_string_population(fp_population)

    # Run the algorithm
    else:
        initial_population = pop_initializer.initialize(
            problem_description="Solve the math word problem"
        )

        evolution.post_step_callback(initial_population)

    final_population = evolution.run(
        {"population": initial_population, "generations": 500}
    )
