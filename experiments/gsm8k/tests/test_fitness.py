import pytest  # noqa: F401
from langchain.llms import Ollama
from prompt_breeder.evolution.fitness import BestMemberFitness
from prompt_breeder.prompts.string import StringTaskPrompt
from experiments.gsm8k.fitness import NiaveContainsCorrectAnswer

dataset = [
    # TODO: replace this with some dataset ingestion method
    # probably use huggingface datasets
    {"question": "What is 3+ 9?", "answer": "12"},
    {"question": "What is 13+ 5?", "answer": "18"},
]


def test_runs():
    llm = Ollama(model="mistral")
    scorer = NiaveContainsCorrectAnswer(llm=llm, dataset=dataset, verbose=1)
    prompt = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    _ = scorer.score(prompt)


def test_max_fitness_over_pop():
    llm = Ollama(model="mistral")
    scorer = NiaveContainsCorrectAnswer(
        llm=llm, dataset=dataset, data_aggfunc=lambda a, b: sum(b), verbose=1
    )
    multi_scorer = BestMemberFitness(scorer=scorer)
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    _ = multi_scorer.score([prompt0, prompt1])
