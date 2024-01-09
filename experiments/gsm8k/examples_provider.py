from datasets import load_dataset  # type: ignore
from itertools import cycle

from prompt_breeder.prompts.string import (
    StringPhenotype,
)


def random_gsmk_iterator(kind: str = "train"):
    for z in cycle(load_dataset("gsm8k", "main").shuffle()[kind]):
        yield StringPhenotype(text=z["answer"])
