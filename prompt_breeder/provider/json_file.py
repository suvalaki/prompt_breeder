from typing import Any, Callable, List
import os
import json
import random
from pydantic import BaseModel, ConfigDict
from prompt_breeder.types import Population, UnitOfEvolution
from prompt_breeder.prompts.string import (
    StringTaskPrompt,
    StringProblemDescription,
    StringMutationPrompt,
)


class JsonListLoad(BaseModel):
    factory: Callable[[str], Any]
    data: List[Any] = []
    repeating: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def load(self, fp: str | bytes | os.PathLike):
        self.index = 0
        with open(fp, "r") as f:
            self.data = [self.factory(x) for x in json.load(f)]
        return self

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            if self.repeating:
                self.index = 0
                result = self.data[self.index]
                self.index += 1
                return result
            raise StopIteration


class RandomJsonListLoad(JsonListLoad):
    def __iter__(self):
        return self

    def __next__(self):
        return random.choice(self.data)


def load_string_population(fp: str) -> Population:
    d = json.load(open(fp, "r"))
    units = [
        UnitOfEvolution(
            problem_description=StringProblemDescription(
                text=u["problem_description"]["text"]
            ),
            task_prompt_set=[
                StringTaskPrompt(text=t["text"]) for t in u["task_prompt_set"]
            ],
            mutation_prompt=StringMutationPrompt(text=u["mutation_prompt"]["text"]),
            elites=[StringTaskPrompt(text=t["text"]) for t in u["elites"]],
        )
        for u in d["members"]
    ]
    pop = Population(members=units, age=d["age"])
    return pop
