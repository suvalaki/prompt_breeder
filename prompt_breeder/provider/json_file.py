from typing import Any, Callable, List
import os
import json
import random
from pydantic import BaseModel, ConfigDict


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
