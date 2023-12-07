import os
from pydantic import BaseModel


class IncrementAge(BaseModel):
    def __call__(self, population, callbacks=None, **kwargs):
        population.age += 1


class SavePopulation(BaseModel):
    fp: str = "./population.json"

    def __call__(self, population, callbacks=None, **kwargs):
        with open(self.fp, "w") as f:
            f.write(population.model_dump_json())


class SaveEachPopulation(BaseModel):
    fp: str = "./population"
    population_prefix: str = "population_"

    def __call__(self, population, callbacks=None, **kwargs):
        if not os.path.exists(self.fp):
            os.path.makedirs(self.fp)

        pth = os.path.join(
            self.fp,
            self.population_prefix + str(population.age) + ".json",
        )
        print(pth)
        with open(pth, "w") as f:
            f.write(population.model_dump_json())
