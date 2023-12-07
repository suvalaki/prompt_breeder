import os
import logging
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, ConfigDict
from prompt_breeder.evolution.fitness import PopulationFitnessScorer


class UnitFitnessSummary(BaseModel):
    fitness_scorer: PopulationFitnessScorer
    val_fitness_scorer: PopulationFitnessScorer
    fp: str = "./distribution_output.csv"
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __call__(self, population, callbacks=None, **kwargs):
        fitnesses = [
            self.fitness_scorer.score(unit.task_prompt_set)
            for unit in tqdm(population.members, position=2, leave=False)
        ]
        val_fitnesses = [
            self.val_fitness_scorer.score(unit.task_prompt_set)
            for unit in tqdm(population.members, position=2, leave=False)
        ]

        train_metrics = pd.Series(fitnesses).describe()
        val_metrics = pd.Series(val_fitnesses).describe()
        val_metrics.index = ["val_" + n for n in val_metrics.index]
        df = pd.DataFrame(pd.concat([train_metrics, val_metrics])).transpose()
        df.to_csv(self.fp, mode="a", header=not os.path.exists(self.fp))

        logging.info("Population Unit distribution summary: " + str(df))
