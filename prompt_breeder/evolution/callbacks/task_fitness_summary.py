import os
import logging
import pandas as pd
from pydantic import BaseModel, ConfigDict
from prompt_breeder.evolution.fitness import FitnessScorer


class TaskPromptSummary(BaseModel):
    fitness_scorer: FitnessScorer
    val_fitness_scorer: FitnessScorer
    fp: str = "./detailed_output.csv"
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def _make_single_taskprompt_summary(self, population, member, task, counts):
        return pd.DataFrame.from_records(
            [
                {
                    "age": population.age,
                    "prompt": str(task),
                    "mutation_prompt": str(member.mutation_prompt),
                    "count": counts[str(task)],
                    "fitness": self.fitness_scorer.score(task),
                    "val_fitness": self.val_fitness_scorer.score(task),
                    "examples": self.fitness_scorer.get_all(task),
                    "val_examples": self.val_fitness_scorer.get_all(task),
                }
            ]
        )

    def __call__(self, population, callbacks=None, **kwargs):
        counts = {}
        for member in population.members:
            for task in member.task_prompt_set:
                if str(task) in counts:
                    counts[str(task)] += 1
                else:
                    counts[str(task)] = 1

        prompt_summary = pd.DataFrame()
        for member in population.members:
            for task in member.task_prompt_set:
                prompt_summary = pd.concat(
                    [
                        prompt_summary,
                        self._make_single_taskprompt_summary(
                            population, member, task, counts
                        ),
                    ],
                    axis=0,
                    ignore_index=True,
                )

        # TODO: Use logging here
        # Should probably JSON log here
        logging.info(
            "Task Prompt Diversity Detailed: \n"
            + str(
                prompt_summary.sort_values(by="fitness", ascending=False)[
                    ["age", "prompt", "count", "fitness", "val_fitness"]
                ].drop_duplicates()
            )
        )

        prompt_summary.to_csv(
            self.fp,
            mode="a",
            header=not os.path.exists(self.fp),
            index=False,
        )
