import os
import datetime
import tensorflow as tf
from tqdm import tqdm
from pydantic import BaseModel, ConfigDict
from prompt_breeder.evolution.fitness import FitnessScorer


class TensorboardUnitFitness(BaseModel):
    fitness_scorer: FitnessScorer
    val_fitness_scorer: FitnessScorer
    fp: str = "./distribution_output.csv"
    train_summary_writer: tf.summary.SummaryWriter
    test_summary_writer: tf.summary.SummaryWriter
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @classmethod
    def from_fp(
        cls,
        fitness_scorer: FitnessScorer,
        val_fitness_scorer: FitnessScorer,
        fp: str = "./results",
    ):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(fp, current_time + "/train")
        test_log_dir = os.path.join(fp, current_time + "/test")
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        return cls(
            fitness_scorer=fitness_scorer,
            val_fitness_scorer=val_fitness_scorer,
            fp=fp,
            train_summary_writer=train_summary_writer,
            test_summary_writer=test_summary_writer,
        )

    def __call__(self, population, callbacks=None, **kwargs):
        fitnesses = [
            self.fitness_scorer.score(unit.task_prompt_set)
            for unit in tqdm(population.members, position=2, leave=False)
        ]
        with self.train_summary_writer.as_default(step=population.age):
            tf.summary.scalar("max_unit_fitness", max(fitnesses))
            tf.summary.scalar("min_unit_fitness", min(fitnesses))
            tf.summary.histogram("unit_fitness", fitnesses)

        val_fitnesses = [
            self.val_fitness_scorer.score(unit.task_prompt_set)
            for unit in tqdm(population.members, position=2, leave=False)
        ]
        with self.test_summary_writer.as_default(step=population.age):
            tf.summary.scalar("max_unit_fitness", max(val_fitnesses))
            tf.summary.scalar("min_unit_fitness", min(val_fitnesses))
            tf.summary.histogram("unit_fitness", val_fitnesses)
