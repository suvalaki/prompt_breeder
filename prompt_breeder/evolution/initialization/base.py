from typing import List, Any, Dict, Callable
from langchain.chains.base import Chain
from prompt_breeder.types import ProblemDescription, UnitOfEvolution, Population


class UnitInitialization(Chain):
    problem_description_factory: Callable[[str], ProblemDescription]
    n_members_per_unit: int
    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return ["problem_description"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def initialize(self, problem_description: str, **kwargs) -> UnitOfEvolution:
        return self.run({"problem_description": problem_description}, **kwargs)


class PopulationInitialization(Chain):
    initializer: UnitInitialization
    n_units: int
    output_key: str = "output"

    @property
    def input_keys(self) -> List[str]:
        return ["problem_description"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def initialize(self, problem_description: str, **kwargs) -> Population:
        return self.run({"problem_description": problem_description}, **kwargs)

    def _call(self, inputs: Dict[str, Any], run_manager=None, **kwargs):
        return {
            self.output_key: Population(
                members=[
                    self.initializer.initialize(
                        inputs["problem_description"],
                        callbacks=run_manager.get_child() if run_manager else None,
                        **kwargs
                    )
                    for i in range(self.n_units)
                ]
            )
        }
