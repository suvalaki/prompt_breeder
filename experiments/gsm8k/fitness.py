from typing import Dict, List, Callable, Any
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from prompt_breeder.prompts.string import StringTaskPrompt


class NiaveContainsCorrectAnswer(LLMChain):
    prompt: PromptTemplate = PromptTemplate.from_template("{task_prompt} {query}")
    dataset: List[Dict[str, str]]
    data_aggfunc: Callable[[List[Any], List[float]], float] = lambda ds, fs: sum(
        fs
    ) / len(ds)

    @staticmethod
    def _answer_isin_completion(answer: str, completion: str) -> bool:
        return answer in completion

    def _score_one(
        self, prompt: StringTaskPrompt, datum: Dict[str, str], **kwargs
    ) -> float:
        return float(
            self._answer_isin_completion(
                datum["answer"],
                self.run(
                    {"task_prompt": str(prompt), "query": datum["question"]}, **kwargs
                ),
            )
        )

    async def _ascore_one(
        self, prompt: StringTaskPrompt, datum: Dict[str, str], **kwargs
    ) -> float:
        return float(
            self._answer_isin_completion(
                datum["answer"],
                await self.arun(
                    {"task_prompt": str(prompt), "query": datum["question"]}, **kwargs
                ),
            )
        )

    def score(self, prompt: StringTaskPrompt, **kwargs) -> float:
        return self.data_aggfunc(
            self.dataset, [self._score_one(prompt, d) for d in self.dataset]
        )

    async def ascore(self, prompt: StringTaskPrompt, **kwargs) -> float:
        return self.data_aggfunc(
            self.dataset, [await self._ascore_one(prompt, d) for d in self.dataset]
        )


def create_gsm8k_fitness(llm, kind: str = "train"):
    from langchain.output_parsers.regex import RegexParser
    from datasets import load_dataset  # type: ignore

    dataset = load_dataset("gsm8k", "main")
    data = dataset[kind][0:10]
    parser = RegexParser(regex=r"### (.*)", output_keys=["output"])
    data = [
        {"question": a, "answer": parser.parse(b)["output"]}
        for (a, b) in zip(data["question"], data["answer"])
    ]
    return NiaveContainsCorrectAnswer(llm=llm, dataset=data)
