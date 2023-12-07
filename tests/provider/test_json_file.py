import pytest  # noqa: F401
import json
from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringProblemDescription,
    StringTaskPrompt,
    StringMutationPrompt,
)
from prompt_breeder.provider.json_file import (
    JsonListLoad,
    RandomJsonListLoad,
    load_string_population,
)

data = [
    "this is data0",
    "this is data1",
    "this is data2",
    "this is data3",
]


@pytest.fixture(scope="session")
def data_file(tmp_path_factory):
    fn = tmp_path_factory.mktemp("tmpdata") / "data.json"
    with open(fn, "w") as f:
        json.dump(data, fp=f)
    return fn


def test_loads(data_file):
    loader = JsonListLoad(factory=lambda x: StringTaskPrompt(text=x))
    loader.load(data_file)

    assert len(loader.data) == len(data)
    _ = [x for x in loader]

    # Repeating is infinite
    loader = JsonListLoad(factory=lambda x: StringTaskPrompt(text=x), repeating=True)
    loader.load(data_file)

    for i in range(10):
        next(loader)

    loader = RandomJsonListLoad(
        factory=lambda x: StringTaskPrompt(text=x), repeating=True
    )
    loader.load(data_file)

    result = [next(loader) for i in range(4)]
    print(result)


def create_pop():
    prompt00 = StringTaskPrompt(
        text="one Solve the math word problem, show your workings.     "
    )
    prompt01 = StringTaskPrompt(text="one Solve the math word problem.      ")
    unit0 = UnitOfEvolution(
        problem_description=StringProblemDescription(
            text="Solve the math word problem, giving your answer as an arabic numeral."
        ),
        task_prompt_set=[
            prompt00,
            prompt01,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )

    prompt10 = StringTaskPrompt(
        text="two Solve the math word problem, show your workings."
    )
    prompt11 = StringTaskPrompt(text="two Solve the math word problem.")
    unit1 = UnitOfEvolution(
        problem_description=StringProblemDescription(
            text="Solve the math word problem, giving your answer as an arabic numeral."
        ),
        task_prompt_set=[
            prompt10,
            prompt11,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )

    population = Population(members=[unit0, unit1], age=3)
    return population


def test_string_loads_pop(tmp_path_factory):
    from prompt_breeder.evolution.callbacks.save_population import SavePopulation

    pth = tmp_path_factory.mktemp("data") / "pop.json"
    pop = create_pop()
    saver = SavePopulation(fp=str(pth))
    saver(pop)

    loaded_pop = load_string_population(str(pth))

    assert isinstance(loaded_pop, Population)
    assert len(loaded_pop.members) == 2
    assert loaded_pop.age == 3

    assert pop.members[0] == loaded_pop.members[0]
    assert pop.members[1] == loaded_pop.members[1]
    assert pop == loaded_pop
