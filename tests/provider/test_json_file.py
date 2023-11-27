import pytest  # noqa: F401
import json
from prompt_breeder.prompts.string import (
    StringTaskPrompt,
)
from prompt_breeder.provider.json_file import JsonListLoad, RandomJsonListLoad

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
