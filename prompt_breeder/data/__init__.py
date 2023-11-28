import json
from importlib import resources as impresources
from prompt_breeder import data

FP_BASE_MUTATION_PROMPTS = impresources.files(data) / "base_mutation_prompts.json"
with FP_BASE_MUTATION_PROMPTS.open(mode="r") as f:
    BASE_MUTATION_PROMPTS = json.load(f)

FP_BASE_THINKING_STYLES = impresources.files(data) / "base_thinking_styles.json"
with FP_BASE_MUTATION_PROMPTS.open(mode="r") as f:
    BASE_THINKING_STYLES = json.load(f)
