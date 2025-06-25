# Audacia LLM Evaluation Tool
Please note this is a work in progress.

# Notes
* Created initial project structure.
* Use conda and the environment.yml to create the enviornment used in this project, so far only has the python version, update as you add functionality.
* Using project toml to set up package building.

# Things to do...
* Add evaluators to `llm_eval/evaluators.py`
* Add tests to `tests`

## Developement
### Create environment
`pyenv local 3.11.13` 
`uv venv`
`uv sync`

### Update env
`uv add xxx`

### Install package for development
`uv pip install -e .`

