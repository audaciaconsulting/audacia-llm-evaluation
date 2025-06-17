# Audacia LLM Evaluation Tool
Please note this is a work in progress.

# Notes
* Created initial project structure.
* Use conda and the environment.yml to create the enviornment used in this project, so far only has the python version, update as you add functionality.
* Using project toml to set up package building.

# Things to do...
* Add evaluators to `llm_eval/evaluators.py`
* Add tests to `tests`

### Create conda environment
`conda env create -f environment.yaml`

### Update environment.yaml from conda environment
`conda env export --from-history | grep -v '^prefix:' > environment.yaml`
pip packages must be added manually

### Install package for development
`pip install -e .`

