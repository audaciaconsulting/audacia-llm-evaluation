from pathlib import Path

import pytest

from llm_eval.red_teaming.promptfoo_evaluate import evaluate

@pytest.mark.skip(reason="debugging") # comment out to run
def test_evaluate():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    CONFIG_PATH = REPO_ROOT / "llm_eval" / "red_teaming" / "promptfoo_default_config_generated_redteam.yaml"
    evaluate(str(CONFIG_PATH))