import textwrap
from pathlib import Path

import pytest

from llm_eval.red_teaming.promptfoo_generate import mask_pii


def test_mask_pii_without_mapping_returns_original(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("description: Example\n")

    result_path = mask_pii(str(config_file))

    assert result_path == str(config_file)


def test_mask_pii_with_mapping_masks_values_and_removes_section(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent(
            """
            description: Audacia internal tool
            prompts:
              - Audacia provides services worldwide
            piiMasking:
              Audacia: Company A
            """
        ).lstrip()
    )

    masked_path = mask_pii(str(config_file))

    assert masked_path != str(config_file)
    masked_content = Path(masked_path).read_text()

    assert "piiMasking" not in masked_content
    assert "Company A" in masked_content
    assert "Audacia" not in masked_content


def test_mask_pii_invalid_mapping_raises(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent(
            """
            description: Example
            piiMasking:
              - invalid
            """
        ).lstrip()
    )

    with pytest.raises(ValueError, match="piiMasking must be a dictionary"):
        mask_pii(str(config_file))
