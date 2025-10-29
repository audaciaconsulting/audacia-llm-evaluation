import textwrap
from pathlib import Path

import pytest
import yaml

from llm_eval.red_teaming.promptfoo_generate import (
    add_masked_entity_use_summary,
    mask_pii,
    unmask_pii,
)


def test_mask_pii_without_mapping_returns_original(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("description: Example\n")

    result_path, mapping = mask_pii(str(config_file))

    assert result_path == str(config_file)
    assert mapping == {}


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

    masked_path, mapping = mask_pii(str(config_file))

    assert masked_path != str(config_file)
    assert mapping == {"Audacia": "Company A"}
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


def test_unmask_pii_restores_original_values(tmp_path):
    red_team_config = tmp_path / "redteam.yaml"
    red_team_config.write_text("description: Company A tool\n")

    unmask_pii(str(red_team_config), {"Audacia": "Company A"})

    parsed = yaml.safe_load(red_team_config.read_text())
    assert parsed["description"] == "Audacia tool"
    assert parsed["piiMaskingSummary"] == {"Audacia": []}


def test_unmask_pii_no_mapping_noop(tmp_path):
    red_team_config = tmp_path / "redteam.yaml"
    original_content = "description: Example\n"
    red_team_config.write_text(original_content)

    unmask_pii(str(red_team_config), {})

    assert red_team_config.read_text() == original_content


def test_add_masked_entity_use_summary_appends_plugin_usage():
    red_team_yaml = (
        "tests:\n"
        "  - vars:\n"
        "      prompt: Audacia collaborates with Alan Kerby\n"
        "    metadata:\n"
        "      pluginId: pii:example\n"
        "  - vars:\n"
        "      prompt: Alan Kerby leads delivery at Audacia\n"
        "    metadata:\n"
        "      pluginId: pii:db\n"
    )

    result = add_masked_entity_use_summary(
        red_team_yaml,
        {"Audacia": "Company A", "Alan Kerby": "Person B"},
    )

    parsed = yaml.safe_load(result)
    summary = parsed["piiMaskingSummary"]

    assert summary == {
        "Audacia": ["pii:example", "pii:db"],
        "Alan Kerby": ["pii:example", "pii:db"],
    }
