import json
import os

import pytest

from llm_eval.red_teaming.promptfoo_utils import extract_env_vars_from_config, check_env_vars, extract_and_check_vars, \
    substitute_env_vars, mask_api_key_in_json


def test_extract_env_vars_from_config(tmp_path):
    """Test extraction of env vars from config"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
        apiKey: ${API_KEY}
        host: ${API_HOST}
        nested:
          value: ${NESTED_VAR}
        """
    )

    result = extract_env_vars_from_config(str(config_file))

    assert result == {"API_KEY", "API_HOST", "NESTED_VAR"}



def test_check_env_vars_all_set(monkeypatch):
    """Test when all env vars are set"""
    monkeypatch.setenv("VAR1", "value1")
    monkeypatch.setenv("VAR2", "value2")
    check_env_vars({"VAR1", "VAR2"})


def test_check_env_vars_missing(monkeypatch):
    """Test when env vars are missing"""
    monkeypatch.setenv("VAR1", "value1")

    with pytest.raises(EnvironmentError, match="Missing: VAR2"):
        check_env_vars({"VAR1", "VAR2"})


def test_check_env_vars_empty(monkeypatch):
    """Test when env vars are empty"""
    monkeypatch.setenv("VAR1", "")

    with pytest.raises(EnvironmentError, match="Empty: VAR1"):
        check_env_vars({"VAR1"})


def test_check_env_vars_missing_and_empty(monkeypatch):
    """Test when some vars are missing and some empty"""
    monkeypatch.setenv("VAR1", "")

    with pytest.raises(EnvironmentError, match=r"Missing: VAR2[\s\S]*Empty: VAR1"):
        check_env_vars({"VAR1", "VAR2"})


def test_extract_and_check_vars_success(tmp_path, monkeypatch):
    """Test extract_and_check_vars with valid env vars"""
    config = tmp_path / "config.yaml"
    config.write_text("apiKey: ${TEST_VAR}")

    monkeypatch.setenv("TEST_VAR", "value")

    # Should not raise
    extract_and_check_vars(str(config))


def test_substitute_env_vars(tmp_path, monkeypatch):
    """Test substitute_env_vars substitutes correctly"""
    config = tmp_path / "config.yaml"
    config.write_text("apiKey: ${MY_KEY}\nhost: ${MY_HOST}")

    monkeypatch.setenv("MY_KEY", "secret-key")
    monkeypatch.setenv("MY_HOST", "example.com")

    temp_path = substitute_env_vars(str(config))

    try:
        with open(temp_path) as f:
            content = f.read()

        assert "apiKey: secret-key" in content
        assert "host: example.com" in content
        assert "${MY_KEY}" not in content
        assert "${MY_HOST}" not in content
    finally:
        # Cleanup
        os.unlink(temp_path)


def test_mask_api_key_in_json(tmp_path):
    """Test that API keys are properly masked in JSON file."""
    # Create test JSON with API keys
    test_data = {
        "config": {
            "apiKey": "123456789abcd",
            "other": "value"
        },
        "nested": {
            "provider": {
                "apiKey": "123456789efgh"
            }
        }
    }

    # Write to temp file
    test_file = tmp_path / "test.json"
    test_file.write_text(json.dumps(test_data, indent=2))

    # Run masking function
    mask_api_key_in_json(str(test_file))

    # Read back and verify
    result = json.loads(test_file.read_text())

    assert result["config"]["apiKey"] == "xxxxxxxxxabcd"
    assert result["nested"]["provider"]["apiKey"] == "xxxxxxxxxefgh"
    assert result["config"]["other"] == "value"


def test_no_api_keys(tmp_path):
    """Test file with no API keys."""
    test_data = {"config": {"setting": "value"}}

    test_file = tmp_path / "test.json"
    test_file.write_text(json.dumps(test_data))

    mask_api_key_in_json(str(test_file))

    # Should not raise error, file unchanged
    result = json.loads(test_file.read_text())
    assert result == test_data
