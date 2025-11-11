import json
import os
import re
import subprocess
import tempfile

import yaml
from dotenv import load_dotenv, find_dotenv

def mask_pii(config_path: str) -> tuple[str, dict[str, str]]:
    """Mask PII values in a promptfoo configuration file.

    Reads the YAML config, replaces values based on the ``piiMasking`` mapping
    (when present), removes the mapping from the output, writes the masked
    configuration to a temporary file, and returns the path to that file.

    Args:
        config_path: Path to the original promptfoo config YAML file.

    Returns:
        tuple[str, dict[str, str]]: The path to the masked config file and the
        mapping used for masking. Falls back to the original ``config_path``
        and an empty mapping when no ``piiMasking`` block is present.
    """

    with open(config_path, "r", encoding="utf-8") as config_file:
        config_data = yaml.safe_load(config_file) or {}

    if not isinstance(config_data, dict):
        raise ValueError("Promptfoo config must be a mapping at the top level")

    if "piiMasking" not in config_data or config_data["piiMasking"] is None:
        return config_path, {}

    pii_mapping = config_data.pop("piiMasking")

    if not isinstance(pii_mapping, dict):
        raise ValueError("piiMasking must be a dictionary of replacements")

    normalized_mapping = {
        str(target): str(replacement) for target, replacement in pii_mapping.items()
    }

    masked_yaml = yaml.safe_dump(
        config_data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )

    for target, replacement in sorted(
        normalized_mapping.items(), key=lambda item: len(item[0]), reverse=True
    ):
        masked_yaml = masked_yaml.replace(str(target), str(replacement))

    suffix = os.path.splitext(config_path)[1] or ".yaml"
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=suffix, encoding="utf-8"
    ) as temp_file:
        temp_file.write(masked_yaml)
        masked_config_path = temp_file.name

    return masked_config_path, normalized_mapping

def unmask_pii(red_team_config_path: str, pii_mapping: dict[str, str]) -> None:
    """Restore masked PII values in a generated red-team config file."""

    if not pii_mapping:
        return

    if not isinstance(pii_mapping, dict):
        raise ValueError("pii_mapping must be a dictionary of replacements")

    with open(red_team_config_path, "r", encoding="utf-8") as config_file:
        red_team_yaml = config_file.read()

    for original, masked in sorted(
        pii_mapping.items(), key=lambda item: len(item[1]), reverse=True
    ):
        red_team_yaml = red_team_yaml.replace(masked, original)

    updated_yaml = add_masked_entity_use_summary(red_team_yaml, pii_mapping)

    with open(red_team_config_path, "w", encoding="utf-8") as config_file:
        config_file.write(updated_yaml)

def add_masked_entity_use_summary(
    red_team_yaml: str, pii_mapping: dict[str, str]
) -> str:
    """Append a summary of masked entity usage by plugin to the config."""

    if not pii_mapping:
        return red_team_yaml

    marker = "\npiiUseInPrompts:"
    if marker in red_team_yaml:
        red_team_yaml = red_team_yaml.split(marker, 1)[0].rstrip() + "\n"

    pii_use = {key: [] for key in pii_mapping.keys()}

    data = yaml.safe_load(red_team_yaml) or {}
    tests = data.get("tests", []) if isinstance(data, dict) else []

    for test in tests:
        if not isinstance(test, dict):
            continue

        vars_section = test.get("vars")
        prompt_content = (
            vars_section.get("prompt") if isinstance(vars_section, dict) else None
        )

        metadata = test.get("metadata")
        plugin_id = metadata.get("pluginId") if isinstance(metadata, dict) else None

        if not prompt_content or not plugin_id:
            continue

        for entity in pii_use:
            if entity in prompt_content and plugin_id not in pii_use[entity]:
                pii_use[entity].append(plugin_id)

    summary_yaml = yaml.safe_dump(
        {"piiUseInPrompts": pii_use},
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )

    if not red_team_yaml.endswith("\n"):
        red_team_yaml += "\n"

    return f"{red_team_yaml}\n{summary_yaml}"

def load_env_vars():
    """Load environment variables from a `.env` file when available."""
    load_dotenv(find_dotenv())

def extract_env_vars_from_config(config_path: str) -> set:
    """Return environment variables referenced within a config file.

    Args:
        config_path: Path to the configuration file to scan.

    Returns:
        set: Names of environment variables referenced with `${VAR_NAME}`.
    """
    with open(config_path, "r") as f:
        content = f.read()

    # Find all ${VAR_NAME} patterns
    pattern = r"\$\{([A-Z_][A-Z0-9_]*)\}"
    env_vars = set(re.findall(pattern, content))

    return env_vars


def check_env_vars(required_vars: set):
    """Ensure required environment variables are set and non-empty.

    Args:
        required_vars: Collection of environment variable names to validate.

    Raises:
        EnvironmentError: If any variable is missing or contains only
            whitespace.
    """
    missing_vars = []
    empty_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            missing_vars.append(var)
        elif value.strip() == "":
            empty_vars.append(var)

    if missing_vars or empty_vars:
        error_msg = []
        if missing_vars:
            error_msg.append(f"Missing: {', '.join(sorted(missing_vars))}")
        if empty_vars:
            error_msg.append(f"Empty: {', '.join(sorted(empty_vars))}")

        raise EnvironmentError(
            f"Environment variable errors:\n  "
            + "\n  ".join(error_msg)
            + f"\n\nPlease set them in your .env file or export them in your shell."
        )

def extract_and_check_vars(config_path):
    """Validate environment variables referenced by the given config file.

    Args:
        config_path: Path to the configuration file that may contain
            `${VAR}` placeholders.

    Raises:
        EnvironmentError: If `check_env_vars` detects missing or empty values.
    """
    required_vars = extract_env_vars_from_config(config_path)
    if required_vars:
        print(f"Checking environment variables: {', '.join(sorted(required_vars))}")
        check_env_vars(required_vars)

def substitute_env_vars(config_path: str) -> str:
    """Replace `${VAR}` placeholders in a config file using `envsubst`.

    Args:
        config_path: Path to the configuration file to process.

    Returns:
        str: The config content with environment variables expanded.

    Raises:
        subprocess.CalledProcessError: If the `envsubst` invocation fails.
    """
    with open(config_path, "r") as f:
        result = subprocess.run(
            ["envsubst"],
            stdin=f,
            capture_output=True,
            text=True,
            check=True,
        )

    return result.stdout

def mask_api_key_in_json(file_path: str, output_path: str = None):
    """Mask API key strings within a JSON results file.

    Args:
        file_path: Path to the JSON file containing potential API keys.
        output_path: Optional path for writing masked output; defaults to
            overwriting `file_path` when omitted.

    Returns:
        None: Logs masking progress for visibility.

    Raises:
        json.JSONDecodeError: If the target file contains invalid JSON.
        FileNotFoundError: If the target file cannot be located.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Check if apiKey exists in the file at all
        if '"apiKey"' not in content:
            print(f"  No 'apiKey' fields found in {file_path}")
            return

        data = json.loads(content)
        masked_count = 0
        found_keys = []

        def mask_dict(obj, path=""):
            """Recursively traverse JSON-like structures and mask API keys."""
            nonlocal masked_count
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if key == 'apiKey' and isinstance(value, str):
                        found_keys.append(current_path)
                        if len(value) <= 4:
                            obj[key] = 'x' * len(value)
                        else:
                            obj[key] = 'x' * (len(value) - 4) + value[-4:]
                        masked_count += 1
                        print(f"  Masked apiKey at {current_path}: ...{obj[key][-4:]}")
                    elif isinstance(value, (dict, list)):
                        mask_dict(value, current_path)
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    mask_dict(item, f"{path}[{idx}]")

        mask_dict(data)

        if masked_count > 0:
            with open(output_path if output_path else file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Total API keys masked: {masked_count}")
        else:
            print(f"  Warning: Found 'apiKey' in file but couldn't mask any values")
            print(f"  Paths checked: {found_keys if found_keys else 'None found'}")

    except json.JSONDecodeError as e:
        print(f"  Warning: Could not parse JSON file: {e}")
    except FileNotFoundError:
        print(f"  Warning: File not found: {file_path}")
    except Exception as e:
        print(f"  Warning: Error masking API keys: {e}")
