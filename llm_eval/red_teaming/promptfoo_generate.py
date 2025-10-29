import os
import subprocess
import argparse
import tempfile

import yaml

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


def generate(config_path: str, output_path: str = None):
    """Generate promptfoo red-team tests from a config file.

    Args:
        config_path: Path to the promptfoo configuration YAML file.
        output_path: Optional output path for the generated red-team YAML;
            defaults to `<config>_generated_redteam.yaml` beside the config.

    Returns:
        str: Path to the generated red-team configuration file.

    Raises:
        subprocess.CalledProcessError: If the `promptfoo redteam generate`
            command exits with a non-zero status.
    """
    os.environ["PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION"] = "0"
    os.environ["PROMPTFOO_DISABLE_TELEMETRY"] = "1"

    if not output_path:
        output_path = config_path.replace(".yaml", "_generated_redteam.yaml")

    masked_config_path, pii_mapping = mask_pii(config_path)

    cmd = [
        "npx",
        "promptfoo",
        "redteam",
        "generate",
        "--force",
        "--config",
        masked_config_path,
        "--output",
        output_path,
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    unmask_pii(output_path, pii_mapping)


    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate promptfoo red team tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python script.py config.yaml
  python script.py config.yaml --output custom_output.yaml

Environment variables will be automatically detected from config file.
        """,
    )
    parser.add_argument("config", type=str, help="Path to promptfoo config YAML file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for redteam.yaml (default: redteam.yaml in same directory as config)",
    )

    args = parser.parse_args()

    try:
        output = generate(config_path=args.config, output_path=args.output)
        print(f"\n✓ Generated redteam file: {output}")
    except EnvironmentError as e:
        print(f"\n✗ Environment Error: {e}")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running promptfoo (exit code {e.returncode})")
        exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        exit(1)
