import os
import subprocess
import argparse

from llm_eval.red_teaming.promptfoo_utils import mask_pii, unmask_pii


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
