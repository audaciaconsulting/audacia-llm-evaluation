import argparse
import contextlib
import os
import subprocess
from pathlib import Path

from llm_eval.red_teaming.promptfoo_utils import (
    load_env_vars,
    extract_and_check_vars,
    substitute_env_vars,
    mask_api_key_in_json,
)


def evaluate(config_path: str, output_path: str = None):
    """Run promptfoo red-team evaluation with a config file.

    Args:
        config_path: Path to the promptfoo configuration file to evaluate.
        output_path: Optional path for the JSON results file; defaults to
            `<config_name>_results.json` beside the config.

    Returns:
        str: Absolute path to the results JSON output.

    Raises:
        FileNotFoundError: If `config_path` does not exist.
    """
    config_path = Path(config_path).expanduser()

    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")

    config_path = config_path.resolve()

    os.environ["PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION"] = "1"
    os.environ["PROMPTFOO_DISABLE_TELEMETRY"] = "1"

    # Load env vars from .env file
    load_env_vars()

    # Extract and check required env vars from config
    extract_and_check_vars(str(config_path))

    substituted_config = substitute_env_vars(str(config_path))
    config_dir = config_path.parent

    if output_path:
        output_path = Path(output_path)
    else:
        output_path = config_path.with_name(f"{config_path.stem}_results.json")

    output_path = output_path.resolve()

    resolved_config_path = config_dir / f"{config_path.stem}_resolved{config_path.suffix}"

    try:
        resolved_config_path.write_text(substituted_config)

        # Run the promptfoo eval command with the resolved config in-place
        cmd = [
            "npx",
            "promptfoo",
            "redteam",
            "eval",
            "--config",
            str(resolved_config_path),
            "--output",
            str(output_path),
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(config_dir),
        )

        # Print stdout/stderr for visibility
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

    finally:
        with contextlib.suppress(FileNotFoundError):
            resolved_config_path.unlink()

    if output_path.exists():
        print(f"\nMasking API keys in {output_path}...")
        mask_api_key_in_json(str(output_path))

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate promptfoo red team tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python script.py redteam.yaml
  python script.py redteam.yaml --output custom_results.json

Environment variables will be automatically detected from config file.
        """,
    )
    parser.add_argument("config", type=str, help="Path to promptfoo redteam config YAML file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results.json (default: `<config_name>_results.json` in same directory as config)",
    )

    args = parser.parse_args()

    try:
        output = evaluate(config_path=args.config, output_path=args.output)
        print(f"\n✓ Evaluation complete: {output}")
    except EnvironmentError as e:
        print(f"\n✗ Environment Error: {e}")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running promptfoo (exit code {e.returncode})")
        exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        exit(1)
