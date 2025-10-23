import json
import os
import subprocess
import argparse
from llm_eval.red_teaming.promptfoo_utils import load_env_vars, \
    extract_and_check_vars, substitute_env_vars, mask_api_key_in_json


def evaluate(config_path: str, output_path: str = None):
    os.environ["PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION"] = "1"
    os.environ["PROMPTFOO_DISABLE_TELEMETRY"] = "1"

    # Load env vars from .env file
    load_env_vars()

    # Extract and check required env vars from config
    extract_and_check_vars(config_path)

    tmp_path = substitute_env_vars(config_path)

    if not output_path:
        output_path = os.path.join(os.path.dirname(config_path), "results.json")

    try:
        # Run the promptfoo eval command with the temp config
        cmd = [
            "npx",
            "promptfoo",
            "redteam",
            "eval",
            "--config",
            tmp_path,
            "--output",
            output_path,
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print stdout/stderr for visibility
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        result.check_returncode()

    finally:
        # Clean up temp file
        os.unlink(tmp_path)

    if os.path.exists(output_path):
        print(f"\nMasking API keys in {output_path}...")
        mask_api_key_in_json(output_path)

    return output_path


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
        help="Output path for results.json (default: results.json in same directory as config)",
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