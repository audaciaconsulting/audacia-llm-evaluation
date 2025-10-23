import json
import os
import re
import subprocess

from dotenv import load_dotenv, find_dotenv

def load_env_vars():
    """Load environment variables from .env file if it exists"""
    load_dotenv(find_dotenv())

def extract_env_vars_from_config(config_path: str) -> set:
    """Extract all environment variable references from config file"""
    with open(config_path, "r") as f:
        content = f.read()

    # Find all ${VAR_NAME} patterns
    pattern = r"\$\{([A-Z_][A-Z0-9_]*)\}"
    env_vars = set(re.findall(pattern, content))

    return env_vars


def check_env_vars(required_vars: set):
    """Check that all required environment variables are set and not empty"""
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
    required_vars = extract_env_vars_from_config(config_path)
    if required_vars:
        print(f"Checking environment variables: {', '.join(sorted(required_vars))}")
        check_env_vars(required_vars)

def substitute_env_vars(config_path: str) -> str:
    """Return config content with environment variables substituted."""
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
    """Mask API keys in JSON results file."""
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