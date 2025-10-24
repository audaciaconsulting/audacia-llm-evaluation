import json
import os
import re
import subprocess

from dotenv import load_dotenv, find_dotenv

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
