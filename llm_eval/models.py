import os

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer

_REQUIRED_MODELS = {
    "sentiment": {"name": "tabularisai/multilingual-sentiment-analysis"},
    "bias": {
        "name": "d4data/bias-detection-model",
    },
    "toxicity": {"name": "unitary/toxic-bert"},
}


def preload_huggingface_model(
    model_name: str, local_dir: str = None, revision: str = "main"
):
    """
    Downloads and caches a Hugging Face model, tokenizer, and config.
    Automatically uses `from_tf=True` if TensorFlow weights are detected.

    Args:
        model_name (str): Name or path of the model on Hugging Face Hub.
        local_dir (str): Optional path to store the snapshot (overrides default cache).
        revision (str): Branch, tag or commit ID to download (default is 'main').

    Raises:
        EnvironmentError: If model/tokenizer/config loading fails completely.
    """

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        model_path = snapshot_download(
            repo_id=model_name, local_dir=os.path.join(local_dir, "models--" + model_name.replace("/", "--")) if local_dir else local_dir, revision=revision
        )
    except Exception as e:
        raise EnvironmentError(f"Failed to download snapshot for '{model_name}': {e}")

    # Helper: try with fallback if needed
    def try_with_fallback(loader_fn, name: str):
        try:
            return loader_fn(from_tf=False)
        except Exception as e:
            try:
                return loader_fn(from_tf=True)
            except Exception as fallback_e:
                raise EnvironmentError(
                    f"Failed to load {name} for '{model_name}': {e} | Fallback failed: {fallback_e}"
                )

    # Load Config (no from_tf needed here)
    try:
        AutoConfig.from_pretrained(model_path, revision=revision)
    except Exception as e:
        raise EnvironmentError(f"Failed to load config for '{model_name}': {e}")

    # Load Tokenizer
    try:
        AutoTokenizer.from_pretrained(model_path, revision=revision)
    except Exception as e:
        try:
            AutoTokenizer.from_pretrained(model_path, revision=revision, from_tf=True)
        except Exception as fallback_e:
            raise EnvironmentError(
                f"Failed to load tokenizer for '{model_name}': {e} | Fallback failed: {fallback_e}"
            )

    # Load Model: Auto-detect if TF weights exist, then try appropriate option
    def model_loader(from_tf=False):
        return AutoModel.from_pretrained(model_path, revision=revision, from_tf=from_tf)

    try_with_fallback(model_loader, "model")


def cache_required_models(
    use_standard_models: bool = True,
    custom_cache_dir: str = None,
    custom_model_config: dict = None,
):
    """
    Downloads and caches required Hugging Face models.

    Args:
        use_standard_models (bool): If True, uses the default `_REQUIRED_MODELS`.
            If False, `custom_model_config` must be provided.
        custom_cache_dir (str, optional): Directory to store cached models. Defaults to Hugging Face's default cache location.
        custom_model_config (dict, optional): Custom mapping of task keys to model config dicts with 'name' and optional 'revision'.

    Raises:
        ValueError: If `use_standard_models` is False and `custom_model_config` is not provided.
    """
    if not use_standard_models and not custom_model_config:
        raise ValueError("custom_model_config must be provided if use_standard_models is False.")
    
    MODELS = _REQUIRED_MODELS if use_standard_models else custom_model_config
    
    for area in MODELS.keys():
        preload_huggingface_model(
            model_name=MODELS[area]["name"],
            local_dir=custom_cache_dir,
            revision="main"
            if "revision" not in MODELS[area].keys()
            else MODELS[area]["revision"],
        )
