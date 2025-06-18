import os
from typing import Optional

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from transformers import AutoConfig, AutoModel, AutoTokenizer

load_dotenv()

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
            repo_id=model_name,
            local_dir=(
                os.path.join(local_dir, "models--" + model_name.replace("/", "--"))
                if local_dir
                else local_dir
            ),
            revision=revision,
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
        raise ValueError(
            "custom_model_config must be provided if use_standard_models is False."
        )

    MODELS = _REQUIRED_MODELS if use_standard_models else custom_model_config

    for area in MODELS.keys():
        preload_huggingface_model(
            model_name=MODELS[area]["name"],
            local_dir=custom_cache_dir,
            revision=(
                "main"
                if "revision" not in MODELS[area].keys()
                else MODELS[area]["revision"]
            ),
        )


def get_azure_ai_evaluation_model_config():
    return AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL"),
        api_version=os.getenv("AZURE_OPENAI_LLM_API_VERSION"),
    )


def get_azure_openai_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
) -> AzureChatOpenAI:
    """Returns an AzureChatOpenAI client with provided or environment-configured parameters.

    Args:
        model (Optional[str]): Azure OpenAI model deployment name.
        api_key (Optional[str]): Azure OpenAI API key.
        azure_endpoint (Optional[str]): Azure endpoint URL.
        api_version (Optional[str]): API version to use.

    Returns:
        AzureChatOpenAI: Configured Azure OpenAI chat client.
    """

    defaults = {
        "model": os.getenv("AZURE_OPENAI_LLM_MODEL"),
        "api_key": os.getenv("AZURE_OPENAI_LLM_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_LLM_API_VERSION"),
    }

    return AzureChatOpenAI(
        model=model or defaults["model"],
        api_key=api_key or defaults["api_key"],
        azure_endpoint=azure_endpoint or defaults["azure_endpoint"],
        api_version=api_version or defaults["api_version"],
    )


def get_azure_openai_embedding_model():
    openai_embedding = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_ENDPONT"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION"),
    )

    return openai_embedding


def get_ragas_wrapped_embedding_model(model: AzureOpenAIEmbeddings):
    return LangchainEmbeddingsWrapper(model)


def get_azure_openai_llm_inference(
    prompt: str, model: Optional[AzureChatOpenAI] = None
):
    """Invokes the Azure OpenAI model with a given prompt and returns the response content.

    Args:
        prompt (str): The input prompt to send to the model.
        model (Optional[AzureChatOpenAI]): An optional AzureChatOpenAI instance. If not provided,
            a default instance is created using environment configuration.

    Returns:
        str: The content of the model's response.
    """
    model = model or get_azure_openai_llm()
    return model.invoke(prompt).content
