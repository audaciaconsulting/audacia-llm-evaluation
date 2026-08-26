import os
from typing import Optional

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from huggingface_hub import snapshot_download
from langchain.chat_models.base import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

REQUIRED_MODELS = {
    "sentiment": {"name": "tabularisai/multilingual-sentiment-analysis"},
    "bias": {
        "name": "valurank/distilroberta-bias",
    },
    "toxicity": {"name": "s-nlp/roberta_toxicity_classifier"},
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
        ImportError: If the optional 'local-models' dependencies are not installed.
    """

    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Local Hugging Face model support requires the optional 'local-models' "
            "dependencies (torch, transformers). Install them with: "
            "`uv sync --extra local-models` (in a clone) or "
            "`uv add \"audacia-llm-evaluation[local-models]\"` (as a dependency)."
        ) from e

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

    MODELS = REQUIRED_MODELS if use_standard_models else custom_model_config

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


GRADER_TEMPERATURE = 0.0
GRADER_MAX_RETRIES = 2
GRADER_TIMEOUT = 120.0
_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


def get_credential(credential: Optional[TokenCredential] = None) -> TokenCredential:
    """Credential for keyless (Entra ID) auth against Azure OpenAI.

    Defaults to `DefaultAzureCredential` — `az login` locally, managed identity when
    deployed. It resolves whichever credential it finds first, so a session signed
    into the wrong tenant fails with "Tenant provided in token does not match
    resource token". Fix that with `az login --tenant <id>`, or pass a pinned
    `AzureCliCredential(tenant_id=...)` to `get_azure_openai_llm`,
    `get_azure_openai_embedding_model` or a `get_ragas_wrapped_azure_*` factory.
    `get_azure_ai_evaluation_model_config` takes no credential — see its docstring.
    """
    return credential if credential is not None else DefaultAzureCredential()


def _entra_token_provider(credential: Optional[TokenCredential] = None):
    """Bearer-token provider for keyless (Entra ID) auth."""
    return get_bearer_token_provider(
        get_credential(credential), _COGNITIVE_SERVICES_SCOPE
    )


def _auth_kwargs(
    api_key: Optional[str], credential: Optional[TokenCredential] = None
) -> dict:
    """Auth kwargs for a langchain Azure OpenAI client: key auth when a key is
    provided, otherwise Entra ID via a bearer-token provider."""
    if api_key:
        return {"api_key": api_key}
    return {"azure_ad_token_provider": _entra_token_provider(credential)}


def _require_env(env_var: str) -> str:
    """Fail naming the variable, rather than as a 401 from the grader mid-run."""
    value = os.getenv(env_var)
    if not value:
        raise ValueError(
            f"{env_var} is not set. This package does not load .env files — see the README."
        )
    return value


def get_azure_ai_evaluation_model_config() -> AzureOpenAIModelConfiguration:
    """Model configuration for the azure-ai-evaluation scorers.

    Takes no `credential`: the configuration is a dict with no slot for one, and
    azure-ai-evaluation builds a `DefaultAzureCredential` of its own.
    """
    config = AzureOpenAIModelConfiguration(
        azure_endpoint=_require_env("LLM_EVAL_LLM_ENDPOINT"),
        azure_deployment=_require_env("LLM_EVAL_LLM_MODEL"),
        api_version=_require_env("LLM_EVAL_LLM_API_VERSION"),
    )

    api_key = os.getenv("LLM_EVAL_LLM_API_KEY")
    if api_key:
        config["api_key"] = api_key

    return config


def get_azure_openai_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    credential: Optional[TokenCredential] = None,
    temperature: float = GRADER_TEMPERATURE,
    max_retries: int = GRADER_MAX_RETRIES,
    timeout: Optional[float] = GRADER_TIMEOUT,
    rate_limiter: Optional[BaseRateLimiter] = None,
    **kwargs,
) -> AzureChatOpenAI:
    """Returns an AzureChatOpenAI client with provided or environment-configured parameters.

    Args:
        model (Optional[str]): Azure OpenAI model deployment name.
        api_key (Optional[str]): Azure OpenAI API key.
        azure_endpoint (Optional[str]): Azure endpoint URL.
        api_version (Optional[str]): API version to use.
        credential (Optional[TokenCredential]): Pinned credential for keyless auth,
            where `DefaultAzureCredential` resolves the wrong one. Ignored when a
            key is used. See `get_credential`.
        temperature (float): Sampling temperature. Defaults to 0, so a grader's
            variance does not land in the scores it reports. Raise it only where
            you want the grader to sample.
        max_retries (int): Attempts per call, backing off exponentially and
            honouring `Retry-After`. Raise it when many evaluations run against a
            rate-limited deployment.
        timeout (Optional[float]): Seconds per call. The client sets no timeout of
            its own, so `None` lets a stalled call hang the run.
        rate_limiter (Optional[BaseRateLimiter]): Paces calls before they are sent,
            rather than retrying after rejection — the better answer where the
            deployment's limit is known. See `langchain_core.rate_limiters`.
        **kwargs: Passed to `AzureChatOpenAI`, such as `seed` or `max_tokens`.

    Returns:
        AzureChatOpenAI: Configured Azure OpenAI chat client.
    """

    model = model or os.getenv("LLM_EVAL_LLM_MODEL")
    api_key = api_key or os.getenv("LLM_EVAL_LLM_API_KEY")
    azure_endpoint = azure_endpoint or os.getenv("LLM_EVAL_LLM_ENDPOINT")
    api_version = api_version or os.getenv("LLM_EVAL_LLM_API_VERSION")

    return AzureChatOpenAI(
        model=model,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
        rate_limiter=rate_limiter,
        **_auth_kwargs(api_key, credential),
        **kwargs,
    )


def get_ragas_wrapped_llm(model: BaseChatModel):
    return LangchainLLMWrapper(model)


def get_ragas_wrapped_azure_openai_llm(
    credential: Optional[TokenCredential] = None, **kwargs
) -> LangchainLLMWrapper:
    """The default grader for the ragas-backed evaluators, ragas-wrapped.

    Arguments are passed to `get_azure_openai_llm`.
    """
    llm = get_azure_openai_llm(credential=credential, **kwargs)
    return get_ragas_wrapped_llm(llm)


def get_azure_openai_embedding_model(
    credential: Optional[TokenCredential] = None,
    max_retries: int = GRADER_MAX_RETRIES,
    timeout: Optional[float] = GRADER_TIMEOUT,
    **kwargs,
) -> AzureOpenAIEmbeddings:
    """Returns an AzureOpenAIEmbeddings client from the environment.

    Args:
        credential (Optional[TokenCredential]): Pinned credential for keyless auth.
            Ignored when `LLM_EVAL_EMBEDDING_MODEL_API_KEY` is set. See
            `get_credential`.
        max_retries (int): Attempts per call. Embedding calls share the
            deployment's rate limit, so raise it for large runs. This client backs
            off on its own schedule, `retry_min_seconds` to `retry_max_seconds`,
            not the chat client's exponential one.
        timeout (Optional[float]): Seconds per call.
        **kwargs: Passed to `AzureOpenAIEmbeddings`, such as `chunk_size`, which
            batches inputs per request and so bounds what each one costs.
    """
    return AzureOpenAIEmbeddings(
        model=os.getenv("LLM_EVAL_EMBEDDING_MODEL"),
        azure_endpoint=os.getenv("LLM_EVAL_EMBEDDING_MODEL_ENDPOINT"),
        api_version=os.getenv("LLM_EVAL_EMBEDDING_MODEL_API_VERSION"),
        max_retries=max_retries,
        timeout=timeout,
        **_auth_kwargs(os.getenv("LLM_EVAL_EMBEDDING_MODEL_API_KEY"), credential),
        **kwargs,
    )


def get_ragas_wrapped_embedding_model(model: AzureOpenAIEmbeddings):
    return LangchainEmbeddingsWrapper(model)


def get_ragas_wrapped_azure_open_ai_embedding_model(
    credential: Optional[TokenCredential] = None, **kwargs
) -> LangchainEmbeddingsWrapper:
    """The default embedding model for the ragas-backed evaluators, ragas-wrapped.

    Arguments are passed to `get_azure_openai_embedding_model`.
    """
    model = get_azure_openai_embedding_model(credential=credential, **kwargs)
    return get_ragas_wrapped_embedding_model(model)


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
