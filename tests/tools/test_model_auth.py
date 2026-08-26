"""Unit tests for key vs keyless (Entra ID) auth selection in model_tools.

These are pure logic tests — no network or credentials. They assert that when
an API key is provided it is used, and when it's absent the config falls back to
an Entra ID bearer-token provider. The Azure client classes are mocked so no real
client is constructed.
"""
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from llm_eval.tools import model_tools
from llm_eval.tools.model_tools import (
    _auth_kwargs,
    get_azure_ai_evaluation_model_config,
    get_azure_openai_embedding_model,
    get_azure_openai_llm,
    get_credential,
    get_ragas_wrapped_azure_open_ai_embedding_model,
    get_ragas_wrapped_azure_openai_llm,
)

LLM_ENV = {
    "LLM_EVAL_LLM_MODEL": "gpt-4o",
    "LLM_EVAL_LLM_ENDPOINT": "https://example.openai.azure.com/",
    "LLM_EVAL_LLM_API_VERSION": "2024-10-21",
}

EMBEDDING_ENV = {
    "LLM_EVAL_EMBEDDING_MODEL": "text-embedding-3-large",
    "LLM_EVAL_EMBEDDING_MODEL_ENDPOINT": "https://example.openai.azure.com/",
    "LLM_EVAL_EMBEDDING_MODEL_API_VERSION": "2024-02-01",
}


def _set_env(monkeypatch, base_env, api_key_var, api_key):
    for name, value in base_env.items():
        monkeypatch.setenv(name, value)
    if api_key is None:
        monkeypatch.delenv(api_key_var, raising=False)
    else:
        monkeypatch.setenv(api_key_var, api_key)


# --- _auth_kwargs -----------------------------------------------------------

def test_auth_kwargs_with_key_uses_key():
    assert _auth_kwargs("secret") == {"api_key": "secret"}


@pytest.mark.parametrize("empty", [None, ""])
def test_auth_kwargs_without_key_uses_entra(empty):
    kwargs = _auth_kwargs(empty)
    assert "api_key" not in kwargs
    assert callable(kwargs["azure_ad_token_provider"])


# --- judge model config (dict) ----------------------------------------------

def test_judge_config_includes_key_when_set(monkeypatch):
    _set_env(monkeypatch, LLM_ENV, "LLM_EVAL_LLM_API_KEY", "secret")
    config = get_azure_ai_evaluation_model_config()
    assert config["api_key"] == "secret"


def test_judge_config_omits_key_when_absent(monkeypatch):
    _set_env(monkeypatch, LLM_ENV, "LLM_EVAL_LLM_API_KEY", None)
    config = get_azure_ai_evaluation_model_config()
    assert "api_key" not in config
    assert config["azure_deployment"] == "gpt-4o"


# --- langchain chat client wiring -------------------------------------------

def test_llm_uses_key_when_set(monkeypatch):
    _set_env(monkeypatch, LLM_ENV, "LLM_EVAL_LLM_API_KEY", "secret")
    with patch.object(model_tools, "AzureChatOpenAI") as mock_client:
        get_azure_openai_llm()
    kwargs = mock_client.call_args.kwargs
    assert kwargs["api_key"] == "secret"
    assert "azure_ad_token_provider" not in kwargs


def test_llm_uses_entra_when_no_key(monkeypatch):
    _set_env(monkeypatch, LLM_ENV, "LLM_EVAL_LLM_API_KEY", None)
    with patch.object(model_tools, "AzureChatOpenAI") as mock_client:
        get_azure_openai_llm()
    kwargs = mock_client.call_args.kwargs
    assert "api_key" not in kwargs
    assert callable(kwargs["azure_ad_token_provider"])


# --- langchain embedding client wiring --------------------------------------

def test_embedding_uses_key_when_set(monkeypatch):
    _set_env(monkeypatch, EMBEDDING_ENV, "LLM_EVAL_EMBEDDING_MODEL_API_KEY", "secret")
    with patch.object(model_tools, "AzureOpenAIEmbeddings") as mock_client:
        get_azure_openai_embedding_model()
    kwargs = mock_client.call_args.kwargs
    assert kwargs["api_key"] == "secret"
    assert "azure_ad_token_provider" not in kwargs


def test_embedding_uses_entra_when_no_key(monkeypatch):
    _set_env(monkeypatch, EMBEDDING_ENV, "LLM_EVAL_EMBEDDING_MODEL_API_KEY", None)
    with patch.object(model_tools, "AzureOpenAIEmbeddings") as mock_client:
        get_azure_openai_embedding_model()
    kwargs = mock_client.call_args.kwargs
    assert "api_key" not in kwargs
    assert callable(kwargs["azure_ad_token_provider"])


# --- credential selection ---------------------------------------------------

def test_explicit_credential_is_used_as_given():
    sentinel = object()
    assert get_credential(sentinel) is sentinel


def test_defaults_to_default_azure_credential():
    with patch.object(model_tools, "DefaultAzureCredential") as mock_credential:
        get_credential()
    mock_credential.assert_called_once_with()


# --- required config --------------------------------------------------------

@pytest.mark.parametrize(
    "missing",
    ["LLM_EVAL_LLM_ENDPOINT", "LLM_EVAL_LLM_MODEL", "LLM_EVAL_LLM_API_VERSION"],
)
def test_judge_config_names_the_missing_variable(monkeypatch, missing):
    _set_env(monkeypatch, LLM_ENV, "LLM_EVAL_LLM_API_KEY", None)
    monkeypatch.delenv(missing, raising=False)
    with pytest.raises(ValueError, match=missing):
        get_azure_ai_evaluation_model_config()


# --- pinned credentials reach the client ------------------------------------

# Every public factory that takes a `credential`, with the environment it reads and
# the clients to keep out of the way. `get_credential`'s docstring points consumers
# at these, so the argument has to reach the token provider rather than stopping at
# a private helper.
CREDENTIAL_FACTORIES = [
    (get_azure_openai_llm, LLM_ENV, "LLM_EVAL_LLM_API_KEY", ["AzureChatOpenAI"]),
    (
        get_ragas_wrapped_azure_openai_llm,
        LLM_ENV,
        "LLM_EVAL_LLM_API_KEY",
        ["AzureChatOpenAI", "LangchainLLMWrapper"],
    ),
    (
        get_azure_openai_embedding_model,
        EMBEDDING_ENV,
        "LLM_EVAL_EMBEDDING_MODEL_API_KEY",
        ["AzureOpenAIEmbeddings"],
    ),
    (
        get_ragas_wrapped_azure_open_ai_embedding_model,
        EMBEDDING_ENV,
        "LLM_EVAL_EMBEDDING_MODEL_API_KEY",
        ["AzureOpenAIEmbeddings", "LangchainEmbeddingsWrapper"],
    ),
]


@pytest.fixture
def token_provider():
    """The bearer-token provider, mocked to record the credential it was given."""
    with patch.object(model_tools, "get_bearer_token_provider") as provider:
        yield provider


@pytest.mark.parametrize(
    "factory, env, api_key_var, clients",
    CREDENTIAL_FACTORIES,
    ids=[factory.__name__ for factory, *_ in CREDENTIAL_FACTORIES],
)
def test_a_pinned_credential_reaches_the_token_provider(
    monkeypatch, token_provider, factory, env, api_key_var, clients
):
    """A stand-in for `AzureCliCredential(tenant_id=...)`, as the docstring suggests."""
    _set_env(monkeypatch, env, api_key_var, None)
    credential = object()

    with ExitStack() as patched:
        for client in clients:
            patched.enter_context(patch.object(model_tools, client))
        factory(credential=credential)

    assert token_provider.call_args.args[0] is credential


def test_a_key_takes_precedence_over_a_pinned_credential(monkeypatch, token_provider):
    """Key auth needs no token, so the credential is never resolved."""
    _set_env(monkeypatch, LLM_ENV, "LLM_EVAL_LLM_API_KEY", "secret")

    with patch.object(model_tools, "AzureChatOpenAI") as mock_client:
        get_azure_openai_llm(credential=object())

    assert mock_client.call_args.kwargs["api_key"] == "secret"
    token_provider.assert_not_called()


# --- required config, on the client factories -------------------------------

@pytest.mark.parametrize(
    "missing",
    ["LLM_EVAL_LLM_MODEL", "LLM_EVAL_LLM_ENDPOINT", "LLM_EVAL_LLM_API_VERSION"],
)
def test_llm_names_the_missing_variable(monkeypatch, missing):
    """A missing variable used to reach the client as None, failing as an HTTP error."""
    _set_env(monkeypatch, LLM_ENV, "LLM_EVAL_LLM_API_KEY", None)
    monkeypatch.delenv(missing, raising=False)

    with pytest.raises(ValueError, match=missing):
        get_azure_openai_llm()


@pytest.mark.parametrize("missing", list(EMBEDDING_ENV))
def test_embedding_names_the_missing_variable(monkeypatch, missing):
    _set_env(monkeypatch, EMBEDDING_ENV, "LLM_EVAL_EMBEDDING_MODEL_API_KEY", None)
    monkeypatch.delenv(missing, raising=False)

    with pytest.raises(ValueError, match=missing):
        get_azure_openai_embedding_model()


def test_llm_arguments_stand_in_for_the_environment(monkeypatch):
    """Passing the config explicitly is still enough on its own."""
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LLM_EVAL_LLM_API_KEY", raising=False)

    with patch.object(model_tools, "AzureChatOpenAI") as mock_client:
        get_azure_openai_llm(
            model="gpt-4o",
            azure_endpoint="https://example.openai.azure.com/",
            api_version="2024-10-21",
            api_key="secret",
        )

    kwargs = mock_client.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["azure_endpoint"] == "https://example.openai.azure.com/"
    assert kwargs["api_version"] == "2024-10-21"
