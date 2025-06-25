from unittest.mock import MagicMock, patch

import pytest

from llm_eval.tools.model_tools import (
    cache_required_models,
    get_azure_ai_evaluation_model_config,
    get_azure_openai_embedding_model,
    get_azure_openai_llm,
    get_azure_openai_llm_inference,
    preload_huggingface_model,
)


@pytest.fixture
def fake_model_name():
    return "test/test-model"


@pytest.fixture
def custom_model_config():
    return {"custom_task": {"name": "test/test-model", "revision": "main"}}


@patch("llm_eval.tools.model_tools.snapshot_download")
@patch("llm_eval.tools.model_tools.AutoConfig.from_pretrained")
@patch("llm_eval.tools.model_tools.AutoTokenizer.from_pretrained")
@patch("llm_eval.tools.model_tools.AutoModel.from_pretrained")
def test_preload_huggingface_model_success(
    mock_model, mock_tokenizer, mock_config, mock_snapshot, fake_model_name
):
    mock_snapshot.return_value = "/tmp/fake-model"
    mock_config.return_value = MagicMock()
    mock_tokenizer.return_value = MagicMock()
    mock_model.return_value = MagicMock()

    # Should not raise
    preload_huggingface_model(model_name=fake_model_name)

    mock_snapshot.assert_called_once()
    mock_config.assert_called_once()
    mock_tokenizer.assert_called()
    mock_model.assert_called()


@patch("llm_eval.tools.model_tools.preload_huggingface_model")
def test_cache_required_models_with_default(mock_preload):
    # Should call preload 3 times for the default _REQUIRED_MODELS
    cache_required_models()
    assert mock_preload.call_count == 3


@patch("llm_eval.tools.model_tools.preload_huggingface_model")
def test_cache_required_models_with_custom_config(mock_preload, custom_model_config):
    cache_required_models(
        use_standard_models=False,
        custom_model_config=custom_model_config,
        custom_cache_dir="/tmp/custom",
    )
    mock_preload.assert_called_once_with(
        model_name="test/test-model", local_dir="/tmp/custom", revision="main"
    )


def test_cache_required_models_raises_if_no_custom_config():
    with pytest.raises(ValueError):
        # This will raise due to `custom_model_config=None` when use_standard_models=False
        cache_required_models(use_standard_models=False)


def test_get_azure_openai_llm_inference():
    response = get_azure_openai_llm_inference("is your response a string")
    assert isinstance(response, str)


def test_get_azure_openai_embedding_model():
    model = get_azure_openai_embedding_model()
    response = model.embed_query("is embeeding a list of floats")
    assert all(isinstance(x, float) for x in response)


def test_get_azure_ai_evaluation_model_config():
    model_config = get_azure_ai_evaluation_model_config()

    model = get_azure_openai_llm(
        model=model_config["azure_deployment"],
        api_key=model_config["api_key"],
        azure_endpoint=model_config["azure_endpoint"],
        api_version=model_config["api_version"],
    )

    response = get_azure_openai_llm_inference(
        prompt="is your response a string", model=model
    )
    assert isinstance(response, str)
