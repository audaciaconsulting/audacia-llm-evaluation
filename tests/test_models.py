from llm_eval.models import get_azure_ai_evaluation_model_config, get_azure_openai_embedding_model, get_azure_openai_llm, get_azure_openai_llm_inference


def test_get_azure_openai_llm_inference():
    response = get_azure_openai_llm_inference('is your response a string')
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

    response = get_azure_openai_llm_inference(prompt="is your response a string", model=model)
    assert isinstance(response, str)
