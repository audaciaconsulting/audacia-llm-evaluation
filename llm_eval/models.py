from typing import Optional
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os
from ragas.embeddings import LangchainEmbeddingsWrapper
from azure.ai.evaluation import AzureOpenAIModelConfiguration

load_dotenv()


def get_azure_ai_evaluation_model_config():
    return  AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL"),
        api_version=os.getenv("AZURE_OPENAI_LLM_API_VERSION")
    )


def get_azure_openai_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
):
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


def get_azure_openai_llm_inference(prompt: str, model: Optional[AzureChatOpenAI] = None):
    model = model or get_azure_openai_llm()
    return model.invoke(prompt).content
