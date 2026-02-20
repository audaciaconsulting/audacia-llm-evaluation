The [**Audacia LLM Evaluation Repo**](https://github.com/audaciaconsulting/audacia-llm-evaluation) provides tools to evaluate applications that generate text using AI and use retrieval augmented text generation. The repo is comprised of two main components:
1. **LLM Evaluation Toolkit**: A Python package for evaluating Large Language Model (LLM) outputs using various evaluators.
   - Similarity Scoring
   - RAG
   - Sentiment Scoring
   - Bias Scoring
   - Toxicity Scoring
   - Format Consistency
2. **AI Red Teaming**: A framework for automated red teaming of LLMs using Promptfoo.

The [README.md](https://github.com/audaciaconsulting/audacia-llm-evaluation/blob/main/README.md) has detailed guides on how to use both elements.

The [**Bid Writer Repo**](https://dev.azure.com/audacia/Audacia/_git/Audacia.BidWriter.Chatbot) has been used to demonstrate the tools in the LLM Evaluation Repo. 
1. Code to inference the app
2. Tests that run evals can be found in the [example_llm_eval_data_science dir](https://dev.azure.com/audacia/Audacia/_git/Audacia.BidWriter.Chatbot?path=/example_llm_eval_data_science)
3. Example Promptfoo configs can be found in [ai_red_teaming dir](https://dev.azure.com/audacia/Audacia/_git/Audacia.BidWriter.Chatbot?path=/example_llm_eval_data_science/ai_red_teaming)
4. [README.md](https://dev.azure.com/audacia/Audacia/_git/Audacia.BidWriter.Chatbot?path=/example_llm_eval_data_science/README.md)