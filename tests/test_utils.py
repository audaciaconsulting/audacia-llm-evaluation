from llm_eval.utils import format_dict_log

def test_format_dict_log():
        a_dict = {"user_input": "When was the first super bowl?",
                  "response": "The first superbowl was held on Jan 15, 1967",
                  "retrieved_contexts": [
                          "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."],
                  "faithfulness": 1.0,
                  "faithfulness_threshold": 0.9,
                  "faithfulness_result": "pass"}
        formatted = format_dict_log(dictionary=a_dict)
        assert formatted == """

****************************************************************************************************

user_input: When was the first super bowl?
response: The first superbowl was held on Jan 15, 1967
retrieved_contexts: ['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles.']
faithfulness: 1.0
faithfulness_threshold: 0.9
faithfulness_result: pass

****************************************************************************************************

"""
