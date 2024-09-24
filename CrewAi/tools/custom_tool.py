# tools/custom_tool.py
def interpret_gpt4_response(response):
    """
    Interprets the GPT-4 response and maps it to a sentiment label.
    
    Args:
        response (str): The response from GPT-4 as a string.
    
    Returns:
        int: 1 for positive, 0 for negative, None for unclear.
    """
    if 'positive' in response.lower():
        return 1
    elif 'negative' in response.lower():
        return 0
    else:
        return None

def log_result(sentence, true_label, gpt_label, gpt_response):
    """
    Logs the result of a single sentiment comparison between GPT-4 and SST-2.
    
    Args:
        sentence (str): The input sentence.
        true_label (int): The actual sentiment label from SST-2 (0 = negative, 1 = positive).
        gpt_label (int): The predicted sentiment label from GPT-4.
        gpt_response (str): The full response from GPT-4.
    """
    print(f"Sentence: {sentence}")
    print(f"Actual Label: {'Positive' if true_label == 1 else 'Negative'}")
    print(f"GPT-4 Response: {gpt_response}")
    print(f"GPT-4 Predicted Label: {'Positive' if gpt_label == 1 else 'Negative' if gpt_label == 0 else 'Unclear'}")
    print("--------------------------------------------------")

# tools/__init__.py
# Empty file for module initialization
